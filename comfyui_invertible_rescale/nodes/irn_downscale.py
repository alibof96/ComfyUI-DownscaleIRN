import os
import re
import json
import hashlib
import math
import time
from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# Google Drive utility (optional)
try:
    from ..utils.gdrive import download_gdrive_folder
except Exception:
    download_gdrive_folder = None

# IRN architecture (vendored minimal)
from ..irn.Inv_arch import InvRescaleNet
from ..irn.Subnet_constructor import subnet

# ComfyUI passes IMAGE tensors as [B, H, W, C] float32 0..1 CPU by default

DEFAULT_MODELS_DIR = os.path.join(os.path.expanduser("~"), "ComfyUI", "models", "invertible_rescale")
REGISTRY_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_registry.json")

FILENAME_SCALE_RE = re.compile(r"x([2348])(\D|$)", re.IGNORECASE)

def _log(verbose: bool, msg: str):
    if verbose:
        print(f"[DownscaleIRN] {msg}")

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _download(url: str, dest: str, expected_sha256: Optional[str] = None, verbose: bool = False):
    import urllib.request
    _ensure_dir(os.path.dirname(dest))
    tmp = dest + ".part"
    _log(verbose, f"Downloading: {url} -> {dest}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    if expected_sha256:
        got = _sha256(tmp)
        if got.lower() != expected_sha256.lower():
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise RuntimeError(f"Hash mismatch for {url}. expected={expected_sha256} got={got}")
    os.replace(tmp, dest)
    _log(verbose, f"Downloaded OK: {dest}")

def _load_registry(verbose: bool = False) -> Dict:
    if not os.path.exists(REGISTRY_PATH):
        _log(verbose, f"Registry not found, using defaults: {REGISTRY_PATH}")
        return {
            "models_dir": DEFAULT_MODELS_DIR,
            "google_drive_folder_url": "https://drive.google.com/drive/folders/1ym6DvYNQegDrOy_4z733HxrULa1XIN92",
            "models": {
                "builtin_bicubic": {
                    "description": "Use PyTorch interpolate bicubic/area. No download.",
                    "scale": [2, 3, 4],
                    "source": "builtin"
                },
                "IRN_x2": {"description": "IRN x2 (Google Drive auto-download).", "scale": [2], "url": "", "filename": "IRN_x2.pth"},
                "IRN_x3": {"description": "IRN x3 (Google Drive auto-download).", "scale": [3], "url": "", "filename": "IRN_x3.pth"},
                "IRN_x4": {"description": "IRN x4 (Google Drive auto-download).", "scale": [4], "url": "", "filename": "IRN_x4.pth"}
            }
        }
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        reg = json.load(f)
    _log(verbose, f"Loaded registry from {REGISTRY_PATH}")
    return reg

def _discover_local_models(models_dir: str) -> Dict[str, Dict]:
    results = {}
    if not os.path.isdir(models_dir):
        return results
    for fn in os.listdir(models_dir):
        if not fn.lower().endswith((".pt", ".pth")):
            continue
        path = os.path.join(models_dir, fn)
        key = os.path.splitext(fn)[0]
        m = FILENAME_SCALE_RE.search(fn)
        sc = int(m.group(1)) if m else None
        results[key] = {
            "description": f"Local file: {fn}",
            "scale": [sc] if sc else [],
            "source": "local",
            "path": path,
            "filename": fn
        }
    return results

def _to_torch_chw(image_bhwc: torch.Tensor) -> torch.Tensor:
    return image_bhwc.permute(0, 3, 1, 2).contiguous()

def _to_bhwc(image_bchw: torch.Tensor) -> torch.Tensor:
    return image_bchw.permute(0, 2, 3, 1).contiguous()

def _pick_device(device_pref: str) -> torch.device:
    if device_pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_pref == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if device_pref == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _pick_dtype(precision: str):
    p = precision.lower()
    if p == "fp16" and torch.cuda.is_available():
        return torch.float16
    if p == "bf16":
        return torch.bfloat16
    return torch.float32

def _pad_to_multiple(x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    b, c, h, w = x.shape
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    pad = (0, pad_w, 0, pad_h)
    x_pad = F.pad(x, pad, mode="reflect") if (pad_h or pad_w) else x
    return x_pad, pad

def _downscale_builtin(x_bchw: torch.Tensor, scale: int, mode: str = "bicubic") -> torch.Tensor:
    if mode == "area":
        return F.interpolate(x_bchw, scale_factor=1.0 / scale, mode="area", antialias=True)
    h = math.floor(x_bchw.shape[2] / scale)
    w = math.floor(x_bchw.shape[3] / scale)
    return F.interpolate(x_bchw, size=(h, w), mode="bicubic", antialias=True)

def _build_irn_arch_config(scale: int):
    # Reasonable defaults aligning with IRN_x2/x3/x4
    if scale == 2:
        return dict(use_ConvDownsampling=False, down_num=1, down_first=False, block_num=[8], down_scale=2)
    if scale == 3:
        return dict(use_ConvDownsampling=True, down_num=1, down_first=True, block_num=[8], down_scale=3)
    return dict(use_ConvDownsampling=False, down_num=2, down_first=False, block_num=[8, 8], down_scale=4)

class _IRNDownscaleWrapper(nn.Module):
    def __init__(self, core: InvRescaleNet):
        super().__init__()
        self.core = core
    def forward(self, x):
        y = self.core(x, rev=False)
        return y[:, :3, :, :]

def _load_irn_from_state_dict(model_path: str, device: torch.device, dtype, scale: int, verbose: bool = False) -> nn.Module:
    _log(verbose, f"Constructing IRN architecture for scale={scale}")
    arch = _build_irn_arch_config(scale)
    sub = subnet('DBNet', init='xavier', gc=32)
    model = InvRescaleNet(
        channel_in=3, channel_out=3, subnet_constructor=sub,
        block_num=arch["block_num"], down_num=arch["down_num"],
        down_first=arch["down_first"], use_ConvDownsampling=arch["use_ConvDownsampling"],
        down_scale=arch["down_scale"]
    )
    _log(verbose, f"Loading checkpoint: {model_path}")
    ckpt = torch.load(model_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            sd = ckpt["state_dict"]
        elif "params" in ckpt and isinstance(ckpt["params"], dict):
            sd = ckpt["params"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            sd = ckpt["model"]
        elif "netG" in ckpt and isinstance(ckpt["netG"], dict):
            sd = ckpt["netG"]
        else:
            sd = ckpt
    else:
        raise RuntimeError("Unknown IRN checkpoint format")
    cleaned = {}
    for k, v in sd.items():
        if not isinstance(k, str):
            continue
        nk = k
        for prefix in ["module.", "netG.", "model.", "G."]:  
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        cleaned[nk] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    _log(verbose, f"Loaded state_dict (missing={len(missing)}, unexpected={len(unexpected)})")
    model.eval()
    model = model.to(device=device, dtype=dtype)
    return _IRNDownscaleWrapper(model)

def _load_model_generic(model_path: str, device: torch.device, dtype, scale: int, verbose: bool = False) -> nn.Module:
    try:
        _log(verbose, f"Trying torch.jit.load for: {model_path}")
        m = torch.jit.load(model_path, map_location=device)
        m.eval()
        _log(verbose, "Loaded TorchScript model")
        return m.to(device=device, dtype=dtype)
    except Exception as e:
        _log(verbose, f"TorchScript load failed: {e}")
    try:
        _log(verbose, f"Trying torch.load nn.Module for: {model_path}")
        obj = torch.load(model_path, map_location=device)
        if hasattr(obj, "eval") and isinstance(obj, nn.Module):
            obj.eval()
            _log(verbose, "Loaded pickled nn.Module")
            return obj.to(device=device, dtype=dtype)
    except Exception as e:
        _log(verbose, f"Pickled nn.Module load failed: {e}")
    _log(verbose, "Falling back to IRN state_dict loader")
    return _load_irn_from_state_dict(model_path, device, dtype, scale, verbose=verbose)

def _ensure_model_from_registry(model_key: str, registry: Dict, verbose: bool = False) -> Optional[str]:
    entry = registry["models"].get(model_key)
    if not entry or entry.get("source") == "builtin":
        return None
    models_dir = os.path.expanduser(registry.get("models_dir", DEFAULT_MODELS_DIR))
    _ensure_dir(models_dir)
    filename = entry.get("filename") or f"{model_key}.pt"
    dest = os.path.join(models_dir, filename)

    if os.path.exists(dest):
        _log(verbose, f"Found existing model at {dest}")
        if entry.get("sha256"):
            got = _sha256(dest)
            if got.lower() != entry["sha256"].lower():
                _log(verbose, f"SHA256 mismatch for {dest}, re-downloading")
                try:
                    os.remove(dest)
                except OSError:
                    pass
            else:
                return dest
        else:
            return dest

    url = entry.get("url", "")
    if url:
        _download(url, dest, expected_sha256=entry.get("sha256"), verbose=verbose)
        return dest

    gdf = registry.get("google_drive_folder_url", "")
    if gdf and download_gdrive_folder:
        _log(verbose, f"Downloading from Google Drive folder -> {models_dir}")
        download_gdrive_folder(gdf, models_dir)
        if os.path.exists(dest):
            _log(verbose, f"Found downloaded file at {dest}")
            return dest
        else:
            _log(verbose, f"Expected file {filename} not found after GDrive download")
    return None

def _build_model_choices(registry: Dict) -> Tuple[List[str], Dict[str, Dict]]:
    models_dir = os.path.expanduser(registry.get("models_dir", DEFAULT_MODELS_DIR))
    local = _discover_local_models(models_dir)
    merged = {}
    for k, v in registry.get("models", {}).items():
        merged[k] = v.copy()
        merged[k]["_kind"] = "registry"
    for k, v in local.items():
        if k not in merged:
            merged[k] = v
            merged[k]["_kind"] = "local"
    merged["auto_best_match"] = {
        "description": "Pick IRN_x{scale} if present (downloads if enabled), else fallback to builtin.",
        "scale": [2, 3, 4],
        "source": "virtual",
        "_kind": "virtual"
    }
    order = ["builtin_bicubic", "auto_best_match"]
    keys = order + [k for k in sorted(merged.keys()) if k not in order]
    return keys, merged

def _blend_overlap_frames(frames: List[torch.Tensor], overlap: int, verbose: bool = False) -> torch.Tensor:
    """
    Blend overlapping frames using weighted averaging at boundaries.
    
    Args:
        frames: List of frame tensors, each with shape [batch_size, H, W, C]
        overlap: Number of overlapping frames between batches
        verbose: Whether to log blending operations
    
    Returns:
        Combined tensor with overlapping frames blended
    """
    if len(frames) == 0:
        raise ValueError("Cannot blend empty frame list")
    
    if len(frames) == 1:
        return frames[0]
    
    if overlap == 0:
        # No overlap, just concatenate
        return torch.cat(frames, dim=0)
    
    result = []
    for i, batch_frames in enumerate(frames):
        if i == 0:
            # First batch: keep all frames
            result.append(batch_frames)
            _log(verbose, f"Batch {i}: Added all {batch_frames.shape[0]} frames")
        else:
            # Subsequent batches: blend the overlapping region
            prev_overlap_frames = result[-1][-overlap:]
            curr_overlap_frames = batch_frames[:overlap]
            
            # Create blending weights (linear ramp)
            # First overlap frame: more weight to previous, last overlap frame: more weight to current
            blended_overlap = []
            for j in range(overlap):
                weight_curr = (j + 1) / (overlap + 1)
                weight_prev = 1.0 - weight_curr
                blended = weight_prev * prev_overlap_frames[j] + weight_curr * curr_overlap_frames[j]
                blended_overlap.append(blended)
            
            blended_overlap = torch.stack(blended_overlap, dim=0)
            
            # Replace the overlapping frames in result with blended versions
            result[-1] = torch.cat([result[-1][:-overlap], blended_overlap], dim=0)
            
            # Add remaining frames from current batch (skip the overlap)
            if batch_frames.shape[0] > overlap:
                result.append(batch_frames[overlap:])
                _log(verbose, f"Batch {i}: Blended {overlap} overlap frames, added {batch_frames.shape[0] - overlap} new frames")
            else:
                _log(verbose, f"Batch {i}: Blended {overlap} overlap frames (entire batch)")
    
    return torch.cat(result, dim=0)

class IRNDownscale:
    def __init__(self):
        self._cache: Dict[str, nn.Module] = {}

    @classmethod
    def INPUT_TYPES(cls):
        registry = _load_registry()
        keys, _ = _build_model_choices(registry)
        return {
            "required": {
                "image": ("IMAGE",),
                "model_key": (keys, {"default": "auto_best_match"}),
                "scale": ("INT", {"default": 4, "min": 2, "max": 8, "step": 1}),
            },
            "optional": {
                "device": (["auto", "cuda", "cpu", "mps"], {"default": "auto"}),
                "precision": (["fp16", "bf16", "fp32"], {"default": "fp16"}),
                "tile_size": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 16}),
                "tile_overlap": ("INT", {"default": 16, "min": 0, "max": 512, "step": 2}),
                "not_divisible_mode": (["pad", "crop", "resize_to_multiple"], {"default": "pad"}),
                "builtin_mode": (["bicubic", "area"], {"default": "bicubic"}),
                "keep_alpha": ("BOOLEAN", {"default": True}),
                "auto_download_selected": ("BOOLEAN", {"default": True}),
                "auto_download_all": ("BOOLEAN", {"default": False}),
                "use_google_drive": ("BOOLEAN", {"default": True}),
                "models_dir_override": ("STRING", {"default": ""}),
                "verbose": ("BOOLEAN", {"default": True}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 9999, "step": 1}),
                "overlap": ("INT", {"default": 2, "min": 0, "max": 16, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "DICT",)
    RETURN_NAMES = ("image", "meta",)
    FUNCTION = "downscale"
    CATEGORY = "image/resize"

    def _maybe_prepare_input(self, image: torch.Tensor, scale: int, mode: str, verbose: bool) -> Tuple[torch.Tensor, Optional[Tuple[int,int,int,int]]]:
        x = _to_torch_chw(image)
        if mode == "pad":
            x2, pad = _pad_to_multiple(x, scale)
            if pad != (0,0,0,0):
                _log(verbose, f"Pad to multiple of {scale}: pad={pad}")
            return x2, pad
        if mode == "crop":
            b, c, h, w = x.shape
            h2 = h - (h % scale)
            w2 = w - (w % scale)
            _log(verbose, f"Crop to divisible: ({h},{w}) -> ({h2},{w2})")
            return x[:, :, :h2, :w2].contiguous(), None
        if mode == "resize_to_multiple":
            b, c, h, w = x.shape
            h2 = h - (h % scale)
            w2 = w - (w % scale)
            if h2 == h and w2 == w:
                return x, None
            _log(verbose, f"Resize to multiple via area: ({h},{w}) -> ({h2},{w2})")
            x = F.interpolate(x, size=(h2, w2), mode="area", antialias=True)
            return x, None
        return x, None

    def _run_model(self, x_bchw: torch.Tensor, model: nn.Module, tile: int, overlap: int, verbose: bool) -> torch.Tensor:
        def fn(inp):
            with torch.no_grad():
                return model(inp)
        if tile > 0:
            b, c, h, w = x_bchw.shape
            stride = min(tile, max(1, tile - overlap))
            xs = list(range(0, w, stride))
            ys = list(range(0, h, stride))
            if xs[-1] + tile > w:
                xs[-1] = max(0, w - tile)
            if ys[-1] + tile > h:
                ys[-1] = max(0, h - tile)
            _log(verbose, f"Tiled inference: tile={tile} overlap={overlap} stride={stride} grid={len(xs)}x{len(ys)}")
            pieces = {}
            for yy in ys:
                for xx in xs:
                    pieces[(yy, xx)] = fn(x_bchw[:, :, yy:yy + tile, xx:xx + tile])
            any_tile = next(iter(pieces.values()))
            scale_est = x_bchw.shape[-1] / any_tile.shape[-1]
            out_h = math.ceil(h / scale_est)
            out_w = math.ceil(w / scale_est)
            out = x_bchw.new_zeros((b, any_tile.shape[1], out_h, out_w))
            out_tile_h, out_tile_w = any_tile.shape[-2:]
            for yy in ys:
                for xx in xs:
                    oy = int(round(yy / scale_est))
                    ox = int(round(xx / scale_est))
                    out[:, :, oy:oy + out_tile_h, ox:ox + out_tile_w] = pieces[(yy, xx)]
            return out
        return fn(x_bchw)

    def _pick_model_path(self, model_key: str, registry: Dict, scale: int, auto_download_selected: bool, auto_download_all: bool, use_google_drive: bool, verbose: bool) -> Tuple[Optional[str], str, str]:
        models_dir = os.path.expanduser(registry.get("models_dir", DEFAULT_MODELS_DIR))
        if auto_download_all:
            _log(verbose, "auto_download_all=True: attempting to prefetch all registry models")
            for k, v in registry.get("models", {}).items():
                if v.get("source") == "builtin":
                    continue
                if v.get("url"):
                    try:
                        _ensure_model_from_registry(k, registry, verbose=verbose)
                    except Exception as e:
                        _log(verbose, f"Prefetch failed for {k}: {e}")
            if use_google_drive and download_gdrive_folder and registry.get("google_drive_folder_url"):
                try:
                    _log(verbose, f"Prefetch via Google Drive folder -> {models_dir}")
                    download_gdrive_folder(registry["google_drive_folder_url"], models_dir)
                except Exception as e:
                    _log(verbose, f"GDrive prefetch failed: {e}")

        keys, merged = _build_model_choices(registry)
        orig_key = model_key
        if model_key == "auto_best_match":
            candidate = f"IRN_x{scale}"
            if candidate in merged:
                model_key = candidate
            else:
                for k in merged:
                    if k == "builtin_bicubic":
                        continue
                    m = FILENAME_SCALE_RE.search(k)
                    if m and str(scale) == m.group(1):
                        model_key = k
                        break
        _log(verbose, f"Model key requested='{orig_key}' resolved='{model_key}' scale={scale}")

        entry = merged.get(model_key, {})
        kind = entry.get("_kind", "")

        if model_key == "builtin_bicubic":
            _log(verbose, f"Using builtin downscale mode (no model file).")
            return None, "builtin", model_key
        if kind == "local":
            path = entry.get("path")
            _log(verbose, f"Using local model: {path}")
            return path, "local", model_key
        if kind == "registry":
            if auto_download_selected:
                _log(verbose, f"Ensuring registry model present: {model_key}")
                path = _ensure_model_from_registry(model_key, registry, verbose=verbose)
            else:
                filename = entry.get("filename") or f"{model_key}.pt"
                path = os.path.join(models_dir, filename)
                path = path if os.path.exists(path) else None
                _log(verbose, f"Registry model (no auto-download): looking for {path or '(missing)'}")
            if not path and use_google_drive and download_gdrive_folder and registry.get("google_drive_folder_url"):
                try:
                    _log(verbose, f"Trying Google Drive folder for {model_key} -> {models_dir}")
                    download_gdrive_folder(registry["google_drive_folder_url"], models_dir)
                    filename = entry.get("filename") or f"{model_key}.pt"
                    cand = os.path.join(models_dir, filename)
                    if os.path.exists(cand):
                        path = cand
                        _log(verbose, f"Found model after GDrive: {cand}")
                except Exception as e:
                    _log(verbose, f"GDrive attempt failed for {model_key}: {e}")
            if path:
                _log(verbose, f"Using registry model: {path}")
            else:
                _log(verbose, f"Registry model unavailable; will fallback to builtin.")
            return path, "registry", model_key
        _log(verbose, f"Unknown model key kind='{kind}', falling back to builtin")
        return None, "unknown", model_key

    def downscale(self, image, model_key, scale, device="auto", precision="fp16",
                  tile_size=0, tile_overlap=16, not_divisible_mode="pad",
                  builtin_mode="bicubic", keep_alpha=True,
                  auto_download_selected=True, auto_download_all=False,
                  use_google_drive=True, models_dir_override="",
                  verbose=True, batch_size=1, overlap=2):
        t0 = time.time()
        registry = _load_registry(verbose=verbose)
        if models_dir_override.strip():
            registry["models_dir"] = models_dir_override.strip()
            _log(verbose, f"Override models_dir: {registry['models_dir']}")

        device = _pick_device(device)
        dtype = _pick_dtype(precision)
        _log(verbose, f"Start downscale: scale={scale} device={device} dtype={dtype} "
                      f"tile_size={tile_size} tile_overlap={tile_overlap} "
                      f"mode={not_divisible_mode} builtin={builtin_mode} keep_alpha={keep_alpha}")

        # Batch processing setup
        input_frame_count = image.shape[0]
        _log(verbose, f"Input frame count: {input_frame_count}, batch_size: {batch_size}, overlap: {overlap}")
        
        has_alpha = (image.shape[-1] == 4) and keep_alpha
        if has_alpha:
            _log(verbose, "Alpha channel detected; will downscale alpha with area mode and reattach")
            rgb = image[:, :, :, :3]
            alpha = image[:, :, :, 3:4]
        else:
            rgb = image
            alpha = None

        mp_t0 = time.time()
        model_path, kind, resolved_key = self._pick_model_path(
            model_key, registry, scale, auto_download_selected, auto_download_all, use_google_drive, verbose
        )
        _log(verbose, f"Model selection time: {round(time.time()-mp_t0, 4)}s")

        # Process frames in batches with overlap
        processed_rgb_batches = []
        processed_alpha_batches = []
        
        # Calculate batch indices with overlap
        num_frames = rgb.shape[0]
        batch_indices = []
        i = 0
        while i < num_frames:
            start_idx = max(0, i - overlap) if i > 0 else 0
            end_idx = min(i + batch_size, num_frames)
            batch_indices.append((start_idx, end_idx))
            i += batch_size
            
        _log(verbose, f"Processing {len(batch_indices)} batches with overlap={overlap}")
        
        used_model = "builtin"
        load_error = ""
        
        for batch_num, (start_idx, end_idx) in enumerate(batch_indices):
            _log(verbose, f"Processing batch {batch_num + 1}/{len(batch_indices)}: frames {start_idx}-{end_idx - 1}")
            
            # Extract batch
            rgb_batch = rgb[start_idx:end_idx]
            
            # Prepare input
            rgb_chw, _ = self._maybe_prepare_input(rgb_batch, scale, not_divisible_mode, verbose)
            rgb_chw = rgb_chw.to(device=device, dtype=dtype)
            
            # Process batch
            if model_path is None:
                if batch_num == 0:
                    _log(verbose, f"Executing builtin {builtin_mode} downscale")
                    used_model = "builtin"
                out_rgb_chw = _downscale_builtin(rgb_chw, scale, mode=builtin_mode)
            else:
                try:
                    cache_key = f"{model_path}:{scale}:{str(device)}:{str(dtype)}"
                    if cache_key not in self._cache:
                        _log(verbose, f"Loading model into cache: {cache_key}")
                        mdl_t0 = time.time()
                        self._cache[cache_key] = _load_model_generic(model_path, device, dtype, scale, verbose=verbose)
                        _log(verbose, f"Model loaded in {round(time.time()-mdl_t0, 4)}s")
                    else:
                        if batch_num == 0:
                            _log(verbose, f"Reusing cached model: {cache_key}")
                    model = self._cache[cache_key]
                    inf_t0 = time.time()
                    out_rgb_chw = self._run_model(rgb_chw, model, tile_size, tile_overlap, verbose)
                    _log(verbose, f"Batch {batch_num + 1} inference done in {round(time.time()-inf_t0, 4)}s")
                    if batch_num == 0:
                        used_model = f"{kind}:{os.path.basename(model_path)}"
                except Exception as e:
                    _log(verbose, f"Model execution failed, falling back to builtin: {e}")
                    out_rgb_chw = _downscale_builtin(rgb_chw, scale, mode=builtin_mode)
                    load_error = str(e)
                    if batch_num == 0:
                        used_model = f"fallback_builtin_due_to_error"
            
            out_rgb_bhwc = _to_bhwc(out_rgb_chw.to(dtype=torch.float32, device="cpu")).clamp(0, 1)
            processed_rgb_batches.append(out_rgb_bhwc)
            
            # Process alpha channel if present
            if alpha is not None:
                alpha_batch = alpha[start_idx:end_idx]
                a_chw, _ = self._maybe_prepare_input(alpha_batch, scale, not_divisible_mode, verbose)
                a_chw = a_chw.to(device=device, dtype=dtype)
                out_a_chw = _downscale_builtin(a_chw, scale, mode="area")
                out_a_bhwc = _to_bhwc(out_a_chw.to(dtype=torch.float32, device="cpu")).clamp(0, 1)
                processed_alpha_batches.append(out_a_bhwc)
        
        # Blend overlapping frames
        _log(verbose, "Blending overlapping frames...")
        out_rgb_bhwc = _blend_overlap_frames(processed_rgb_batches, overlap, verbose)
        
        if alpha is not None:
            out_a_bhwc = _blend_overlap_frames(processed_alpha_batches, overlap, verbose)
            out = torch.cat([out_rgb_bhwc, out_a_bhwc], dim=-1)
            _log(verbose, f"Reattached alpha; final shape: {tuple(out.shape)}")
        else:
            out = out_rgb_bhwc
        
        # Integrity check: verify output frame count matches input
        output_frame_count = out.shape[0]
        if output_frame_count != input_frame_count:
            _log(verbose, f"WARNING: Frame count mismatch! Input: {input_frame_count}, Output: {output_frame_count}")
            raise RuntimeError(f"Frame count integrity check failed: input={input_frame_count}, output={output_frame_count}")
        else:
            _log(verbose, f"Integrity check passed: {output_frame_count} frames in/out")

        keys, merged = _build_model_choices(registry)
        local_list = []
        models_dir = os.path.expanduser(registry.get("models_dir", DEFAULT_MODELS_DIR))
        for k, v in merged.items():
            if k in ("builtin_bicubic", "auto_best_match"):
                continue
            if v.get("_kind") == "local":
                local_list.append(v.get("filename", k))
            elif v.get("_kind") == "registry":
                fn = v.get("filename", f"{k}.pt")
                local_list.append(fn if os.path.exists(os.path.join(models_dir, fn)) else f"{fn} (missing)")

        elapsed = round(time.time() - t0, 4)
        _log(verbose, f"Done. used_model={used_model} resolved_key={resolved_key} elapsed={elapsed}s")

        meta = {
            "model_used": used_model,
            "resolved_key": resolved_key,
            "kind": kind,
            "scale": int(scale),
            "device": str(device),
            "precision": str(dtype),
            "tile_size": int(tile_size),
            "tile_overlap": int(tile_overlap),
            "auto_download_selected": bool(auto_download_selected),
            "auto_download_all": bool(auto_download_all),
            "use_google_drive": bool(use_google_drive),
            "available_models": local_list,
            "elapsed_sec": elapsed,
            "load_error": load_error if 'load_error' in locals() and load_error else "",
            "batch_size": int(batch_size),
            "overlap": int(overlap),
            "input_frame_count": int(input_frame_count),
            "output_frame_count": int(output_frame_count),
            "num_batches": len(batch_indices) if 'batch_indices' in locals() else 1
        }
        return (out, meta)