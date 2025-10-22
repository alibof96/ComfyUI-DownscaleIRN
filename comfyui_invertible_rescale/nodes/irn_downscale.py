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

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _download(url: str, dest: str, expected_sha256: Optional[str] = None):
    import urllib.request
    _ensure_dir(os.path.dirname(dest))
    tmp = dest + ".part"
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
            raise RuntimeError(f"Hash mismatch for {url}. expected={{expected_sha256}} got={{got}}")
    os.replace(tmp, dest)

def _load_registry() -> Dict:
    if not os.path.exists(REGISTRY_PATH):
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
        return json.load(f)

def _discover_local_models(models_dir: str) -> Dict[str, Dict]:
    results = {}
    if not os.path.isdir(models_dir):
        return results
    for fn in os.listdir(models_dir):
        if not fn.lower().endswith(('.pt', '.pth')):
            continue
        path = os.path.join(models_dir, fn)
        key = os.path.splitext(fn)[0]
        m = FILENAME_SCALE_RE.search(fn)
        sc = int(m.group(1)) if m else None
        results[key] = {
            "description": f"Local file: {{fn}}",
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

def _load_irn_from_state_dict(model_path: str, device: torch.device, dtype, scale: int) -> nn.Module:
    arch = _build_irn_arch_config(scale)
    sub = subnet('DBNet', init='xavier', gc=32)
    model = InvRescaleNet(
        channel_in=3, channel_out=3, subnet_constructor=sub,
        block_num=arch["block_num"], down_num=arch["down_num"],
        down_first=arch["down_first"], use_ConvDownsampling=arch["use_ConvDownsampling"],
        down_scale=arch["down_scale"]
    )
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
    model.load_state_dict(cleaned, strict=False)
    model.eval()
    model = model.to(device=device, dtype=dtype)
    return _IRNDownscaleWrapper(model)

def _load_model_generic(model_path: str, device: torch.device, dtype, scale: int) -> nn.Module:
    try:
        m = torch.jit.load(model_path, map_location=device)
        m.eval()
        return m.to(device=device, dtype=dtype)
    except Exception:
        pass
    try:
        obj = torch.load(model_path, map_location=device)
        if hasattr(obj, "eval") and isinstance(obj, nn.Module):
            obj.eval()
            return obj.to(device=device, dtype=dtype)
    except Exception:
        pass
    return _load_irn_from_state_dict(model_path, device, dtype, scale)

def _ensure_model_from_registry(model_key: str, registry: Dict) -> Optional[str]:
    entry = registry["models"].get(model_key)
    if not entry or entry.get("source") == "builtin":
        return None
    models_dir = os.path.expanduser(registry.get("models_dir", DEFAULT_MODELS_DIR))
    _ensure_dir(models_dir)
    filename = entry.get("filename") or f"{{model_key}}.pt"
    dest = os.path.join(models_dir, filename)
    if os.path.exists(dest):
        if entry.get("sha256"):
            got = _sha256(dest)
            if got.lower() != entry["sha256"].lower():
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
        _download(url, dest, expected_sha256=entry.get("sha256"))
        return dest
    gdf = registry.get("google_drive_folder_url", "")
    if gdf and download_gdrive_folder:
        download_gdrive_folder(gdf, models_dir)
        if os.path.exists(dest):
            return dest
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
        "description": "Pick IRN_x{{scale}} if present (downloads if enabled), else fallback to builtin.",
        "scale": [2, 3, 4],
        "source": "virtual",
        "_kind": "virtual"
    }
    order = ["builtin_bicubic", "auto_best_match"]
    keys = order + [k for k in sorted(merged.keys()) if k not in order]
    return keys, merged

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
                "models_dir_override": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES = ("IMAGE", "DICT",)
    RETURN_NAMES = ("image", "meta",)
    FUNCTION = "downscale"
    CATEGORY = "image/resize"

    def _maybe_prepare_input(self, image: torch.Tensor, scale: int, mode: str) -> Tuple[torch.Tensor, Optional[Tuple[int,int,int,int]]]:
        x = _to_torch_chw(image)
        if mode == "pad":
            x, pad = _pad_to_multiple(x, scale)
            return x, pad
        if mode == "crop":
            b, c, h, w = x.shape
            h2 = h - (h % scale)
            w2 = w - (w % scale)
            return x[:, :, :h2, :w2].contiguous(), None
        if mode == "resize_to_multiple":
            b, c, h, w = x.shape
            h2 = h - (h % scale)
            w2 = w - (w % scale)
            if h2 == h and w2 == w:
                return x, None
            x = F.interpolate(x, size=(h2, w2), mode="area", antialias=True)
            return x, None
        return x, None

    def _run_model(self, x_bchw: torch.Tensor, model: nn.Module, tile: int, overlap: int) -> torch.Tensor:
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

    def _pick_model_path(self, model_key: str, registry: Dict, scale: int, auto_download_selected: bool, auto_download_all: bool, use_google_drive: bool) -> Tuple[Optional[str], str, str]:
        models_dir = os.path.expanduser(registry.get("models_dir", DEFAULT_MODELS_DIR))
        if auto_download_all:
            for k, v in registry.get("models", {}).items():
                if v.get("source") == "builtin":
                    continue
                if v.get("url"):
                    try:
                        _ensure_model_from_registry(k, registry)
                    except Exception:
                        pass
            if use_google_drive and download_gdrive_folder and registry.get("google_drive_folder_url"):
                try:
                    download_gdrive_folder(registry["google_drive_folder_url"], models_dir)
                except Exception:
                    pass

        keys, merged = _build_model_choices(registry)
        if model_key == "auto_best_match":
            candidate = f"IRN_x{{scale}}"
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

        entry = merged.get(model_key, {})
        kind = entry.get("_kind", "")

        if model_key == "builtin_bicubic":
            return None, "builtin", model_key
        if kind == "local":
            return entry.get("path"), "local", model_key
        if kind == "registry":
            if auto_download_selected:
                path = _ensure_model_from_registry(model_key, registry)
            else:
                filename = entry.get("filename") or f"{{model_key}}.pt"
                path = os.path.join(models_dir, filename)
                path = path if os.path.exists(path) else None
            if not path and use_google_drive and download_gdrive_folder and registry.get("google_drive_folder_url"):
                try:
                    download_gdrive_folder(registry["google_drive_folder_url"], models_dir)
                    filename = entry.get("filename") or f"{{model_key}}.pt"
                    cand = os.path.join(models_dir, filename)
                    if os.path.exists(cand):
                        path = cand
                except Exception:
                    pass
            return path, "registry", model_key
        return None, "unknown", model_key

    def downscale(self, image, model_key, scale, device="auto", precision="fp16",
                  tile_size=0, tile_overlap=16, not_divisible_mode="pad",
                  builtin_mode="bicubic", keep_alpha=True,
                  auto_download_selected=True, auto_download_all=False,
                  use_google_drive=True, models_dir_override=""):
        t0 = time.time()
        registry = _load_registry()
        if models_dir_override.strip():
            registry["models_dir"] = models_dir_override.strip()

        device = _pick_device(device)
        dtype = _pick_dtype(precision)

        has_alpha = (image.shape[-1] == 4) and keep_alpha
        if has_alpha:
            rgb = image[:, :, :, :3]
            alpha = image[:, :, :, 3:4]
        else:
            rgb = image
            alpha = None

        rgb_chw, _ = self._maybe_prepare_input(rgb, scale, not_divisible_mode)
        rgb_chw = rgb_chw.to(device=device, dtype=dtype)

        model_path, kind, resolved_key = self._pick_model_path(
            model_key, registry, scale, auto_download_selected, auto_download_all, use_google_drive
        )

        used_model = "builtin"
        if model_path is None:
            out_rgb_chw = _downscale_builtin(rgb_chw, scale, mode=builtin_mode)
            used_model = "builtin"
            load_error = ""
        else:
            try:
                cache_key = f"{{model_path}}:{{scale}}:{{str(device)}}:{{str(dtype)}}"
                if cache_key not in self._cache:
                    self._cache[cache_key] = _load_model_generic(model_path, device, dtype, scale)
                model = self._cache[cache_key]
                out_rgb_chw = self._run_model(rgb_chw, model, tile_size, tile_overlap)
                used_model = f"{{kind}}:{{os.path.basename(model_path)}}"
                load_error = ""
            except Exception as e:
                out_rgb_chw = _downscale_builtin(rgb_chw, scale, mode=builtin_mode)
                load_error = str(e)
                used_model = f"fallback_builtin_due_to_error"

        out_rgb_bhwc = _to_bhwc(out_rgb_chw.to(dtype=torch.float32, device="cpu")).clamp(0, 1)

        if alpha is not None:
            a_chw, _ = self._maybe_prepare_input(alpha, scale, not_divisible_mode)
            a_chw = a_chw.to(device=device, dtype=dtype)
            out_a_chw = _downscale_builtin(a_chw, scale, mode="area")
            out_a_bhwc = _to_bhwc(out_a_chw.to(dtype=torch.float32, device="cpu")).clamp(0, 1)
            out = torch.cat([out_rgb_bhwc, out_a_bhwc], dim=-1)
        else:
            out = out_rgb_bhwc

        keys, merged = _build_model_choices(registry)
        local_list = []
        models_dir = os.path.expanduser(registry.get("models_dir", DEFAULT_MODELS_DIR))
        for k, v in merged.items():
            if k in ("builtin_bicubic", "auto_best_match"):
                continue
            if v.get("_kind") == "local":
                local_list.append(v.get("filename", k))
            elif v.get("_kind") == "registry":
                fn = v.get("filename", f"{{k}}.pt")
                local_list.append(fn if os.path.exists(os.path.join(models_dir, fn)) else f"{{fn}} (missing)")

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
            "elapsed_sec": round(time.time() - t0, 4),
            "load_error": load_error if 'load_error' in locals() and load_error else ""
        }
        return (out, meta)