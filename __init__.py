import traceback

# Register node mappings directly from the package root so ComfyUI can import
# this folder as a module without depending on other files importing first.
try:
    try:
        # Prefer relative import when this folder is imported as a package
        from .comfyui_invertible_rescale.nodes.irn_downscale import IRNDownscale
    except Exception:
        # Fallback to absolute import for safety if relative fails
        from comfyui_invertible_rescale.nodes.irn_downscale import IRNDownscale
except Exception as e:
    print("[ComfyUI-DownscaleIRN] Failed to import IRNDownscale node:", e)
    traceback.print_exc()

    class IRNDownscale:
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "message": (
                        [
                            "Import error: open the ComfyUI console/log for details.\n"
                            "Common fixes: 1) update repo, 2) correct folder path under custom_nodes,\n"
                            "3) install requirements with the same Python env ComfyUI uses (pip install -r requirements.txt)."
                        ],
                    ),
                }
            }

        RETURN_TYPES = ("STRING",)
        RETURN_NAMES = ("message",)
        FUNCTION = "explain"
        CATEGORY = "load_errors"

        def explain(self, message):
            return (message,)

NODE_CLASS_MAPPINGS = {
    "IRNDownscale": IRNDownscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IRNDownscale": "IRN Downscale",
}