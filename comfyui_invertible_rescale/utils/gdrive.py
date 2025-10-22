def download_gdrive_folder(folder_url: str, dest_dir: str):
    """
    Download a public Google Drive folder to dest_dir using gdown.
    Requires: pip install gdown
    """
    import os
    import subprocess
    os.makedirs(dest_dir, exist_ok=True)
    try:
        import gdown  # noqa: F401
    except Exception:
        try:
            subprocess.check_call(["python", "-m", "pip", "install", "gdown", "--quiet"])
        except Exception as e:
            raise RuntimeError(f"gdown is not installed and automatic installation failed: {e}")
    cmd = ["gdown", "--folder", folder_url, "-O", dest_dir, "--no-cookies"]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"gdown folder download failed: {e}")