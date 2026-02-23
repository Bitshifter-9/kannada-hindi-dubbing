import shutil
import subprocess
from pathlib import Path

def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def require_cmd(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Missing required command: {name}")

def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)
