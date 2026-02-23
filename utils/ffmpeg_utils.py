import os
import shutil
import subprocess

def ensure_parent_dir(path):
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
def require_cmd(name):
    if shutil.which(name) is None:
        raise RuntimeError("Missing required command: " + name)
def run(cmd):
    subprocess.run(cmd, check=True)
def run_capture(cmd):
    p = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return p.stdout, p.stderr
