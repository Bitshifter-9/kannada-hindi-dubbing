import argparse
import json
import os
import re
import subprocess
import wave
import numpy as np

from utils.ffmpeg_utils import ensure_parent_dir, require_cmd

def load_json(p):
    if not os.path.exists(p):
        raise RuntimeError("Missing file: " + p)
    f = open(p, "r", encoding="utf-8")
    d = json.load(f)
    f.close()
    return d

def read_wav_i16(p):
    w = wave.open(p, "rb")
    ch = w.getnchannels()
    sr = w.getframerate()
    sw = w.getsampwidth()
    n = w.getnframes()
    raw = w.readframes(n)
    w.close()
    if sw != 2:
        raise RuntimeError("Expected 16-bit wav, got sampwidth=" + str(sw))
    a = np.frombuffer(raw, dtype=np.int16)
    if ch == 2:
        a = a.reshape(-1, 2).mean(axis=1).astype(np.int16)
    elif ch != 1:
        raise RuntimeError("Expected 1 or 2 channels, got " + str(ch))
    return a, sr

def write_wav_i16(p, a, sr):
    ensure_parent_dir(p)
    w = wave.open(p, "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(a.tobytes())
    w.close()

def pick_bin(name):
    if name and name != "auto":
        require_cmd(name)
        return name
    if shutil_which("whisper-cli"):
        return "whisper-cli"
    if shutil_which("whisper-cpp"):
        return "whisper-cpp"
    raise RuntimeError("Missing whisper binary. Install with: brew install whisper-cpp")

def shutil_which(x):
    try:
        import shutil
        return shutil.which(x) is not None
    except Exception:
        return False

def clean_txt(s):
    s = s.replace("\r\n", "\n")
    lines = []
    for line in s.split("\n"):
        t = line.strip()
        if not t:
            continue
        t = re.sub(r"^\[[0-9:. ]+-->[0-9:. ]+\]\s*", "", t)
        lines.append(t)
    out = " ".join(lines)
    out = re.sub(r"\s+", " ", out).strip()
    return out

def run_whisper(bin_name, model, lang, wav, out_base):
    if not os.path.exists(model):
        raise RuntimeError("Missing model file: " + model)
    ensure_parent_dir(out_base)
    cmd = [bin_name, "-m", model, "-f", wav, "-l", lang, "-otxt", "-of", out_base]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError("Whisper failed:\n" + p.stderr.decode("utf-8", errors="replace"))

def read_txt(out_base):
    p1 = out_base + ".txt"
    if os.path.exists(p1):
        f = open(p1, "r", encoding="utf-8", errors="ignore")
        s = f.read()
        f.close()
        return s
    if os.path.exists(out_base):
        f = open(out_base, "r", encoding="utf-8", errors="ignore")
        s = f.read()
        f.close()
        return s
    return ""

def transcribe_segments(wav_path, seg_json, out_json, bin_name, model, lang):
    segs = load_json(seg_json)
    a, sr = read_wav_i16(wav_path)
    b = pick_bin(bin_name)
    out = []
    for seg in segs:
        i = int(seg["id"])
        st = float(seg["start"])
        en = float(seg["end"])
        s0 = int(st * sr)
        s1 = int(en * sr)
        if s1 <= s0:
            out.append({"id": i, "scene": int(seg["scene"]), "start": st, "end": en, "text": ""})
            continue
        wseg = a[s0:s1]
        seg_wav = "data/interim/asr/wav/seg_" + str(i).zfill(4) + ".wav"
        out_base = "data/interim/asr/txt/seg_" + str(i).zfill(4)
        write_wav_i16(seg_wav, wseg, sr)
        if not os.path.exists(out_base + ".txt"):
            run_whisper(b, model, lang, seg_wav, out_base)
        txt = clean_txt(read_txt(out_base))
        out.append({"id": i, "scene": int(seg["scene"]), "start": st, "end": en, "text": txt})
    ensure_parent_dir(out_json)
    f = open(out_json, "w", encoding="utf-8")
    json.dump(out, f, indent=2)
    f.close()
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wav", required=True)
    p.add_argument("--segs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--bin", default="auto")
    p.add_argument("--model", required=True)
    p.add_argument("--lang", default="kn")
    a = p.parse_args()
    r = transcribe_segments(a.wav, a.segs, a.out, a.bin, a.model, a.lang)
    print("ASR items:", len(r))
    print("Output:", a.out)

if __name__ == "__main__":
    main()
