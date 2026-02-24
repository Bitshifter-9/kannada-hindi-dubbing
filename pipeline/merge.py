"""
Merge TTS segments onto the clip video, replacing the original audio track.

Usage:
  python pipeline/merge.py \
    --tts  data/interim/tts/tts.json \
    --clip data/interim/clip/clip.mp4 \
    --out  data/processed/dubbed.mp4 \
    --sr   16000
"""

import argparse
import json
import os
import subprocess
import sys
import struct

p = argparse.ArgumentParser()
p.add_argument("--tts", required=True, help="TTS manifest JSON")
p.add_argument("--clip", required=True, help="Source clip video")
p.add_argument("--out", required=True, help="Output dubbed video")
p.add_argument("--sr", type=int, default=16000, help="Audio sample rate")
p.add_argument("--bg_vol", type=float, default=0.08,
               help="Volume of original audio kept as background (0=mute, 1=full)")
args = p.parse_args()


def get_duration(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return float(r.stdout.decode().strip())


def make_silence_wav(path, dur_s, sr):
    """Write a mono 16-bit silence WAV."""
    n = int(dur_s * sr)
    data = b'\x00\x00' * n
    with open(path, 'wb') as f:
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + len(data)))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<IHHIIHH', 16, 1, 1, sr, sr * 2, 2, 16))
        f.write(b'data')
        f.write(struct.pack('<I', len(data)))
        f.write(data)


# ── Load TTS manifest ──────────────────────────────────────────────────
with open(args.tts, 'r', encoding='utf-8') as f:
    segs = json.load(f)

clip_dur = get_duration(args.clip)
print(f"Clip duration: {clip_dur:.3f}s  |  TTS segments: {len(segs)}")

# ── Build a full-length dubbed audio track ──────────────────────────────
# Strategy: create silence WAV of clip length, then overlay each TTS
# segment at its start time using ffmpeg amix / adelay.

tmp_dir = "data/interim/merge_tmp"
os.makedirs(tmp_dir, exist_ok=True)

# Create base silence track
base_wav = os.path.join(tmp_dir, "base_silence.wav")
make_silence_wav(base_wav, clip_dur, args.sr)

# Build ffmpeg filter to overlay all segments
inputs = ["-i", base_wav]  # input 0 = silence base
filter_parts = []

for idx, seg in enumerate(segs):
    wav_path = seg["wav"]
    start_ms = int(seg["start"] * 1000)
    inp_idx = idx + 1
    inputs.extend(["-i", wav_path])
    # Delay each segment to its start time, pad to fill rest
    filter_parts.append(
        f"[{inp_idx}]adelay={start_ms}|{start_ms},apad=whole_dur={clip_dur}[d{idx}]"
    )

# Mix all delayed segments with the base
mix_inputs = "[0]" + "".join(f"[d{i}]" for i in range(len(segs)))
filter_parts.append(
    f"{mix_inputs}amix=inputs={len(segs) + 1}:duration=first:normalize=0,volume=1.00,alimiter=limit=0.95[out]"
)

filter_str = ";\n".join(filter_parts)

dubbed_audio = os.path.join(tmp_dir, "dubbed_audio.wav")
cmd = ["ffmpeg", "-y"] + inputs + [
    "-filter_complex", filter_str,
    "-map", "[out]",
    "-ac", "1", "-ar", str(args.sr),
    dubbed_audio
]

print("Mixing TTS segments...")
r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
if r.returncode != 0:
    print("ffmpeg mix failed:")
    print(r.stderr.decode("utf-8", errors="ignore"))
    sys.exit(1)
print(f"  Mixed audio: {dubbed_audio} ({get_duration(dubbed_audio):.3f}s)")

# ── Combine with video ──────────────────────────────────────────────────
out_dir = os.path.dirname(args.out)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)

if args.bg_vol > 0:
    # Keep original audio at low volume as background
    print(f"Merging with original audio (bg_vol={args.bg_vol})...")
    cmd = [
        "ffmpeg", "-y",
        "-i", args.clip,         # input 0: original video+audio
        "-i", dubbed_audio,      # input 1: dubbed TTS audio
        "-filter_complex",
        f"[0:a]volume={args.bg_vol},aresample={args.sr}[bg];"
        f"[1:a]aresample={args.sr}[fg];"
        f"[bg][fg]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[aout]",
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-ac", "2", "-ar", "48000",
        "-shortest",
        args.out
    ]
else:
    # Replace audio entirely
    print("Replacing audio track...")
    cmd = [
        "ffmpeg", "-y",
        "-i", args.clip,
        "-i", dubbed_audio,
        "-map", "0:v",
        "-map", "1:a",
        "-c:v", "copy",
        "-ac", "2", "-ar", "48000",
        "-shortest",
        args.out
    ]

r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
if r.returncode != 0:
    print("ffmpeg merge failed:")
    print(r.stderr.decode("utf-8", errors="ignore"))
    sys.exit(1)

final_dur = get_duration(args.out)
print(f"\nDubbed video: {args.out}")
print(f"Duration: {final_dur:.3f}s")
print("Done!")
