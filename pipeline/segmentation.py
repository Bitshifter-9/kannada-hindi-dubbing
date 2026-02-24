import argparse
import json
import os
import wave
import numpy as np
import torch
from utils.ffmpeg_utils import ensure_parent_dir

def read_wav(path):
    if not os.path.exists(path):
        raise RuntimeError("Missing wav: " + path)
    w = wave.open(path, "rb")
    ch = w.getnchannels()
    sr = w.getframerate()
    sw = w.getsampwidth()
    n = w.getnframes()
    raw = w.readframes(n)
    w.close()
    if sw != 2:
        raise RuntimeError("Expected 16-bit PCM wav, got sampwidth=" + str(sw))
    data = np.frombuffer(raw, dtype=np.int16)
    if ch == 2:
        data = data.reshape(-1, 2).mean(axis=1)
    elif ch != 1:
        raise RuntimeError("Expected 1 or 2 channels, got " + str(ch))
    audio = data.astype(np.float32) / 32768.0
    return audio, sr

def load_json(path):
    if not path or not os.path.exists(path):
        return []
    f = open(path, "r", encoding="utf-8")
    data = json.load(f)
    f.close()
    return data

def load_vad():
    try:
        model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", force_reload=False, trust_repo=True)
    except TypeError:
        model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", force_reload=False)
    model.to("cpu")
    get_ts = utils[0]
    return model, get_ts


def scene_id(t, scenes):
    if not scenes:
        return 0
    i = 0
    while i < len(scenes):
        s = scenes[i]
        if t >= float(s["start"]) and t < float(s["end"]):
            return i
        i += 1
    return len(scenes) - 1


def split_on_scenes(segs, scenes):
    if not scenes:
        out = []
        for seg in segs:
            out.append({"start": seg["start"], "end": seg["end"], "scene": 0})
        return out
    cuts = []
    i = 0
    while i < len(scenes) - 1:
        cuts.append(float(scenes[i]["end"]))
        i += 1
    out = []
    for seg in segs:
        st = seg["start"]
        en = seg["end"]
        pts = []
        for c in cuts:
            if c > st and c < en:
                pts.append(c)
        if not pts:
            mid = (st + en) / 2.0
            out.append({"start": st, "end": en, "scene": scene_id(mid, scenes)})
        else:
            cur = st
            for c in pts:
                mid = (cur + c) / 2.0
                out.append({"start": cur, "end": c, "scene": scene_id(mid, scenes)})
                cur = c
            mid = (cur + en) / 2.0
            out.append({"start": cur, "end": en, "scene": scene_id(mid, scenes)})
    return out


def merge_by_gap(segs, gap, max_len):
    if not segs:
        return []
    segs.sort(key=lambda x: x["start"])
    out = [segs[0]]
    for seg in segs[1:]:
        prev = out[-1]
        g = seg["start"] - prev["end"]
        if seg["scene"] == prev["scene"] and g <= gap and (seg["end"] - prev["start"]) <= max_len:
            prev["end"] = seg["end"]
        else:
            out.append(seg)
    return out


def force_split_long(segs, max_len, wav, sr):
    """Split any segment longer than max_len at the lowest-energy point."""
    out = []
    for seg in segs:
        dur = seg["end"] - seg["start"]
        if dur <= max_len:
            out.append(seg)
            continue
        # recursively split at the quietest point
        stack = [seg]
        while stack:
            s = stack.pop(0)
            d = s["end"] - s["start"]
            if d <= max_len:
                out.append(s)
                continue
            # find lowest-energy 50ms frame in the middle 80% of the segment
            s0 = int(s["start"] * sr)
            s1 = int(s["end"] * sr)
            margin = int((s1 - s0) * 0.10)
            search_start = s0 + margin
            search_end = s1 - margin
            frame = int(0.05 * sr)  # 50ms
            if search_end - search_start < frame:
                out.append(s)
                continue
            best_e = float("inf")
            best_i = (search_start + search_end) // 2
            i = search_start
            while i + frame <= search_end:
                chunk = wav[i:i + frame]
                e = float(np.sum(chunk * chunk))
                if e < best_e:
                    best_e = e
                    best_i = i
                i += frame // 2  # 25ms hop
            split_t = round(best_i / sr, 3)
            left = {"start": s["start"], "end": split_t, "scene": s["scene"]}
            right = {"start": split_t, "end": s["end"], "scene": s["scene"]}
            stack.insert(0, left)
            stack.insert(1, right)
    out.sort(key=lambda x: x["start"])
    return out


def make_segments(wav_path, scenes_path, out_json, th, min_sp, min_sil, pad, min_len, max_len, gap):
    wav, sr = read_wav(wav_path)
    if sr != 16000:
        raise RuntimeError("Expected 16000 Hz wav, got " + str(sr) + ". Re-extract clip.wav at 16k.")
    scenes = load_json(scenes_path)
    model, get_ts = load_vad()
    x = torch.from_numpy(wav)
    ts = get_ts(x, model, sampling_rate=sr, threshold=th, min_speech_duration_ms=min_sp, min_silence_duration_ms=min_sil, speech_pad_ms=pad)
    raw = []
    for t in ts:
        st = float(t["start"]) / float(sr)
        en = float(t["end"]) / float(sr)
        if en > st:
            raw.append({"start": round(st, 3), "end": round(en, 3)})
    segs = split_on_scenes(raw, scenes)
    segs = merge_by_gap(segs, gap, max_len)
    segs = force_split_long(segs, max_len, wav, sr)
    final = []
    i = 0
    for seg in segs:
        st = round(float(seg["start"]), 3)
        en = round(float(seg["end"]), 3)
        dur = round(en - st, 3)
        if dur <= 0:
            continue
        if dur < min_len and i > 0:
            prev = final[-1]
            if prev["scene"] == seg["scene"] and (en - prev["start"]) <= max_len and (st - prev["end"]) <= gap:
                prev["end"] = en
                prev["dur"] = round(prev["end"] - prev["start"], 3)
                continue
        final.append({"id": i, "scene": int(seg["scene"]), "start": st, "end": en, "dur": dur})
        i += 1
    ensure_parent_dir(out_json)
    f = open(out_json, "w", encoding="utf-8")
    json.dump(final, f, indent=2)
    f.close()
    return final


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wav", required=True)
    p.add_argument("--scenes", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--th", type=float, default=0.5)
    p.add_argument("--min_sp", type=int, default=250)
    p.add_argument("--min_sil", type=int, default=200)
    p.add_argument("--pad", type=int, default=50)
    p.add_argument("--min_len", type=float, default=1.0)
    p.add_argument("--max_len", type=float, default=4.0)
    p.add_argument("--gap", type=float, default=0.35)
    a = p.parse_args()
    segs = make_segments(a.wav, a.scenes, a.out, a.th, a.min_sp, a.min_sil, a.pad, a.min_len, a.max_len, a.gap)
    print("Segments:", len(segs))
    print("Output:", a.out)


if __name__ == "__main__":
    main()
