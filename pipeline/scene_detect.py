import argparse
import json
import re
from utils.ffmpeg_utils import ensure_parent_dir, require_cmd, run_capture
def get_duration_seconds(video_path):
    require_cmd("ffprobe")
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    out, err = run_capture(cmd)

    text = out.strip()
    if not text:
        text = err.strip()

    if not text:
        raise RuntimeError("ffprobe did not return duration")

    return float(text)


def find_raw_cut_times(video_path, threshold):
    require_cmd("ffmpeg")

    filter_str = "select='gt(scene,{0})',showinfo".format(threshold)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        video_path,
        "-filter:v",
        filter_str,
        "-an",
        "-f",
        "null",
        "-",
    ]

    out, err = run_capture(cmd)
    logs = err

    times = []
    for s in re.findall(r"pts_time:([0-9]+(?:\.[0-9]+)?)", logs):
        t = float(s)
        if t > 0.01:
            times.append(t)

    times.sort()
    return times


def merge_close_times(times, window_seconds):
    merged = []
    for t in times:
        if not merged:
            merged.append(t)
            continue

        if t - merged[-1] >= window_seconds:
            merged.append(t)

    return merged


def build_scenes(duration, cut_times, min_scene_len):
    boundaries = [0.0]

    for t in cut_times:
        if t > 0.0 and t < duration:
            boundaries.append(t)

    boundaries.append(duration)
    boundaries = sorted(set([round(x, 3) for x in boundaries]))

    scenes = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        seg_len = end - start

        if seg_len <= 0:
            continue

        if scenes and seg_len < min_scene_len:
            scenes[-1]["end"] = round(end, 3)
            scenes[-1]["duration"] = round(scenes[-1]["end"] - scenes[-1]["start"], 3)
        else:
            scenes.append(
                {
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "duration": round(seg_len, 3),
                }
            )

    if not scenes:
        scenes = [{"start": 0.0, "end": round(duration, 3), "duration": round(duration, 3)}]

    return scenes

def detect_scenes(input_video, output_json, threshold, min_scene_len):
    duration = get_duration_seconds(input_video)
    raw_times = find_raw_cut_times(input_video, threshold)
    cut_times = merge_close_times(raw_times, window_seconds=0.30)
    scenes = build_scenes(duration, cut_times, min_scene_len)
    ensure_parent_dir(output_json)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(scenes, f, indent=2)
    return scenes
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--min_scene_len", type=float, default=0.80)
    args = parser.parse_args()
    scenes = detect_scenes(args.input, args.output, args.threshold, args.min_scene_len)
    print("Saved scenes:", len(scenes))
    print("Output:", args.output)


if __name__ == "__main__":
    main()
