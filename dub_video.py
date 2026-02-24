import argparse
import yaml

from pipeline.clip_extract import extract_clip
from pipeline.scene_detect import detect_scenes
from pipeline.segmentation import make_segments


def load_config(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    extract_clip(cfg["paths"]["input_video"], cfg["clip"]["start"], float(cfg["clip"]["duration"]), cfg["paths"]["clip_video"], cfg["paths"]["clip_audio"], int(cfg["audio"]["sample_rate"]), int(cfg["audio"]["channels"]))
    scenes = detect_scenes(cfg["paths"]["clip_video"], cfg["paths"]["scenes_json"], float(cfg["scene"]["threshold"]), float(cfg["scene"]["min_scene_len"]))
    segs = make_segments(cfg["paths"]["clip_audio"], cfg["paths"]["scenes_json"], cfg["paths"]["segments_json"], float(cfg["vad"]["threshold"]), int(cfg["vad"]["min_speech_ms"]), int(cfg["vad"]["min_silence_ms"]), int(cfg["vad"]["pad_ms"]), float(cfg["seg"]["min_len"]), float(cfg["seg"]["max_len"]), float(cfg["seg"]["gap"]))

    print("Wrote:", cfg["paths"]["clip_video"])
    print("Wrote:", cfg["paths"]["clip_audio"])
    print("Wrote:", cfg["paths"]["scenes_json"], "scenes:", len(scenes))
    print("Wrote:", cfg["paths"]["segments_json"], "segments:", len(segs))


if __name__ == "__main__":
    main()
