import argparse
import yaml

from pipeline.clip_extract import extract_clip
from pipeline.scene_detect import detect_scenes


def load_config(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    extract_clip(
        cfg["paths"]["input_video"],
        cfg["clip"]["start"],
        float(cfg["clip"]["duration"]),
        cfg["paths"]["clip_video"],
        cfg["paths"]["clip_audio"],
        int(cfg["audio"]["sample_rate"]),
        int(cfg["audio"]["channels"]),
    )

    scenes = detect_scenes(
        cfg["paths"]["clip_video"],
        cfg["paths"]["scenes_json"],
        float(cfg["scene"]["threshold"]),
        float(cfg["scene"]["min_scene_len"]),
    )
    print("Wrote:", cfg["paths"]["clip_video"])
    print("Wrote:", cfg["paths"]["clip_audio"])
    print("Wrote:", cfg["paths"]["scenes_json"])
    print("Scene count:", len(scenes))

if __name__ == "__main__":
    main()
