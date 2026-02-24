import argparse
import yaml

from pipeline.clip_extract import extract_clip
from pipeline.scene_detect import detect_scenes
from pipeline.segmentation import make_segments
from pipeline.alignment import transcribe_segments


def load_config(p):
    f = open(p, "r")
    d = yaml.safe_load(f)
    f.close()
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    extract_clip(cfg["paths"]["input_video"], cfg["clip"]["start"], float(cfg["clip"]["duration"]), cfg["paths"]["clip_video"], cfg["paths"]["clip_audio"], int(cfg["audio"]["sample_rate"]), int(cfg["audio"]["channels"]))
    scenes = detect_scenes(cfg["paths"]["clip_video"], cfg["paths"]["scenes_json"], float(cfg["scene"]["threshold"]), float(cfg["scene"]["min_scene_len"]))
    segs = make_segments(cfg["paths"]["clip_audio"], cfg["paths"]["scenes_json"], cfg["paths"]["segments_json"], float(cfg["vad"]["threshold"]), int(cfg["vad"]["min_speech_ms"]), int(cfg["vad"]["min_silence_ms"]), int(cfg["vad"]["pad_ms"]), float(cfg["seg"]["min_len"]), float(cfg["seg"]["max_len"]), float(cfg["seg"]["gap"]))
    task = cfg.get("asr", {}).get("task", "transcribe")
    no_gpu = bool(cfg.get("asr", {}).get("no_gpu", False))
    prompt = (cfg.get("asr", {}).get("prompt", "") or "").strip()
    redo = bool(cfg.get("asr", {}).get("redo", False))
    ar = transcribe_segments(cfg["paths"]["clip_audio"], cfg["paths"]["segments_json"], cfg["paths"]["asr_json"], cfg["asr"]["bin"], cfg["asr"]["model"], cfg["asr"]["lang"], task, no_gpu, prompt, redo)
    print("Scenes:", len(scenes))
    print("Segments:", len(segs))
    print("ASR:", len(ar))
    print("Wrote:", cfg["paths"]["asr_json"])

if __name__ == "__main__":
    main()
