from utils.ffmpeg_utils import ensure_parent_dir, require_cmd, run
def extract_clip(input_video, start, duration, out_video, out_wav, sample_rate=16000, channels=1):
    require_cmd("ffmpeg")
    ensure_parent_dir(out_video)
    ensure_parent_dir(out_wav)
    video_cmd = [
        "ffmpeg",
        "-y",
        "-ss", start,
        "-i", input_video,
        "-t", str(duration),
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "medium",
        "-c:a", "aac",
        "-movflags", "+faststart",
        out_video
    ]

    run(video_cmd)
    audio_cmd = [
        "ffmpeg",
        "-y",
        "-ss", start,
        "-i", input_video,
        "-t", str(duration),
        "-vn",
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        out_wav
    ]

    run(audio_cmd)