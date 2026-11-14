#!/usr/bin/env python3
"""
Cloud Video Studio Pipeline (Runs on GitHub Actions 4-vCPU / 16GB RAM Runner)
Features:
1. Download input video (from YouTube/Direct URL or local repo path)
2. Trim start/end timestamps losslessly or with re-encode
3. Auto-remove silence using auto-editor
4. Convert aspect ratio (e.g. 9:16 vertical for Shorts/TikTok/Reels with blurred background or center crop)
5. Normalize audio loudness (-14 LUFS for YouTube/TikTok standard)
6. Extract SEO metadata & generate meta.json for YouTube upload
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str]):
    print(f"\n[EXEC] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def download_source(source: str, output_dir: Path) -> Path:
    if os.path.exists(source):
        return Path(source)
    out_template = str(output_dir / "source_video.%(ext)s")
    run_cmd([
        sys.executable, "-m", "yt_dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", out_template,
        source
    ])
    matches = list(output_dir.glob("source_video.*"))
    if not matches:
        raise FileNotFoundError("Failed to download source video.")
    return matches[0]


def process_video(
    input_path: Path,
    output_path: Path,
    trim_start: str = "",
    trim_end: str = "",
    remove_silence: bool = True,
    aspect_mode: str = "original",
    normalize_audio: bool = True,
    speed: float = 1.0,
):
    current_file = input_path
    work_dir = output_path.parent

    # Step 1: Trim if requested
    if trim_start or trim_end:
        trimmed = work_dir / "step1_trimmed.mp4"
        cmd = ["ffmpeg", "-y"]
        if trim_start:
            cmd += ["-ss", trim_start]
        if trim_end:
            cmd += ["-to", trim_end]
        cmd += ["-i", str(current_file), "-c", "copy", str(trimmed)]
        run_cmd(cmd)
        current_file = trimmed

    # Step 2: Auto-remove silence via auto-editor
    if remove_silence:
        unsilenced = work_dir / "step2_unsilenced.mp4"
        run_cmd([
            "auto-editor", str(current_file),
            "--margin", "0.2sec",
            "--output-file", str(unsilenced),
            "--no-open"
        ])
        current_file = unsilenced

    # Step 3: FFmpeg Filters (Aspect Ratio 9:16, Speed, Loudness Normalization)
    vf_filters = []
    af_filters = []

    if aspect_mode == "9:16-crop":
        vf_filters.append("crop=ih*9/16:ih,scale=1080:1920")
    elif aspect_mode == "9:16-blur":
        # Create blurred 1080x1920 background with centered foreground
        complex_filter = (
            "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920,boxblur=25:5[bg];"
            "[0:v]scale=1080:1920:force_original_aspect_ratio=decrease[fg];"
            "[bg][fg]overlay=(W-w)/2:(H-h)/2"
        )
        if speed != 1.0 and speed > 0:
            complex_filter += f",setpts={1.0/speed}*PTS"
        if normalize_audio:
            af_filters.append("loudnorm=I=-14:TP=-1.5:LRA=11")
        if speed != 1.0 and speed > 0:
            af_filters.append(f"atempo={speed}")

        cmd = ["ffmpeg", "-y", "-i", str(current_file), "-filter_complex", complex_filter]
        if af_filters:
            cmd += ["-af", ",".join(af_filters)]
        cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "20", "-c:a", "aac", "-b:a", "192k", str(output_path)]
        run_cmd(cmd)
        return

    if speed != 1.0 and speed > 0:
        vf_filters.append(f"setpts={1.0/speed}*PTS")
        af_filters.append(f"atempo={speed}")

    if normalize_audio:
        af_filters.append("loudnorm=I=-14:TP=-1.5:LRA=11")

    if vf_filters or af_filters:
        cmd = ["ffmpeg", "-y", "-i", str(current_file)]
        if vf_filters:
            cmd += ["-vf", ",".join(vf_filters)]
        if af_filters:
            cmd += ["-af", ",".join(af_filters)]
        cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "20", "-c:a", "aac", "-b:a", "192k", str(output_path)]
        run_cmd(cmd)
    else:
        if current_file != output_path:
            os.replace(current_file, output_path)


def generate_seo_meta(keyword: str, title: str, output_dir: Path) -> Path:
    meta_file = output_dir / "meta.json"
    tags = [keyword] if keyword else []
    if keyword:
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "yt_dlp", f"ytsearch3:{keyword}", "--dump-json", "--skip-download", "--quiet"],
                capture_output=True, text=True, encoding="utf-8", errors="replace"
            )
            for line in proc.stdout.splitlines():
                if line.strip():
                    d = json.loads(line)
                    for t in (d.get("tags") or [])[:8]:
                        if t.lower() not in [x.lower() for x in tags]:
                            tags.append(t)
        except Exception as e:
            print(f"[WARN] SEO tag extraction warning: {e}")

    final_title = title if title else (f"{keyword.title()} - Hướng Dẫn Chi Tiết" if keyword else "Processed Video")
    meta = {
        "title": final_title[:95],
        "description": f"{final_title}\n\n📌 Nội dung chính:\n00:00 Mở đầu\n\n#{keyword.replace(' ', '') if keyword else 'video'} #shorts",
        "tags": tags[:25],
        "privacyStatus": "private",
        "embeddable": True,
        "categoryId": "22"
    }
    meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta_file


def main():
    parser = argparse.ArgumentParser(description="Cloud Video Editor & YouTube SEO Pipeline")
    parser.add_argument("--source", required=True, help="Video URL (YouTube/Direct MP4/Google Drive) or local path")
    parser.add_argument("--trim-start", default="", help="Start time (e.g. 00:00:10)")
    parser.add_argument("--trim-end", default="", help="End time (e.g. 00:02:30)")
    parser.add_argument("--remove-silence", default="true", help="true/false to auto-cut silence")
    parser.add_argument("--aspect", default="original", choices=["original", "9:16-crop", "9:16-blur"], help="Aspect ratio mode")
    parser.add_argument("--normalize", default="true", help="true/false to normalize audio to -14 LUFS")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (e.g. 1.05)")
    parser.add_argument("--seo-keyword", default="", help="Target keyword for YouTube SEO metadata")
    parser.add_argument("--title", default="", help="Custom video title")
    parser.add_argument("--output-dir", default="output", help="Output folder")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path("work_tmp")
    work_dir.mkdir(parents=True, exist_ok=True)

    src_file = download_source(args.source, work_dir)
    out_video = out_dir / "final_video.mp4"

    process_video(
        input_path=src_file,
        output_path=out_video,
        trim_start=args.trim_start.strip(),
        trim_end=args.trim_end.strip(),
        remove_silence=(args.remove_silence.lower() == "true"),
        aspect_mode=args.aspect,
        normalize_audio=(args.normalize.lower() == "true"),
        speed=args.speed,
    )

    meta_path = generate_seo_meta(args.seo_keyword.strip(), args.title.strip(), out_dir)
    print(f"\n✅ Done! Output video: {out_video} | SEO Metadata: {meta_path}")


if __name__ == "__main__":
    main()
