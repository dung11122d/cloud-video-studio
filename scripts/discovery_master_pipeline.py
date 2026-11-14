#!/usr/bin/env python3
"""
Discovery Documentary 4K 60fps Master Pipeline (Single-Video QA First)
Fixes all previous issues:
1. Video Quality: True 4K UHD (3840x2160) @ 60fps with Lanczos + CAS (Contrast Adaptive Sharpening)
   + Natural Cinema Color Grading (NO artificial noise/grain filter that degrades sharpness).
2. Original Audio Preserved 100%: Keeps the original narration & nature audio from each scene
   (resampled cleanly to 48kHz Stereo AAC 320kbps), with smooth audio crossfades (`acrossfade=0.5s`)
   and 0.5s visual `xfade` transitions between scenes.
3. Clean Warm Ambient Music Bed (Optional/Subtle): Pure warm harmonic sine pads (174Hz, 261Hz, 329Hz, 392Hz)
   with gentle echo — NO pink/white noise hiss ("rồ rồ") — mixed softly at 15% volume under the original voice/audio.
4. Deep AI Artifact Trimming: OpenCV checks tail frames for sudden blur/morphing and trims corrupted tail frames
   while preserving speech continuity.
"""
import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ALL_XFADE_TRANSITIONS = [
    "fade", "dissolve", "smoothleft", "smoothright", "smoothup", "smoothdown",
    "circleopen", "circleclose", "wipeleft", "wiperight", "wipeup", "wipedown",
    "slideleft", "slideright", "slideup", "slidedown", "radial", "rectcrop",
    "circlecrop", "distance", "fadeblack", "vertopen", "vertclose",
    "horzopen", "horzclose", "diagtl", "diagtr", "diagbl", "diagbr",
    "hlslice", "hrslice", "vuslice", "vdslice", "hblur", "fadegrays",
    "wipetl", "wipetr", "wipebl", "wipebr", "squeezeh", "squeezev",
    "zoomin", "coverleft", "coverright", "coverup", "coverdown",
    "revealleft", "revealright", "revealup", "revealdown"
]


def run(cmd: list[str], check: bool = True):
    print(f"[EXEC] {' '.join(cmd[:8])} ...", flush=True)
    return subprocess.run(cmd, check=check)


def natural_scene_num(name: str) -> int:
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else 9999


def download_single_drive_file(args_tuple) -> Path | None:
    scene_name, file_id, dest_dir = args_tuple
    out_path = dest_dir / scene_name
    if out_path.exists() and out_path.stat().st_size > 50_000:
        return out_path

    url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = resp.read()
            if len(data) > 50_000 and not data[:15].startswith(b"<!DOCTYPE html"):
                out_path.write_bytes(data)
                return out_path
    except Exception:
        pass

    try:
        subprocess.run(
            [sys.executable, "-m", "gdown", "--no-check-certificate", "-q", file_id, "-O", str(out_path)],
            check=False, timeout=90
        )
        if out_path.exists() and out_path.stat().st_size > 50_000:
            return out_path
    except Exception as e:
        print(f"[WARN] Failed downloading {scene_name} ({file_id}): {e}", flush=True)
    return None


def probe_clip_duration_and_audio(clip_path: Path) -> tuple[float, bool]:
    """Uses ffmpeg -i stderr to get duration and check if audio stream exists (no ffprobe dependency needed)."""
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(clip_path)], capture_output=True, text=True)
    stderr = proc.stderr
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", stderr)
    dur = 10.0
    if m:
        dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    has_audio = "Audio:" in stderr
    return dur, has_audio


def detect_ai_tail_glitch(clip_path: Path, raw_dur: float) -> tuple[float, float, float]:
    """
    Checks if the very end of the AI clip (last 0.4s) has severe frame morphing/blur breakdown.
    Only trims if an actual anomaly is detected, keeping 96-100% of the clip intact so audio/narration is never cut mid-word.
    """
    try:
        import cv2
        import numpy as np
        cap = cv2.VideoCapture(str(clip_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if fps > 0 else raw_dur

        sharpness = []
        step = max(int(fps // 4), 1)
        for idx in range(0, frame_count, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            small = cv2.resize(frame, (320, 180))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            sharpness.append((idx / fps, float(cv2.Laplacian(gray, cv2.CV_64F).var())))
        cap.release()

        avg_sharp = sum(s for _, s in sharpness) / len(sharpness) if sharpness else 100.0
        end_t = duration
        # Only trim tail 0.3s if the last frames suffer catastrophic AI blur collapse (<40% of clip sharpness)
        if sharpness and sharpness[-1][1] < avg_sharp * 0.40 and duration > 4.0:
            end_t = round(duration - 0.30, 2)
        return 0.0, round(end_t, 2), round(avg_sharp, 1)
    except Exception:
        return 0.0, round(raw_dur, 2), 100.0


def normalize_scene_worker(args_tuple) -> dict | None:
    idx, clip_path, graded_dir, width, height, fps = args_tuple
    raw_dur, has_audio = probe_clip_duration_and_audio(clip_path)
    st, et, score = detect_ai_tail_glitch(clip_path, raw_dur)
    dur = max(et - st, 2.0)

    dst = graded_dir / f"scene_{idx:03d}.mp4"

    # Crystal-Clear 4K / High-Bitrate 60fps Upscaling & Cinema Color Grade (NO artificial noise!)
    # - scale with lanczos + accurate_rnd for razor-sharp upscaling from 720p to 4K/2K
    # - cas=0.45: Contrast Adaptive Sharpening (makes feathers, eyes, and foliage crisp)
    # - eq & colorbalance: Natural BBC Earth / Discovery contrast and subtle golden-hour warmth
    vf_chain = (
        f"scale={width}:{height}:flags=lanczos+accurate_rnd+full_chroma_int,"
        f"fps={fps},"
        f"cas=0.45,"
        f"eq=contrast=1.05:brightness=0.005:saturation=1.10:gamma=0.98,"
        f"colorbalance=rs=-0.015:gs=0.0:bs=0.02:rh=0.02:gh=0.01:bh=-0.015,"
        f"format=yuv420p"
    )

    # First scene linear blur-in (0.0s -> 0.5s: 0% sharp to 100% sharp) as in user's original Colab design
    if idx == 1:
        vf_chain += (
            ",split=2[sharp][blur_in];"
            "[blur_in]gblur=sigma=20[blurred];"
            "[blurred][sharp]blend=all_expr='if(lte(T,0.5), A*(1-T/0.5)+B*(T/0.5), B)'"
        )

    # Audio chain: Keep 100% original narration/wildlife sound, clean sub-60Hz rumble, crystal clear 48kHz stereo
    if has_audio:
        af_chain = "highpass=f=60,aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo"
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{st:.2f}", "-to", f"{et:.2f}",
            "-i", str(clip_path),
            "-vf", vf_chain,
            "-af", af_chain,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
            "-c:a", "aac", "-b:a", "320k", "-ar", "48000", "-ac", "2",
            str(dst)
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{st:.2f}", "-to", f"{et:.2f}",
            "-i", str(clip_path),
            "-f", "lavfi", "-t", f"{dur:.2f}", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-vf", vf_chain,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
            "-c:a", "aac", "-b:a", "320k", "-ar", "48000", "-ac", "2",
            "-shortest",
            str(dst)
        ]

    subprocess.run(cmd, check=True)
    actual_dur, _ = probe_clip_duration_and_audio(dst)
    return {
        "idx": idx,
        "scene": clip_path.name,
        "path": dst,
        "duration": actual_dur,
        "trim": f"{st}s -> {et}s",
        "sharpness_score": score
    }


def join_batch_with_xfade(batch_items: list[dict], out_path: Path, overlap: float = 0.5, start_t_idx: int = 0) -> float:
    if len(batch_items) == 1:
        shutil.copy2(batch_items[0]["path"], out_path)
        return batch_items[0]["duration"]

    inputs = []
    for item in batch_items:
        inputs.extend(["-i", str(item["path"])])

    n = len(batch_items)
    v_chains = []
    a_chains = []
    accumulated_offset = batch_items[0]["duration"]

    for i in range(1, n):
        offset = max(0.1, accumulated_offset - overlap)
        trans_name = ALL_XFADE_TRANSITIONS[(start_t_idx + i - 1) % len(ALL_XFADE_TRANSITIONS)]

        vin_a = "[0:v]" if i == 1 else f"[vx{i-1}]"
        vin_b = f"[{i}:v]"
        vout = "[vout]" if i == n - 1 else f"[vx{i}]"
        v_chains.append(f"{vin_a}{vin_b}xfade=transition={trans_name}:duration={overlap:.3f}:offset={offset:.3f}{vout}")

        ain_a = "[0:a]" if i == 1 else f"[ax{i-1}]"
        ain_b = f"[{i}:a]"
        aout = "[aout]" if i == n - 1 else f"[ax{i}]"
        a_chains.append(f"{ain_a}{ain_b}acrossfade=d={overlap:.3f}:c1=tri:c2=tri{aout}")

        accumulated_offset = offset + batch_items[i]["duration"]

    filter_complex = ";".join(v_chains + a_chains)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "320k", "-ar", "48000",
        str(out_path)
    ]
    subprocess.run(cmd, check=True)
    actual_dur, _ = probe_clip_duration_and_audio(out_path)
    return actual_dur


def generate_pure_ambient_bgm(out_m4a: Path):
    """
    Pure warm harmonic sine pad (NO pink noise / NO static hiss 'rồ rồ'!).
    Soft C-major / F-maj7 warm chords (174.61Hz, 261.63Hz, 329.63Hz, 392.00Hz) with gentle reverb.
    """
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "sine=frequency=174.61:duration=180",
        "-f", "lavfi", "-i", "sine=frequency=261.63:duration=180",
        "-f", "lavfi", "-i", "sine=frequency=329.63:duration=180",
        "-f", "lavfi", "-i", "sine=frequency=392.00:duration=180",
        "-filter_complex",
        "[0:a]volume=0.08,tremolo=f=0.12:d=0.35[a0];"
        "[1:a]volume=0.07,tremolo=f=0.10:d=0.40[a1];"
        "[2:a]volume=0.05,tremolo=f=0.08:d=0.40[a2];"
        "[3:a]volume=0.04,tremolo=f=0.06:d=0.45[a3];"
        "[a0][a1][a2][a3]amix=inputs=4:duration=longest,lowpass=f=1200,aecho=0.8:0.88:60:0.35,"
        "afade=t=in:st=0:d=3,afade=t=out:st=177:d=3[outa]",
        "-map", "[outa]", "-c:a", "aac", "-b:a", "256k", "-ar", "48000",
        str(out_m4a)
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Discovery 4K 60fps Master Pipeline (Original Audio + Clean BGM + Xfade)")
    parser.add_argument("--project-key", required=True)
    parser.add_argument("--manifest", default="scripts/drive_manifest.json")
    parser.add_argument("--resolution", default="3840x2160", help="3840x2160 (4K) or 1920x1080")
    parser.add_argument("--fps", type=int, default=60, help="Target FPS (60)")
    parser.add_argument("--bgm-volume", type=float, default=0.14, help="Soft ambient pad volume under original audio")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    w_str, h_str = args.resolution.lower().split("x")
    width, height = int(w_str), int(h_str)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8-sig"))
    project_info = manifest[args.project_key]
    sorted_items = sorted(project_info["scenes"].items(), key=lambda kv: natural_scene_num(kv[0]))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path("work_raw")
    graded_dir = Path("work_graded")
    batch_dir = Path("work_batches")
    for d in (raw_dir, graded_dir, batch_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"📥 [1/4] Downloading {len(sorted_items)} scenes for '{args.project_key}'...", flush=True)
    dl_tasks = [(name, fid, raw_dir) for name, fid in sorted_items]
    downloaded = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        for res in pool.map(download_single_drive_file, dl_tasks):
            if res is not None:
                downloaded.append(res)
    downloaded.sort(key=lambda p: natural_scene_num(p.name))

    print(f"🎬 [2/4] Upscaling & Color Grading {len(downloaded)} scenes to {width}x{height} @ {args.fps}fps (Preserving Original Audio)...", flush=True)
    norm_tasks = [(idx, p, graded_dir, width, height, args.fps) for idx, p in enumerate(downloaded, 1)]
    norm_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for res in pool.map(normalize_scene_worker, norm_tasks):
            if res is not None:
                norm_results.append(res)
    norm_results.sort(key=lambda r: r["idx"])

    print(f"🎞️ [3/4] Applying 0.5s Xfade Video & Acrossfade Audio Transitions across all {len(norm_results)} scenes...", flush=True)
    BATCH_SIZE = 6
    batch_items = []
    for b_idx, start_i in enumerate(range(0, len(norm_results), BATCH_SIZE)):
        chunk = norm_results[start_i:start_i + BATCH_SIZE]
        b_out = batch_dir / f"batch_{b_idx+1:02d}.mp4"
        b_dur = join_batch_with_xfade(chunk, b_out, overlap=0.5, start_t_idx=start_i)
        batch_items.append({"path": b_out, "duration": b_dur})
        print(f"   ✔️ Batch {b_idx+1}/{(len(norm_results)+BATCH_SIZE-1)//BATCH_SIZE} complete ({b_dur:.2f}s)", flush=True)

    merged_scenes = out_dir / "merged_no_bgm.mp4"
    total_dur = join_batch_with_xfade(batch_items, merged_scenes, overlap=0.5, start_t_idx=len(norm_results))

    print(f"🎵 [4/4] Mixing Clean Original Audio (100%) + Pure Warm Harmonic Pad ({int(args.bgm_volume*100)}%, NO noise hiss)...", flush=True)
    bgm_file = out_dir / "pure_bgm.m4a"
    generate_pure_ambient_bgm(bgm_file)

    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", args.project_key).strip("_")
    final_output = out_dir / f"{safe_name}_4K60FPS_Master.mp4"
    fade_out_st = max(0.0, total_dur - 3.5)

    bgm_filter = (
        f"[0:a]volume=1.0[orig_a];"
        f"[1:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
        f"volume={args.bgm_volume:.2f},afade=t=in:st=0:d=2.5,afade=t=out:st={fade_out_st:.2f}:d=3.5[bgm_a];"
        f"[orig_a][bgm_a]amix=inputs=2:duration=first:dropout_transition=2,alimiter=limit=0.95[final_a]"
    )
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(merged_scenes),
        "-stream_loop", "-1", "-i", str(bgm_file),
        "-filter_complex", bgm_filter,
        "-map", "0:v:0", "-map", "[final_a]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "320k", "-ar", "48000",
        "-movflags", "+faststart",
        "-shortest",
        str(final_output)
    ], check=True)

    shutil.rmtree(raw_dir, ignore_errors=True)
    shutil.rmtree(graded_dir, ignore_errors=True)
    shutil.rmtree(batch_dir, ignore_errors=True)
    merged_scenes.unlink(missing_ok=True)
    bgm_file.unlink(missing_ok=True)

    size_mb = final_output.stat().st_size / (1024 * 1024)
    print(f"🎉 MASTER 4K60FPS COMPLETE: {final_output} ({size_mb:.1f} MB | {total_dur/60:.2f} mins)", flush=True)


if __name__ == "__main__":
    main()
