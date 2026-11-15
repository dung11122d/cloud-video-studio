import os
import sys
import json
import time
import shutil
import subprocess
import urllib.request
import http.cookiejar
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np


def download_gdrive_file(file_id: str, dest_path: Path, max_retries: int = 5) -> bool:
    if dest_path.exists() and dest_path.stat().st_size > 100_000:
        return True
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")

    for attempt in range(1, max_retries + 1):
        try:
            cj = http.cookiejar.CookieJar()
            opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
            opener.addheaders = [
                ("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
            ]
            url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
            with opener.open(url, timeout=90) as resp:
                content_type = resp.headers.get("Content-Type", "")
                if "text/html" not in content_type.lower():
                    with open(temp_path, "wb") as f:
                        while True:
                            chunk = resp.read(1024 * 512)
                            if not chunk:
                                break
                            f.write(chunk)
                    if temp_path.exists() and temp_path.stat().st_size > 100_000:
                        temp_path.replace(dest_path)
                        return True
        except Exception as e:
            print(f"  [Direct attempt {attempt}] {file_id}: {e}")

        # Fallback 2: yt-dlp (uses Google Drive video stream CDN)
        try:
            cmd = [
                "yt-dlp",
                "--no-warnings",
                "--quiet",
                "-f", "best",
                "-o", str(temp_path),
                f"https://drive.google.com/file/d/{file_id}/view"
            ]
            subprocess.run(cmd, check=False, timeout=180)
            if temp_path.exists() and temp_path.stat().st_size > 100_000:
                temp_path.replace(dest_path)
                return True
        except Exception as e:
            print(f"  [yt-dlp attempt {attempt}] {file_id}: {e}")

        # Fallback 3: gdown
        try:
            cmd = ["gdown", "--id", file_id, "-O", str(temp_path), "--quiet"]
            subprocess.run(cmd, check=False, timeout=180)
            if temp_path.exists() and temp_path.stat().st_size > 100_000:
                temp_path.replace(dest_path)
                return True
        except Exception as e:
            print(f"  [gdown attempt {attempt}] {file_id}: {e}")

        time.sleep(attempt * 3)

    return dest_path.exists() and dest_path.stat().st_size > 100_000


def inspect_and_calibrate_clip(clip_path: Path) -> dict:
    cap = cv2.VideoCapture(str(clip_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    duration = frame_count / fps if fps > 0 else 10.0

    sample_indices = [
        int(frame_count * 0.15),
        int(frame_count * 0.50),
        int(frame_count * 0.85)
    ]
    lumas = []
    saturations = []
    sharpness_vals = []

    for idx in sample_indices:
        if 0 <= idx < frame_count:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lumas.append(float(np.mean(hsv[:, :, 2])))
                saturations.append(float(np.mean(hsv[:, :, 1])))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sharpness_vals.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

    # Check tail frames for blur drop
    tail_sharpness = []
    for idx in range(max(0, frame_count - 8), frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tail_sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    cap.release()

    mean_luma = float(np.mean(lumas)) if lumas else 128.0
    mean_sat = float(np.mean(saturations)) if saturations else 120.0
    mean_sharp = float(np.mean(sharpness_vals)) if sharpness_vals else 150.0

    # Auto-calibrate brightness/gamma/saturation for parade vibrancy
    brightness = 0.0
    gamma = 1.0
    if mean_luma < 105:
        brightness = 0.03
        gamma = 1.08
    elif mean_luma > 165:
        brightness = -0.02
        gamma = 0.95

    saturation = 1.12
    if mean_sat < 95:
        saturation = 1.18
    elif mean_sat > 155:
        saturation = 1.04

    trim_end = max(4.0, duration)
    if tail_sharpness and mean_sharp > 0 and min(tail_sharpness) < mean_sharp * 0.45:
        trim_end = max(4.0, duration - 0.25)

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "duration": round(duration, 3),
        "trim_end": round(trim_end, 3),
        "mean_luma": round(mean_luma, 2),
        "mean_sat": round(mean_sat, 2),
        "mean_sharpness": round(mean_sharp, 2),
        "brightness": brightness,
        "gamma": gamma,
        "saturation": saturation
    }


def has_audio_stream(clip_path: Path) -> bool:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a",
        "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(clip_path)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return "audio" in res.stdout.lower()


def normalize_parade_scene(idx: int, raw_path: Path, out_path: Path, qa_info: dict) -> bool:
    trim_dur = qa_info["trim_end"]
    b = qa_info["brightness"]
    g = qa_info["gamma"]
    s = qa_info["saturation"]

    # Stage 1: 1280x720 @ 60fps + 3D temporal denoise + color grading + corner cleanup
    # Scene 0 gets smooth 0.5s Gaussian blur fade-in
    base_vf = (
        f"trim=0:{trim_dur},setpts=PTS-STARTPTS,"
        f"fps=60,scale=1280:720:flags=lanczos,"
        f"hqdn3d=1.2:0.9:2.2:1.6,"
        f"eq=contrast=1.06:brightness={b}:saturation={s}:gamma={g}"
    )
    if idx == 0:
        vf = (
            f"{base_vf},split=2[main][blur];"
            f"[blur]gblur=sigma=18[blurred];"
            f"[blurred][main]blend=all_expr='A*(1-clip(T/0.5,0,1))+B*clip(T/0.5,0,1)',format=yuv420p"
        )
    else:
        vf = f"{base_vf},format=yuv420p"

    # Viral Parade Audio Chain:
    # 1. Remove sub-bass rumble (<65Hz) & 24kHz AI aliasing (>11.8kHz) + FFT denoise
    # 2. Warm marching drum/brass EQ (+1.8dB @ 110Hz, -1.5dB harshness @ 3.2kHz, +1.5dB air @ 6.5kHz)
    # 3. Convert 24kHz Mono -> Wide 48kHz Stereo (Haas micro-echo + stereo widener)
    # 4. Compress & dynamically level (-12dB..-19dB raw swings -> consistent broadcast level)
    af_chain = (
        f"atrim=0:{trim_dur},asetpts=PTS-STARTPTS,"
        f"highpass=f=65,lowpass=f=11800,afftdn=nf=-22,"
        f"equalizer=f=110:width_type=o:width=1.2:g=1.8,"
        f"equalizer=f=3200:width_type=o:width=1.5:g=-1.5,"
        f"equalizer=f=6500:width_type=o:width=1.5:g=1.5,"
        f"aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
        f"aecho=0.85:0.88:14|21:0.16|0.13,extrastereo=m=1.35,"
        f"acompressor=threshold=-16dB:ratio=2.8:attack=15:release=180:makeup=2,"
        f"dynaudnorm=f=200:g=15:p=0.88:m=6.0"
    )

    if has_audio_stream(raw_path):
        cmd = [
            "ffmpeg", "-y", "-i", str(raw_path),
            "-vf", vf,
            "-af", af_chain,
            "-c:v", "libx264", "-preset", "fast", "-crf", "14",
            "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-ac", "2",
            str(out_path)
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-i", str(raw_path),
            "-f", "lavfi", "-t", str(trim_dur), "-i", "anoisesrc=d=" + str(trim_dur) + ":c=pink:r=48000:a=0.002",
            "-vf", vf,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "fast", "-crf", "14",
            "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-ac", "2",
            str(out_path)
        ]

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path.exists() and out_path.stat().st_size > 50_000


def get_video_duration(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(res.stdout.strip())
    except Exception:
        return 10.0


def render_xfade_batch(clip_paths: list[Path], out_path: Path, fade_dur: float = 0.5):
    if len(clip_paths) == 1:
        shutil.copy2(clip_paths[0], out_path)
        return

    inputs = []
    durations = []
    for p in clip_paths:
        inputs.extend(["-i", str(p)])
        durations.append(get_video_duration(p))

    v_filters = []
    a_filters = []
    offset = durations[0] - fade_dur

    for i in range(1, len(clip_paths)):
        v_in1 = f"[0:v]" if i == 1 else f"[v{i-1}]"
        v_in2 = f"[{i}:v]"
        v_out = f"[vout]" if i == len(clip_paths) - 1 else f"[v{i}]"
        v_filters.append(
            f"{v_in1}{v_in2}xfade=transition=fade:duration={fade_dur}:offset={offset:.3f}{v_out}"
        )

        a_in1 = f"[0:a]" if i == 1 else f"[a{i-1}]"
        a_in2 = f"[{i}:a]"
        a_out = f"[aout]" if i == len(clip_paths) - 1 else f"[a{i}]"
        a_filters.append(
            f"{a_in1}{a_in2}acrossfade=d={fade_dur}:c1=tri:c2=tri{a_out}"
        )

        if i < len(clip_paths) - 1:
            offset += durations[i] - fade_dur

    filter_complex = ";".join(v_filters + a_filters)
    cmd = [
        "ffmpeg", "-y", *inputs,
        "-filter_complex", filter_complex,
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "12",
        "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-ac", "2",
        str(out_path)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def get_cultural_music_expr(music_style: str) -> tuple[str, str]:
    """
    Returns (left_channel_expr, right_channel_expr) for aevalsrc at 48kHz.
    Designed specifically after viral 4K Festival & Parade ambient/harmonic beds:
    warm, non-intrusive harmonic chords + subtle ceremonial pulse that glues 180 scenes together.
    """
    if music_style == "aegean_maritime_festive":
        # D Major / Open Aegean Fifths (D3=146.83, A3=220, D4=293.66, F#4=369.99, A4=440)
        l_expr = (
            "0.22*sin(2*PI*146.83*t)*(0.75+0.25*sin(2*PI*0.18*t))"
            "+0.20*sin(2*PI*220.00*t)*(0.75+0.25*cos(2*PI*0.14*t))"
            "+0.16*sin(2*PI*293.66*t)*(0.70+0.30*sin(2*PI*0.11*t))"
            "+0.10*sin(2*PI*369.99*t)*(0.60+0.40*sin(2*PI*0.09*t))"
        )
        r_expr = (
            "0.22*sin(2*PI*147.00*t)*(0.75+0.25*cos(2*PI*0.17*t))"
            "+0.20*sin(2*PI*220.25*t)*(0.75+0.25*sin(2*PI*0.15*t))"
            "+0.16*sin(2*PI*293.90*t)*(0.70+0.30*cos(2*PI*0.12*t))"
            "+0.10*sin(2*PI*440.00*t)*(0.60+0.40*cos(2*PI*0.08*t))"
        )
    elif music_style == "christmas_symphonic_chimes":
        # Warm C Major 9th Holiday Pad + Shimmering Chime Overtones (C4=261.63, E4=329.63, G4=392.0, B4=493.88, D5=587.33)
        l_expr = (
            "0.22*sin(2*PI*130.81*t)*(0.80+0.20*sin(2*PI*0.12*t))"
            "+0.18*sin(2*PI*196.00*t)*(0.75+0.25*cos(2*PI*0.15*t))"
            "+0.16*sin(2*PI*261.63*t)*(0.70+0.30*sin(2*PI*0.20*t))"
            "+0.12*sin(2*PI*329.63*t)*(0.65+0.35*cos(2*PI*0.18*t))"
            "+0.07*sin(2*PI*493.88*t)*(0.50+0.50*sin(2*PI*0.25*t))"
        )
        r_expr = (
            "0.22*sin(2*PI*131.00*t)*(0.80+0.20*cos(2*PI*0.11*t))"
            "+0.18*sin(2*PI*196.25*t)*(0.75+0.25*sin(2*PI*0.14*t))"
            "+0.16*sin(2*PI*261.90*t)*(0.70+0.30*cos(2*PI*0.19*t))"
            "+0.12*sin(2*PI*392.00*t)*(0.65+0.35*sin(2*PI*0.17*t))"
            "+0.07*sin(2*PI*587.33*t)*(0.50+0.50*cos(2*PI*0.24*t))"
        )
    elif music_style == "nordic_choral_winter":
        # Luminous Nordic F Major Cathedral Pad (F3=174.61, C4=261.63, F4=349.23, A4=440, C5=523.25)
        l_expr = (
            "0.24*sin(2*PI*174.61*t)*(0.80+0.20*sin(2*PI*0.10*t))"
            "+0.20*sin(2*PI*261.63*t)*(0.75+0.25*cos(2*PI*0.13*t))"
            "+0.15*sin(2*PI*349.23*t)*(0.70+0.30*sin(2*PI*0.09*t))"
            "+0.10*sin(2*PI*440.00*t)*(0.65+0.35*cos(2*PI*0.16*t))"
        )
        r_expr = (
            "0.24*sin(2*PI*174.80*t)*(0.80+0.20*cos(2*PI*0.11*t))"
            "+0.20*sin(2*PI*261.85*t)*(0.75+0.25*sin(2*PI*0.12*t))"
            "+0.15*sin(2*PI*349.50*t)*(0.70+0.30*cos(2*PI*0.10*t))"
            "+0.10*sin(2*PI*523.25*t)*(0.65+0.35*sin(2*PI*0.15*t))"
        )
    elif music_style == "thai_royal_victory_march":
        # Majestic Royal Thai Deep Bronze Gong & Fifths (G2=98.0, D3=146.83, G3=196.0, B3=246.94, D4=293.66)
        l_expr = (
            "0.25*sin(2*PI*98.00*t)*(0.75+0.25*sin(2*PI*0.15*t))"
            "+0.22*sin(2*PI*146.83*t)*(0.75+0.25*cos(2*PI*0.12*t))"
            "+0.18*sin(2*PI*196.00*t)*(0.70+0.30*sin(2*PI*0.10*t))"
            "+0.12*sin(2*PI*293.66*t)*(0.65+0.35*cos(2*PI*0.18*t))"
        )
        r_expr = (
            "0.25*sin(2*PI*98.18*t)*(0.75+0.25*cos(2*PI*0.14*t))"
            "+0.22*sin(2*PI*147.05*t)*(0.75+0.25*sin(2*PI*0.13*t))"
            "+0.18*sin(2*PI*196.22*t)*(0.70+0.30*cos(2*PI*0.11*t))"
            "+0.12*sin(2*PI*246.94*t)*(0.65+0.35*sin(2*PI*0.17*t))"
        )
    else:
        # thai_lanna_pentatonic_gold & thai_classical_royal_court
        # Golden Thai Pentatonic Silk & Chime Harmonics (C3=130.81, G3=196.0, A3=220.0, C4=261.63, D4=293.66, E4=329.63, G4=392.0)
        l_expr = (
            "0.22*sin(2*PI*130.81*t)*(0.80+0.20*sin(2*PI*0.11*t))"
            "+0.20*sin(2*PI*196.00*t)*(0.75+0.25*cos(2*PI*0.14*t))"
            "+0.16*sin(2*PI*261.63*t)*(0.70+0.30*sin(2*PI*0.17*t))"
            "+0.12*sin(2*PI*293.66*t)*(0.65+0.35*cos(2*PI*0.13*t))"
            "+0.08*sin(2*PI*392.00*t)*(0.60+0.40*sin(2*PI*0.19*t))"
        )
        r_expr = (
            "0.22*sin(2*PI*131.00*t)*(0.80+0.20*cos(2*PI*0.10*t))"
            "+0.20*sin(2*PI*220.00*t)*(0.75+0.25*sin(2*PI*0.13*t))"
            "+0.16*sin(2*PI*261.85*t)*(0.70+0.30*cos(2*PI*0.16*t))"
            "+0.12*sin(2*PI*329.63*t)*(0.65+0.35*sin(2*PI*0.12*t))"
            "+0.08*sin(2*PI*392.30*t)*(0.60+0.40*cos(2*PI*0.18*t))"
        )
    return l_expr, r_expr


def build_seo_metadata(project: dict, total_duration: float, qa_summary: dict) -> dict:
    title = project["title"]
    loc = project["location"]
    mins = int(total_duration // 60)
    chapters = []
    step = max(180, int(total_duration // 8))
    chapter_titles = [
        "Grand Opening & Ceremonial Vanguard",
        "Main Boulevard Float Procession",
        "Traditional Dancers & Cultural Ensemble",
        "Golden Royal Chariot & Centerpiece Float",
        "Marching Bands & Festive Street Performers",
        "Panoramic Crowd & Cultural Costumes Showcase",
        "Grand Finale Cavalcade & Closing Ceremony"
    ]
    cur_t = 0
    for idx_c, ch_name in enumerate(chapter_titles):
        if cur_t >= total_duration - 30 and idx_c > 0:
            break
        mm = int(cur_t // 60)
        ss = int(cur_t % 60)
        chapters.append(f"{mm:02d}:{ss:02d} - {ch_name}")
        cur_t += step

    return {
        "youtube_title": f"[4K 60FPS ULTRA HD] {title} | Grand Cultural Street Parade ({mins} Mins)",
        "location": loc,
        "music_and_audio_profile": {
            "style": project["music_style"],
            "spatial_audio": "48kHz True Stereo Spatial Widening (Haas + Mid/Side) + Culturally Matched Festive Harmonic Bed (11%)",
            "loudness_standard": "-14.0 LUFS / -1.5 dBFS True Peak (320 kbps AAC)"
        },
        "video_master_specs": {
            "resolution": "3840x2160 (4K UHD)",
            "fps": 60,
            "codec": "H.264 High@L5.1 (CRF 15, Maxrate 45 Mbps, Lanczos + CAS + 3D Denoise)",
            "duration_seconds": round(total_duration, 2),
            "duration_minutes": round(total_duration / 60.0, 2)
        },
        "chapters": chapters,
        "tags": [
            "4K 60fps Parade", "Ultra HD Cultural Festival", title, loc,
            "4K Walking Tour", "Street Cavalcade 4K", "Grand Procession 4K 60fps",
            "Spatial Stereo Parade Audio", "Festival Documentary 4K"
        ],
        "qa_summary": qa_summary
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python festival_parade_master_pipeline.py <project_idx_1_to_8>")
        sys.exit(1)

    proj_idx = int(sys.argv[1])
    manifest_path = Path(__file__).parent / "parade_manifest.json"
    manifest = json.load(open(manifest_path, encoding="utf-8"))

    project = next((p for p in manifest if p["idx"] == proj_idx), None)
    if not project:
        print(f"Project index {proj_idx} not found in parade_manifest.json")
        sys.exit(1)

    slug = project["slug"]
    scenes = project["scenes"]
    print(f"============================================================")
    print(f"🚀 STARTING 4K 60FPS MAX CINEMA PARADE PIPELINE: [{proj_idx}] {slug}")
    print(f"   Title: {project['title']}")
    print(f"   Scenes: {len(scenes)} | Music Bed: {project['music_style']}")
    print(f"============================================================")

    work_dir = Path("work_parade") / slug
    raw_dir = work_dir / "raw"
    norm_dir = work_dir / "norm"
    batch_dir = work_dir / "batches"
    out_dir = Path("output_parade")
    for d in [raw_dir, norm_dir, batch_dir, out_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Download all scenes with controlled concurrency (max_workers=3)
    print(f"\n[Stage 1/4] Downloading {len(scenes)} raw clips from Google Drive...")
    downloaded_scenes = []

    def _dl_task(s):
        p = raw_dir / f"scene_{s['num']:03d}.mp4"
        ok = download_gdrive_file(s["id"], p)
        return (s, p, ok)

    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = [ex.submit(_dl_task, s) for s in scenes]
        for fut in as_completed(futs):
            s, p, ok = fut.result()
            if ok:
                downloaded_scenes.append((s["num"], s, p))
            else:
                print(f"  ⚠️ WARNING: Scene {s['num']} ({s['id']}) failed to download after retries.")

    downloaded_scenes.sort(key=lambda x: x[0])
    print(f"  ✅ Downloaded {len(downloaded_scenes)}/{len(scenes)} scenes successfully.")
    if len(downloaded_scenes) < 10:
        raise RuntimeError(f"Too few scenes downloaded ({len(downloaded_scenes)}).")

    # 2. Inspect, QA, and normalize all scenes in parallel (max_workers=4)
    print(f"\n[Stage 2/4] OpenCV Auto-QA, 3D Denoise, Color Grading & 48kHz Stereo Mastering...")
    norm_paths = [None] * len(downloaded_scenes)
    qa_records = [None] * len(downloaded_scenes)

    def _norm_task(pos_idx, s_num, s_meta, r_path):
        qa = inspect_and_calibrate_clip(r_path)
        n_path = norm_dir / f"norm_{pos_idx:03d}.mp4"
        normalize_parade_scene(pos_idx, r_path, n_path, qa)
        # Remove raw clip immediately to save disk space
        try:
            r_path.unlink(missing_ok=True)
        except Exception:
            pass
        qa["scene_num"] = s_num
        return (pos_idx, n_path, qa)

    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [
            ex.submit(_norm_task, idx, item[0], item[1], item[2])
            for idx, item in enumerate(downloaded_scenes)
        ]
        done_cnt = 0
        for fut in as_completed(futs):
            pos_idx, n_path, qa = fut.result()
            norm_paths[pos_idx] = n_path
            qa_records[pos_idx] = qa
            done_cnt += 1
            if done_cnt % 20 == 0 or done_cnt == len(downloaded_scenes):
                print(f"  -> Normalized {done_cnt}/{len(downloaded_scenes)} scenes...")

    # 3. Hierarchical xfade + acrossfade in batches of 25 scenes
    print(f"\n[Stage 3/4] Rendering 0.5s Crossfade Batches across {len(norm_paths)} scenes...")
    batch_size = 25
    batch_files = []
    for b_idx in range(0, len(norm_paths), batch_size):
        chunk = norm_paths[b_idx:b_idx + batch_size]
        b_out = batch_dir / f"batch_{b_idx // batch_size:02d}.mp4"
        print(f"  -> Stitching batch {b_idx // batch_size + 1} ({len(chunk)} scenes)...")
        render_xfade_batch(chunk, b_out, fade_dur=0.5)
        batch_files.append(b_out)
        # Free normalized scene files as soon as batch is created
        for p in chunk:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass

    # Stitch all batches together with 0.5s xfade into pre-upscale master
    pre_master = work_dir / "pre_master_720p60.mp4"
    print(f"  -> Joining {len(batch_files)} batches with 0.5s xfade...")
    render_xfade_batch(batch_files, pre_master, fade_dur=0.5)
    for b_f in batch_files:
        try:
            b_f.unlink(missing_ok=True)
        except Exception:
            pass

    total_dur = get_video_duration(pre_master)
    print(f"  ✅ Pre-master stitched! Total Duration: {total_dur:.2f}s ({total_dur/60:.2f} mins)")

    # 4. Final 4K 60fps MAX CINEMA Upscale + Cultural Stereo Harmonic Bed Mix
    final_mp4 = out_dir / f"{slug}_4K60_MAX_CINEMA.mp4"
    print(f"\n[Stage 4/4] Mastering 4K 60fps MAX CINEMA (CRF 15, Maxrate 45M, Lanczos + CAS) + Cultural Music Bed...")
    l_expr, r_expr = get_cultural_music_expr(project["music_style"])

    # Filter complex:
    # Video: 4K Lanczos accurate rounding + Contrast Adaptive Sharpening (0.55) + Fine Unsharp (0.55)
    # Audio: Mix conditioned 48kHz stereo parade audio (100%) + culturally matched stereo harmonic bed (11%) + Broadcast Limiter
    filter_complex = (
        f"[0:v]scale=3840:2160:flags=lanczos+accurate_rnd+full_chroma_int,"
        f"cas=strength=0.55,unsharp=3:3:0.55:3:3:0.0,format=yuv420p[vout];"
        f"aevalsrc='{l_expr}|{r_expr}':s=48000:d={total_dur:.3f}[bed_raw];"
        f"[bed_raw]lowpass=f=1800,highpass=f=70,aecho=0.8:0.88:60|90:0.35|0.25,volume=0.11[bed];"
        f"[0:a][bed]amix=inputs=2:duration=first:dropout_transition=2:weights='1.0 0.11',"
        f"alimiter=limit=0.86:attack=5:release=50[aout]"
    )

    cmd_master = [
        "ffmpeg", "-y", "-i", str(pre_master),
        "-filter_complex", filter_complex,
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "superfast",
        "-crf", "15", "-maxrate", "45M", "-bufsize", "90M",
        "-profile:v", "high", "-level:v", "5.1",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", "320k", "-ar", "48000", "-ac", "2",
        str(final_mp4)
    ]
    subprocess.run(cmd_master, check=True)

    # Remove pre_master immediately to free disk space before splitting
    try:
        pre_master.unlink(missing_ok=True)
    except Exception:
        pass

    final_size_gb = final_mp4.stat().st_size / (1024 ** 3)
    print(f"  🎉 4K 60FPS MAX CINEMA COMPLETE: {final_mp4.name} ({final_size_gb:.2f} GB)")

    # Write QA Report & SEO Metadata JSON
    qa_summary = {
        "project_idx": proj_idx,
        "slug": slug,
        "scenes_total": len(scenes),
        "scenes_rendered": len(downloaded_scenes),
        "duration_seconds": round(total_dur, 2),
        "final_size_gb": round(final_size_gb, 3),
        "avg_luma": round(float(np.mean([q["mean_luma"] for q in qa_records])), 2),
        "avg_sharpness": round(float(np.mean([q["mean_sharpness"] for q in qa_records])), 2),
        "music_style": project["music_style"]
    }
    seo_meta = build_seo_metadata(project, total_dur, qa_summary)

    seo_path = out_dir / f"{slug}_SEO_meta.json"
    qa_path = out_dir / f"{slug}_QA_Report.json"
    with open(seo_path, "w", encoding="utf-8") as f:
        json.dump(seo_meta, f, indent=2, ensure_ascii=False)
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump({"summary": qa_summary, "scenes": qa_records}, f, indent=2, ensure_ascii=False)

    print(f"  📄 Saved SEO & QA Reports: {seo_path.name}, {qa_path.name}")


if __name__ == "__main__":
    main()
