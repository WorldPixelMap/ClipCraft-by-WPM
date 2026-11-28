"""FFmpeg-based clip slicer, face-tracked 9:16 reframer, face-tracked streamer PiP, and ASS caption burner."""

from __future__ import annotations

import json
import math
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.face_tracker import (
    FaceCropTrack,
    analyze_clip_faces,
    build_ffmpeg_lerp_expr,
)
from app.models import AspectRatio, CornerPosition, EncoderMode, QualityPreset, SplitCamPosition
from app.youtube_service import find_ffmpeg_dir


def _tool_exe(name: str) -> str:
    ffdir = find_ffmpeg_dir()
    if ffdir:
        cand = Path(ffdir) / f"{name}.exe"
        if cand.exists():
            return str(cand)
    return name


_GPU_CACHE_LOCK = threading.Lock()
_CACHED_GPU_ENCODERS: list[str] | None = None


def detect_available_encoders() -> list[str]:
    """Probe which hardware H.264 GPU encoders actually initialize on this PC (nvenc, qsv, amf)."""
    global _CACHED_GPU_ENCODERS
    if _CACHED_GPU_ENCODERS is not None:
        return list(_CACHED_GPU_ENCODERS)

    with _GPU_CACHE_LOCK:
        if _CACHED_GPU_ENCODERS is not None:
            return list(_CACHED_GPU_ENCODERS)
        ffmpeg = _tool_exe("ffmpeg")
        found: list[str] = []
        try:
            list_res = subprocess.run(
                [ffmpeg, "-hide_banner", "-encoders"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
                check=False,
            )
            txt = list_res.stdout or ""
            for short, codec in (("nvenc", "h264_nvenc"), ("qsv", "h264_qsv"), ("amf", "h264_amf")):
                if codec not in txt:
                    continue
                probe = subprocess.run(
                    [
                        ffmpeg, "-v", "error", "-y",
                        "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.1",
                        "-frames:v", "1", "-c:v", codec, "-f", "null", "-",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=6,
                    check=False,
                )
                if probe.returncode == 0:
                    found.append(short)
        except Exception:
            pass
        _CACHED_GPU_ENCODERS = found
        return list(found)


def _encoder_args(mode: EncoderMode) -> tuple[list[str], str]:
    """Resolve FFmpeg video encoder arguments for the requested CPU/GPU mode."""
    available = detect_available_encoders()
    chosen = "cpu"
    if mode == EncoderMode.AUTO:
        chosen = available[0] if available else "cpu"
    elif mode != EncoderMode.CPU and mode.value in available:
        chosen = mode.value

    if chosen == "nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "22"], "h264_nvenc"
    if chosen == "qsv":
        return ["-c:v", "h264_qsv", "-preset", "veryfast", "-global_quality", "22"], "h264_qsv"
    if chosen == "amf":
        return ["-c:v", "h264_amf", "-quality", "speed", "-rc", "cqp", "-qp_i", "21", "-qp_p", "21"], "h264_amf"
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21"], "libx264"


def _run_cancellable(
    cmd: list[str],
    cwd: Path | None = None,
    cancel_event: threading.Event | None = None,
    proc_cb: Callable[[subprocess.Popen | None], None] | None = None,
) -> tuple[int, str]:
    """Run an FFmpeg command while supporting instant user cancellation."""
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc_cb:
        proc_cb(proc)
    try:
        while proc.poll() is None:
            if cancel_event and cancel_event.is_set():
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise RuntimeError("Job cancelled by user.")
            time.sleep(0.15)
        _, stderr_txt = proc.communicate()
        return proc.returncode or 0, stderr_txt or ""
    finally:
        if proc_cb:
            proc_cb(None)


@dataclass
class VideoInfo:
    width: int
    height: int
    duration: float
    has_audio: bool


def probe_video(video_path: Path) -> VideoInfo:
    """Inspect source resolution and audio presence using ffprobe."""
    cmd = [
        _tool_exe("ffprobe"),
        "-v", "error",
        "-show_entries", "stream=codec_type,width,height:format=duration",
        "-of", "json",
        str(video_path),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    data = json.loads(res.stdout)
    width, height, has_audio = 1280, 720, False
    for st in data.get("streams") or []:
        if st.get("codec_type") == "video" and st.get("width") and st.get("height"):
            width = int(st["width"])
            height = int(st["height"])
        elif st.get("codec_type") == "audio":
            has_audio = True
    duration = float((data.get("format") or {}).get("duration") or 0.0)
    return VideoInfo(width=width, height=height, duration=duration, has_audio=has_audio)


def target_dimensions(
    aspect_ratio: AspectRatio,
    quality: QualityPreset,
    src_info: VideoInfo | None = None,
) -> tuple[int, int]:
    """Return output (width, height) as even dimensions."""
    if aspect_ratio == AspectRatio.ORIGINAL and src_info is not None:
        w = src_info.width if src_info.width % 2 == 0 else max(2, src_info.width - 1)
        h = src_info.height if src_info.height % 2 == 0 else max(2, src_info.height - 1)
        return (w, h)
    if aspect_ratio == AspectRatio.LANDSCAPE:
        return (1920, 1080) if quality == QualityPreset.HD else (1280, 720)
    return (1080, 1920) if quality == QualityPreset.HD else (720, 1280)


def _even(n: float, minimum: int = 32) -> int:
    val = max(minimum, int(round(n)))
    return val if val % 2 == 0 else val - 1


def _build_main_filter_chain(
    src_info: VideoInfo,
    w_out: int,
    h_out: int,
    aspect_ratio: AspectRatio,
    main_track: FaceCropTrack | None = None,
) -> tuple[str, bool]:
    """Return (filter_body, uses_split_bg) transforming [0:v] into base output stream."""
    if aspect_ratio == AspectRatio.ORIGINAL:
        return (f"scale={w_out}:{h_out}:flags=lanczos,setsar=1", False)

    if aspect_ratio == AspectRatio.VERTICAL_CROP:
        src_ar = src_info.width / max(1, src_info.height)
        if src_ar > (9.0 / 16.0):
            if main_track and main_track.found_faces:
                expr_x = build_ffmpeg_lerp_expr(main_track.keyframes, axis="x")
                return (
                    f"crop={main_track.cw}:{main_track.ch}:'{expr_x}':0,"
                    f"scale={w_out}:{h_out}:flags=lanczos,setsar=1",
                    False,
                )
            return (
                f"crop=trunc(ih*9/16/2)*2:ih:(iw-trunc(ih*9/16/2)*2)/2:0,"
                f"scale={w_out}:{h_out}:flags=lanczos,setsar=1",
                False,
            )
        return (
            f"scale={w_out}:{h_out}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={w_out}:{h_out},setsar=1",
            False,
        )

    if aspect_ratio == AspectRatio.VERTICAL_FIT:
        return (
            f"split=2[bgsrc][fgsrc];"
            f"[bgsrc]scale={w_out}:{h_out}:force_original_aspect_ratio=increase,"
            f"crop={w_out}:{h_out},boxblur=22:6[bg];"
            f"[fgsrc]scale={w_out}:{h_out}:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1",
            True,
        )

    return (
        f"scale={w_out}:{h_out}:force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={w_out}:{h_out}:(ow-iw)/2:(oh-ih)/2:black,setsar=1",
        False,
    )


def _cam_source_crop(src_w: int, src_h: int, corner: CornerPosition) -> str:
    """Fallback static 9:16 portrait crop in the chosen corner when no face was detected."""
    k = max(2, int(round((src_h * 0.27) / 32.0)) * 2)
    while 16 * k > src_h or 9 * k > src_w:
        k = max(2, k - 2)
    cw, ch = 9 * k, 16 * k

    pad_y = max(0, int(round(src_h * 0.015)))
    if corner in (CornerPosition.TOP_LEFT, CornerPosition.BOTTOM_LEFT):
        cx = max(0, min(src_w - cw, int(round(src_w * 0.115 - cw / 2.0))))
    else:
        cx = max(0, min(src_w - cw, int(round(src_w * 0.885 - cw / 2.0))))

    if corner in (CornerPosition.TOP_LEFT, CornerPosition.TOP_RIGHT):
        cy = min(src_h - ch, pad_y)
    else:
        cy = max(0, src_h - ch - pad_y)

    return f"crop={cw}:{ch}:{cx}:{cy}"


def _tracked_cam_crop(track: FaceCropTrack) -> str:
    """Build smooth face-tracked FFmpeg crop filter string with instant snap on hard scene cuts."""
    expr_x = build_ffmpeg_lerp_expr(track.keyframes, axis="x", jump_thresh=max(32.0, track.cw * 0.65))
    expr_y = build_ffmpeg_lerp_expr(track.keyframes, axis="y", jump_thresh=max(32.0, track.ch * 0.65))
    return f"crop={track.cw}:{track.ch}:'{expr_x}':'{expr_y}'"


def _pip_dimensions(w_out: int, h_out: int, crop_ar: float | None = None) -> tuple[int, int]:
    """
    Return compact PiP (width, height) sized proportionally to the output clip.
    When crop_ar is provided from the fitted human face crop, exact-match the aspect ratio
    so the face fills 100% of the Cam Frame with zero blurry filler bars.
    """
    if crop_ar is None or crop_ar <= 0.0:
        pip_w = _even(min(w_out, h_out) * 0.225, 90)
        pip_w = max(90, (pip_w // 18) * 18 or pip_w)
        pip_h = _even(pip_w * 16.0 / 9.0, 160)
        return pip_w, pip_h

    clamped_ar = max(0.5625, min(1.35, float(crop_ar)))
    min_dim = min(w_out, h_out)
    is_vertical = h_out > w_out
    max_w = int(round(w_out * (0.29 if is_vertical else 0.22)))
    max_h = int(round(h_out * (0.23 if is_vertical else 0.34)))

    # Maintain consistent compact PiP visual area (~175x230 for 3:4 on 720x1280)
    target_area = (min_dim * 0.25) ** 2 * 1.32
    raw_w = math.sqrt(target_area * clamped_ar)
    pip_w = _even(min(max_w, max(96.0, raw_w)), 64)
    pip_h = _even(pip_w / clamped_ar, 64)
    if pip_h > max_h:
        pip_h = _even(max_h, 64)
        pip_w = _even(pip_h * clamped_ar, 64)
    return pip_w, pip_h


def _cam_overlay_coords(w_out: int, h_out: int, corner: CornerPosition, aspect_ratio: AspectRatio) -> str:
    """Compute PiP overlay (x:y) coordinates inside safe zone so bottom captions stay unobstructed."""
    mx = max(16, int(round(w_out * 0.035)))
    my_top = max(20, int(round(h_out * 0.035)))
    is_vertical = aspect_ratio != AspectRatio.LANDSCAPE
    my_bot = int(round(h_out * (0.28 if is_vertical else 0.14)))

    if corner == CornerPosition.TOP_LEFT:
        return f"{mx}:{my_top}"
    if corner == CornerPosition.TOP_RIGHT:
        return f"W-w-{mx}:{my_top}"
    if corner == CornerPosition.BOTTOM_LEFT:
        return f"{mx}:H-h-{my_bot}"
    if corner == CornerPosition.BOTTOM_RIGHT:
        return f"W-w-{mx}:H-h-{my_bot}"
    if corner == CornerPosition.TOP_CENTER:
        return f"(W-w)/2:{my_top}"
    if corner == CornerPosition.BOTTOM_CENTER:
        return f"(W-w)/2:H-h-{my_bot}"
    if corner == CornerPosition.MID_LEFT:
        return f"{mx}:(H-h)/2"
    if corner == CornerPosition.MID_RIGHT:
        return f"W-w-{mx}:(H-h)/2"
    return f"W-w-{mx}:{my_top}"


def _corner_placement_from_region(region: CornerPosition) -> CornerPosition:
    """Map any detected source face region to a safe non-obstructing output corner for auto-placement."""
    if region in (
        CornerPosition.TOP_LEFT,
        CornerPosition.TOP_RIGHT,
        CornerPosition.BOTTOM_LEFT,
        CornerPosition.BOTTOM_RIGHT,
    ):
        return region
    if region == CornerPosition.MID_LEFT:
        return CornerPosition.TOP_LEFT
    if region == CornerPosition.BOTTOM_CENTER:
        return CornerPosition.BOTTOM_RIGHT
    return CornerPosition.TOP_RIGHT


def render_clip(
    source_video: Path,
    src_info: VideoInfo,
    output_mp4: Path,
    output_thumb: Path,
    start: float,
    end: float,
    aspect_ratio: AspectRatio,
    quality: QualityPreset,
    ass_path: Path | None,
    work_dir: Path,
    streamer_cam_source: CornerPosition = CornerPosition.OFF,
    streamer_cam_target: CornerPosition = CornerPosition.TOP_RIGHT,
    split_cam_position: SplitCamPosition = SplitCamPosition.TOP,
    audio_boost: bool = True,
    keep_intervals: list[tuple[float, float]] | None = None,
    encoder_mode: EncoderMode = EncoderMode.AUTO,
    bg_music_path: Path | None = None,
    music_volume: float = 0.14,
    cancel_event: threading.Event | None = None,
    proc_cb: Callable[[subprocess.Popen | None], None] | None = None,
) -> str:
    """Cut clip (with optional jump-cut silence removal), reframe/split-screen, track faces, duck music & burn ASS captions."""
    if cancel_event and cancel_event.is_set():
        raise RuntimeError("Job cancelled by user.")

    w_out, h_out = target_dimensions(aspect_ratio, quality, src_info=src_info)
    clip_dur = max(0.5, end - start)
    ffmpeg = _tool_exe("ffmpeg")

    active_source = source_video
    active_info = src_info
    active_start = start
    active_end = end

    # Optional dead-air silence jump-cut pre-pass so face tracking & captions align cleanly on compacted timeline
    has_jumpcuts = False
    if keep_intervals and (
        len(keep_intervals) > 1
        or keep_intervals[0][0] > 0.25
        or keep_intervals[0][1] < clip_dur - 0.35
    ):
        valid_ivs = [(max(0.0, a), min(clip_dur, b)) for a, b in keep_intervals if b > a + 0.15]
        if valid_ivs:
            has_jumpcuts = True
            jumpcut_mp4 = work_dir / f"_jumpcut_{output_mp4.stem}.mp4"
            segs: list[str] = []
            concat_refs: list[str] = []
            for idx, (a, b) in enumerate(valid_ivs):
                segs.append(f"[0:v]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS[jv{idx}]")
                concat_refs.append(f"[jv{idx}]")
                if src_info.has_audio:
                    segs.append(f"[0:a]atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS[ja{idx}]")
                    concat_refs.append(f"[ja{idx}]")
            a_cnt = 1 if src_info.has_audio else 0
            jc_fc = ";".join(segs) + f";{''.join(concat_refs)}concat=n={len(valid_ivs)}:v=1:a={a_cnt}[vout]" + (
                "[aout]" if src_info.has_audio else ""
            )
            jc_cmd = [
                ffmpeg, "-v", "error", "-y",
                "-ss", f"{start:.3f}",
                "-i", str(source_video.resolve()),
                "-t", f"{clip_dur:.3f}",
                "-filter_complex", jc_fc,
                "-map", "[vout]",
                *(["-map", "[aout]", "-c:a", "aac", "-b:a", "192k"] if src_info.has_audio else ["-an"]),
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "15", "-pix_fmt", "yuv420p",
                str(jumpcut_mp4.resolve()),
            ]
            rc, err = _run_cancellable(jc_cmd, cwd=work_dir, cancel_event=cancel_event, proc_cb=proc_cb)
            if rc == 0 and jumpcut_mp4.exists():
                active_source = jumpcut_mp4
                active_info = probe_video(jumpcut_mp4)
                active_start = 0.0
                active_end = max(0.5, active_info.duration)
                clip_dur = active_end
            else:
                has_jumpcuts = False

    # Run fast offline YuNet face tracking for corner PiP / split-screen and/or main 9:16 framing
    track_main = aspect_ratio == AspectRatio.VERTICAL_CROP and (active_info.width / max(1, active_info.height)) > (9.0 / 16.0)
    cam_src_effective = (
        CornerPosition.AUTO
        if (aspect_ratio == AspectRatio.SPLIT_SCREEN and streamer_cam_source == CornerPosition.OFF)
        else streamer_cam_source
    )
    corner_track, main_track = analyze_clip_faces(
        video_path=active_source,
        src_w=active_info.width,
        src_h=active_info.height,
        start=active_start,
        end=active_end,
        streamer_cam_source=cam_src_effective,
        track_main=track_main,
    )

    ass_filter = f",ass={ass_path.name}" if ass_path and ass_path.exists() else ""
    use_music = bool(bg_music_path and bg_music_path.exists() and music_volume > 0.0)
    video_fc = ""
    uses_split_bg = False

    if aspect_ratio == AspectRatio.SPLIT_SCREEN:
        half_h = _even(h_out / 2.0, 64)
        bot_h = h_out - half_h
        cam_on_bottom = split_cam_position == SplitCamPosition.BOTTOM
        cam_h = bot_h if cam_on_bottom else half_h
        content_h = half_h if cam_on_bottom else bot_h
        if corner_track and corner_track.found_faces:
            cam_src_crop = _tracked_cam_crop(corner_track)
        else:
            cw = _even(active_info.width * 0.65, 64)
            ch = _even(active_info.height * 0.65, 64)
            cx = max(0, (active_info.width - cw) // 2)
            cam_src_crop = f"crop={cw}:{ch}:{cx}:0"
        stack_order = "[content][cam]" if cam_on_bottom else "[cam][content]"
        video_fc = (
            f"[0:v]split=3[camsrc][cbgsrc][cfgsrc];"
            f"[camsrc]{cam_src_crop},scale={w_out}:{cam_h}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={w_out}:{cam_h}:(iw-ow)/2:(ih-oh)/2,setsar=1[cam];"
            f"[cbgsrc]scale={w_out}:{content_h}:force_original_aspect_ratio=increase,"
            f"crop={w_out}:{content_h},boxblur=18:5,eq=brightness=-0.12[cbg];"
            f"[cfgsrc]scale={w_out}:{content_h}:force_original_aspect_ratio=decrease:flags=lanczos[cfg];"
            f"[cbg][cfg]overlay=(W-w)/2:(H-h)/2,setsar=1[content];"
            f"{stack_order}vstack=inputs=2{ass_filter}[vout]"
        )
    else:
        main_chain, uses_split_bg = _build_main_filter_chain(
            active_info, w_out, h_out, aspect_ratio, main_track=main_track
        )
        show_pip = bool(corner_track and corner_track.found_faces and streamer_cam_source != CornerPosition.OFF)
        cam_crop = _tracked_cam_crop(corner_track) if show_pip and corner_track else ""

        if show_pip and corner_track:
            crop_ar = corner_track.cw / max(1, corner_track.ch)
            pip_w, pip_h = _pip_dimensions(w_out, h_out, crop_ar=crop_ar)
            target_corner = (
                _corner_placement_from_region(corner_track.detected_corner)
                if streamer_cam_target == CornerPosition.AUTO
                else streamer_cam_target
            )
            if target_corner in (CornerPosition.OFF, CornerPosition.AUTO):
                target_corner = CornerPosition.TOP_RIGHT
            overlay_xy = _cam_overlay_coords(w_out, h_out, target_corner, aspect_ratio)

            pip_ar = pip_w / max(1, pip_h)
            if abs(crop_ar - pip_ar) <= 0.035:
                pip_chain = (
                    f"{cam_crop},scale={pip_w}:{pip_h}:flags=lanczos,"
                    f"pad=iw+4:ih+4:2:2:color=white@0.92,pad=iw+4:ih+4:2:2:color=black@0.45,setsar=1"
                )
            else:
                pip_chain = (
                    f"{cam_crop},scale={pip_w}:{pip_h}:force_original_aspect_ratio=increase:flags=lanczos,"
                    f"crop={pip_w}:{pip_h}:(iw-ow)/2:(ih-oh)/2,"
                    f"pad=iw+4:ih+4:2:2:color=white@0.92,pad=iw+4:ih+4:2:2:color=black@0.45,setsar=1"
                )

            enable_opt = ""
            if corner_track.active_intervals and not (
                len(corner_track.active_intervals) == 1
                and corner_track.active_intervals[0][0] <= 0.05
                and corner_track.active_intervals[0][1] >= clip_dur - 0.5
            ):
                expr = "+".join(f"between(t,{a:.2f},{b:.2f})" for a, b in corner_track.active_intervals)
                enable_opt = f":enable='{expr}'"

            if uses_split_bg:
                video_fc = (
                    f"[0:v]split=3[bgsrc][fgsrc][camsrc];"
                    f"[bgsrc]scale={w_out}:{h_out}:force_original_aspect_ratio=increase,"
                    f"crop={w_out}:{h_out},boxblur=22:6[bg];"
                    f"[fgsrc]scale={w_out}:{h_out}:force_original_aspect_ratio=decrease[fg];"
                    f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[base];"
                    f"[camsrc]{pip_chain}[pip];"
                    f"[base][pip]overlay={overlay_xy}{enable_opt}{ass_filter}[vout]"
                )
            else:
                video_fc = (
                    f"[0:v]split=2[mainsrc][camsrc];"
                    f"[mainsrc]{main_chain}[base];"
                    f"[camsrc]{pip_chain}[pip];"
                    f"[base][pip]overlay={overlay_xy}{enable_opt}{ass_filter}[vout]"
                )
        elif uses_split_bg or use_music:
            video_fc = f"[0:v]{main_chain}{ass_filter}[vout]"

    audio_fc = ""
    if use_music:
        vol = max(0.01, min(0.6, float(music_volume)))
        if active_info.has_audio:
            voice_fx = (
                "[0:a]aformat=sample_rates=48000:channel_layouts=stereo,acompressor=threshold=-18dB:ratio=3:attack=10:release=120,volume=1.35,alimiter=limit=0.95"
                if audio_boost
                else "[0:a]aformat=sample_rates=48000:channel_layouts=stereo"
            )
            audio_fc = (
                f";{voice_fx},asplit=2[vmix][vsc];"
                f"[1:a]atrim=0:{clip_dur:.3f},asetpts=PTS-STARTPTS,aformat=sample_rates=48000:channel_layouts=stereo,volume={vol:.3f}[mus];"
                f"[mus][vsc]sidechaincompress=threshold=0.03:ratio=5:attack=25:release=260[ducked];"
                f"[vmix][ducked]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[aout]"
            )
        else:
            audio_fc = (
                f";[1:a]atrim=0:{clip_dur:.3f},asetpts=PTS-STARTPTS,"
                f"aformat=sample_rates=48000:channel_layouts=stereo,volume={vol:.3f},alimiter=limit=0.95[aout]"
            )

    if video_fc:
        filter_args = ["-filter_complex", f"{video_fc}{audio_fc}", "-map", "[vout]"]
        if use_music:
            filter_args += ["-map", "[aout]"]
        elif active_info.has_audio:
            filter_args += ["-map", "0:a?"]
    else:
        filter_args = ["-vf", f"{main_chain}{ass_filter}"]

    input_args = ["-ss", f"{active_start:.3f}", "-i", str(active_source.resolve())]
    if use_music and bg_music_path:
        input_args += ["-stream_loop", "-1", "-i", str(bg_music_path.resolve())]
    dur_out_args = [] if has_jumpcuts else ["-t", f"{clip_dur:.3f}"]

    audio_out_args: list[str]
    if use_music:
        audio_out_args = ["-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-shortest"]
    elif active_info.has_audio:
        audio_out_args = []
        if audio_boost:
            audio_out_args += ["-af", "acompressor=threshold=-18dB:ratio=3:attack=10:release=120,volume=1.35,alimiter=limit=0.95"]
        audio_out_args += ["-c:a", "aac", "-b:a", "160k", "-ar", "48000"]
    else:
        audio_out_args = ["-an"]

    enc_flags, enc_name = _encoder_args(encoder_mode)
    base_cmd = [
        ffmpeg, "-v", "error", "-y",
        *input_args,
        *filter_args,
        *enc_flags,
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        *audio_out_args,
        *dur_out_args,
        str(output_mp4.resolve()),
    ]
    rc, err = _run_cancellable(base_cmd, cwd=work_dir, cancel_event=cancel_event, proc_cb=proc_cb)
    if rc != 0 and enc_name != "libx264":
        # Safe fallback to CPU x264 if hardware encoder fails on unusual source formats
        cpu_flags, enc_name = _encoder_args(EncoderMode.CPU)
        fallback_cmd = [
            ffmpeg, "-v", "error", "-y",
            *input_args,
            *filter_args,
            *cpu_flags,
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            *audio_out_args,
            *dur_out_args,
            str(output_mp4.resolve()),
        ]
        rc, err = _run_cancellable(fallback_cmd, cwd=work_dir, cancel_event=cancel_event, proc_cb=proc_cb)
    if rc != 0:
        raise RuntimeError(f"FFmpeg render failed: {err.strip()[:500]}")

    thumb_time = min(max(0.4, clip_dur * 0.25), max(0.4, clip_dur - 0.2))
    thumb_cmd = [
        ffmpeg,
        "-v", "error",
        "-y",
        "-ss", f"{thumb_time:.2f}",
        "-i", str(output_mp4.resolve()),
        "-frames:v", "1",
        "-q:v", "3",
        str(output_thumb.resolve()),
    ]
    subprocess.run(thumb_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return enc_name
