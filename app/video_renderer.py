"""FFmpeg-based clip slicer, face-tracked 9:16 reframer, face-tracked streamer PiP, and ASS caption burner."""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.face_tracker import (
    FaceCropTrack,
    analyze_clip_faces,
    build_ffmpeg_lerp_expr,
)
from app.models import AspectRatio, CornerPosition, QualityPreset
from app.youtube_service import find_ffmpeg_dir


def _tool_exe(name: str) -> str:
    ffdir = find_ffmpeg_dir()
    if ffdir:
        cand = Path(ffdir) / f"{name}.exe"
        if cand.exists():
            return str(cand)
    return name


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
    audio_boost: bool = True,
) -> None:
    """Cut clip, run offline face tracking for main 9:16 + streamer PiP, boost audio, and burn ASS captions."""
    w_out, h_out = target_dimensions(aspect_ratio, quality, src_info=src_info)
    clip_dur = max(0.5, end - start)
    ffmpeg = _tool_exe("ffmpeg")

    # Run fast 2-fps offline YuNet face tracking for corner PiP and/or main 9:16 framing
    track_main = aspect_ratio == AspectRatio.VERTICAL_CROP and (src_info.width / max(1, src_info.height)) > (9.0 / 16.0)
    corner_track, main_track = analyze_clip_faces(
        video_path=source_video,
        src_w=src_info.width,
        src_h=src_info.height,
        start=start,
        end=end,
        streamer_cam_source=streamer_cam_source,
        track_main=track_main,
    )

    ass_filter = f",ass={ass_path.name}" if ass_path and ass_path.exists() else ""
    main_chain, uses_split_bg = _build_main_filter_chain(src_info, w_out, h_out, aspect_ratio, main_track=main_track)

    # Only render corner PiP when a face is actually detected (prevents pasting blank HUD/chair boxes)
    show_pip = bool(corner_track and corner_track.found_faces)
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

        # Fill the Cam Frame cleanly edge-to-edge without distortion or blurred dead bars
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
            fc = (
                f"[0:v]split=3[bgsrc][fgsrc][camsrc];"
                f"[bgsrc]scale={w_out}:{h_out}:force_original_aspect_ratio=increase,"
                f"crop={w_out}:{h_out},boxblur=22:6[bg];"
                f"[fgsrc]scale={w_out}:{h_out}:force_original_aspect_ratio=decrease[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[base];"
                f"[camsrc]{pip_chain}[pip];"
                f"[base][pip]overlay={overlay_xy}{enable_opt}{ass_filter}[vout]"
            )
        else:
            fc = (
                f"[0:v]split=2[mainsrc][camsrc];"
                f"[mainsrc]{main_chain}[base];"
                f"[camsrc]{pip_chain}[pip];"
                f"[base][pip]overlay={overlay_xy}{enable_opt}{ass_filter}[vout]"
            )
        filter_args = ["-filter_complex", fc, "-map", "[vout]"]
        if src_info.has_audio:
            filter_args += ["-map", "0:a?"]
    elif uses_split_bg:
        fc = f"[0:v]{main_chain}{ass_filter}[vout]"
        filter_args = ["-filter_complex", fc, "-map", "[vout]"]
        if src_info.has_audio:
            filter_args += ["-map", "0:a?"]
    else:
        filter_args = ["-vf", f"{main_chain}{ass_filter}"]

    cmd = [
        ffmpeg,
        "-v", "error",
        "-y",
        "-ss", f"{start:.3f}",
        "-i", str(source_video.resolve()),
        "-t", f"{clip_dur:.3f}",
        *filter_args,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "21",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
    ]
    if src_info.has_audio:
        if audio_boost:
            cmd += ["-af", "acompressor=threshold=-18dB:ratio=3:attack=10:release=120,volume=1.35,alimiter=limit=0.95"]
        cmd += ["-c:a", "aac", "-b:a", "160k", "-ar", "48000"]
    else:
        cmd += ["-an"]

    cmd.append(str(output_mp4.resolve()))

    proc = subprocess.run(cmd, cwd=str(work_dir), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg render failed: {proc.stderr.strip()[:500]}")

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
