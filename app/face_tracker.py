"""Lightweight offline CPU face detection, webcam bounds estimation, & smooth tracking via OpenCV YuNet."""

from __future__ import annotations

import math
import subprocess
import threading
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.models import CornerPosition
from app.youtube_service import find_ffmpeg_dir

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
YUNET_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
_DETECTOR_LOCK = threading.Lock()


@dataclass
class CropKeyframe:
    t: float
    x: float
    y: float


@dataclass
class FaceCropTrack:
    cw: int
    ch: int
    keyframes: list[CropKeyframe]
    found_faces: bool
    detected_corner: CornerPosition = CornerPosition.OFF
    active_intervals: list[tuple[float, float]] = field(default_factory=list)


def _ffmpeg_exe() -> str:
    ffdir = find_ffmpeg_dir()
    if ffdir:
        cand = Path(ffdir) / "ffmpeg.exe"
        if cand.exists():
            return str(cand)
    return "ffmpeg"


def ensure_yunet_model() -> Path | None:
    """Return local YuNet ONNX path, downloading once (227 KB) only if missing."""
    if YUNET_PATH.exists() and YUNET_PATH.stat().st_size > 100_000:
        return YUNET_PATH
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(YUNET_URL, str(YUNET_PATH))
        return YUNET_PATH if YUNET_PATH.exists() else None
    except Exception:
        return None


def is_face_model_cached() -> bool:
    return YUNET_PATH.exists() and YUNET_PATH.stat().st_size > 100_000


def _create_detector(width: int, height: int, score_thresh: float = 0.45):
    model_path = ensure_yunet_model()
    if not model_path:
        return None
    with _DETECTOR_LOCK:
        return cv2.FaceDetectorYN.create(str(model_path), "", (width, height), score_thresh, 0.3, 25)


def sample_clip_frames(
    video_path: Path,
    src_w: int,
    src_h: int,
    start: float,
    end: float,
    sample_fps: float = 2.0,
    target_w: int = 854,
) -> tuple[list[tuple[float, np.ndarray]], float]:
    """Decode sampled BGR frames across [start, end] at reduced resolution via FFmpeg pipe."""
    clip_dur = max(0.5, end - start)
    scale_w = min(target_w, src_w)
    if scale_w % 2 != 0:
        scale_w -= 1
    scale_factor = scale_w / max(1, src_w)
    scale_h = max(2, int(round(src_h * scale_factor / 2.0)) * 2)

    cmd = [
        _ffmpeg_exe(),
        "-v", "error",
        "-ss", f"{start:.3f}",
        "-i", str(video_path.resolve()),
        "-t", f"{clip_dur:.3f}",
        "-vf", f"fps={sample_fps},scale={scale_w}:{scale_h}:flags=fast_bilinear",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_bytes = scale_w * scale_h * 3
    if proc.returncode != 0 or len(proc.stdout) < frame_bytes:
        return [], 1.0

    n_frames = len(proc.stdout) // frame_bytes
    raw = np.frombuffer(proc.stdout[: n_frames * frame_bytes], dtype=np.uint8)
    frames_arr = raw.reshape((n_frames, scale_h, scale_w, 3))
    out: list[tuple[float, np.ndarray]] = []
    for idx in range(n_frames):
        t_rel = min(clip_dur, idx / sample_fps)
        out.append((t_rel, frames_arr[idx]))
    return out, (src_w / scale_w)


def _in_corner_roi(
    cx: float,
    cy: float,
    w: int,
    h: int,
    corner: CornerPosition,
    strict_outer: bool = True,
) -> bool:
    """Restrict corner search to true outer webcam corners so mid-screen game NPCs don't trigger false hits."""
    x_lim = 0.36 if strict_outer else 0.42
    y_lim = 0.38 if strict_outer else 0.44
    if corner == CornerPosition.TOP_LEFT:
        return cx <= w * x_lim and cy <= h * y_lim
    if corner == CornerPosition.TOP_RIGHT:
        return cx >= w * (1.0 - x_lim) and cy <= h * y_lim
    if corner == CornerPosition.BOTTOM_LEFT:
        return cx <= w * x_lim and cy >= h * (1.0 - y_lim)
    if corner == CornerPosition.BOTTOM_RIGHT:
        return cx >= w * (1.0 - x_lim) and cy >= h * (1.0 - y_lim)
    return True


def _smooth_series(vals: list[float], alpha: float = 0.35) -> list[float]:
    """Median-spike-rejection + bidirectional exponential smoothing for camera stability."""
    if len(vals) <= 2:
        return list(vals)
    arr = np.array(vals, dtype=np.float64)
    med = np.copy(arr)
    for i in range(1, len(arr) - 1):
        med[i] = float(np.median(arr[i - 1 : i + 2]))

    fwd = np.copy(med)
    for i in range(1, len(fwd)):
        fwd[i] = alpha * fwd[i] + (1.0 - alpha) * fwd[i - 1]
    bwd = np.copy(fwd)
    for i in range(len(bwd) - 2, -1, -1):
        bwd[i] = alpha * bwd[i] + (1.0 - alpha) * bwd[i + 1]
    return [float(v) for v in bwd]


def _detect_faces_per_frame(
    sampled: list[tuple[float, np.ndarray]],
    inv_scale: float,
) -> list[tuple[float, list[tuple[float, float, float, float, float]]]]:
    """Run YuNet on sampled frames, returning (t, [(x, y, w, h, score), ...]) in native source pixels."""
    if not sampled:
        return []
    sh, sw = sampled[0][1].shape[:2]
    detector = _create_detector(sw, sh)
    if detector is None:
        return []

    results: list[tuple[float, list[tuple[float, float, float, float, float]]]] = []
    for t_rel, frame in sampled:
        _, raw_faces = detector.detect(frame)
        boxes: list[tuple[float, float, float, float, float]] = []
        if raw_faces is not None:
            for f in raw_faces:
                fx = float(f[0]) * inv_scale
                fy = float(f[1]) * inv_scale
                fw = float(f[2]) * inv_scale
                fh = float(f[3]) * inv_scale
                score = float(f[14]) if len(f) > 14 else 0.7
                if fw >= 18 and fh >= 18:
                    boxes.append((fx, fy, fw, fh, score))
        results.append((t_rel, boxes))
    return results


def _detect_webcam_bounds(
    sampled: list[tuple[float, np.ndarray]],
    hit_indices: list[int],
    inv_scale: float,
    src_w: int,
    src_h: int,
    corner: CornerPosition,
    face_box: tuple[float, float, float, float, float],
) -> tuple[int, int, int, int] | None:
    """
    Detect the sharp outer rectangle (x0, y0, x1, y1) of an overlaid corner webcam box
    so the cropped PiP never leaks gameplay outside the webcam overlay.
    """
    if not sampled or not hit_indices or corner in (CornerPosition.OFF, CornerPosition.AUTO):
        return None

    # Average a small subset of hit frames so moving gameplay softens while static webcam border stays crisp
    pick = hit_indices[: min(8, len(hit_indices))]
    grays = [cv2.cvtColor(sampled[i][1], cv2.COLOR_BGR2GRAY).astype(np.float32) for i in pick]
    mean_gray = np.mean(grays, axis=0)
    sh, sw = mean_gray.shape

    fx, fy, fw, fh, _ = face_box
    f_top = fy / inv_scale
    f_bot = (fy + fh) / inv_scale
    f_left = fx / inv_scale
    f_right = (fx + fw) / inv_scale

    # 1. Find horizontal border edge (y_edge) between face and screen center
    x_strip = slice(int(sw * 0.78), sw) if "right" in corner.value else slice(0, int(sw * 0.22))
    row_grad = np.mean(np.abs(np.diff(mean_gray[:, x_strip], axis=0)), axis=1)

    if "top" in corner.value:
        y_lo = max(int(f_bot * 0.92), int(sh * 0.14))
        y_hi = min(sh - 4, int(sh * 0.45))
        if y_hi <= y_lo + 4:
            return None
        seg = row_grad[y_lo:y_hi]
        peak_val = float(np.max(seg))
        med_val = float(np.median(row_grad[:y_hi])) + 1e-3
        if peak_val < max(8.0, med_val * 2.2):
            return None
        y_edge = int((int(np.argmax(seg)) + y_lo) * inv_scale)
        cam_y0, cam_y1 = 0, max(int(fy + fh + 6), min(src_h, y_edge))
    else:
        y_lo = max(4, int(sh * 0.55))
        y_hi = min(int(f_top * 1.08), int(sh * 0.88))
        if y_hi <= y_lo + 4:
            return None
        seg = row_grad[y_lo:y_hi]
        peak_val = float(np.max(seg))
        med_val = float(np.median(row_grad[y_lo:])) + 1e-3
        if peak_val < max(8.0, med_val * 2.2):
            return None
        y_edge = int((int(np.argmax(seg)) + y_lo) * inv_scale)
        cam_y0, cam_y1 = min(int(max(0.0, fy - 6)), max(0, y_edge)), src_h

    # 2. Find vertical border edge (x_edge) between face and screen center
    y_strip = slice(0, int(sh * 0.22)) if "top" in corner.value else slice(int(sh * 0.78), sh)
    col_grad = np.mean(np.abs(np.diff(mean_gray[y_strip, :], axis=1)), axis=0)

    if "left" in corner.value:
        x_lo = max(int(f_right * 0.92), int(sw * 0.14))
        x_hi = min(sw - 4, int(sw * 0.44))
        if x_hi <= x_lo + 4:
            return None
        cseg = col_grad[x_lo:x_hi]
        if float(np.max(cseg)) < max(7.0, (float(np.median(col_grad[:x_hi])) + 1e-3) * 2.0):
            cam_x0, cam_x1 = 0, min(src_w, int(src_w * 0.36))
        else:
            x_edge = int((int(np.argmax(cseg)) + x_lo) * inv_scale)
            cam_x0, cam_x1 = 0, max(int(fx + fw + 6), min(src_w, x_edge))
    else:
        x_lo = max(4, int(sw * 0.56))
        x_hi = min(int(f_left * 1.08), int(sw * 0.86))
        if x_hi <= x_lo + 4:
            return None
        cseg = col_grad[x_lo:x_hi]
        if float(np.max(cseg)) < max(7.0, (float(np.median(col_grad[x_lo:])) + 1e-3) * 2.0):
            cam_x0, cam_x1 = max(0, int(src_w * 0.64)), src_w
        else:
            x_edge = int((int(np.argmax(cseg)) + x_lo) * inv_scale)
            cam_x0, cam_x1 = min(int(max(0.0, fx - 6)), max(0, x_edge)), src_w

    if (cam_x1 - cam_x0) < 64 or (cam_y1 - cam_y0) < 64:
        return None
    return cam_x0, cam_y0, cam_x1, cam_y1


def _is_fullscreen_multi_person_frame(
    boxes: list[tuple[float, float, float, float, float]],
    w: int,
    h: int,
) -> bool:
    """Detect wide two-shot framing where large full-screen speakers sit on opposite mid-screen halves."""
    big = [b for b in boxes if b[3] >= h * 0.18 and b[4] >= 0.60]
    if len(big) < 2:
        return False
    has_left = any((b[0] + b[2] / 2.0) < w * 0.35 and (h * 0.22) <= (b[1] + b[3] / 2.0) <= (h * 0.60) for b in big)
    has_right = any((b[0] + b[2] / 2.0) > w * 0.60 and (h * 0.22) <= (b[1] + b[3] / 2.0) <= (h * 0.60) for b in big)
    return has_left and has_right


def _build_active_intervals(hit_times: list[float], clip_dur: float, max_gap: float = 1.1) -> list[tuple[float, float]]:
    """Group face-present timestamps into continuous intervals so PiP hides cleanly during cutscenes."""
    if not hit_times:
        return []
    intervals: list[tuple[float, float]] = []
    s = max(0.0, hit_times[0] - 0.35)
    e = min(clip_dur, hit_times[0] + 0.55)
    for t in hit_times[1:]:
        if t - e <= max_gap:
            e = min(clip_dur, t + 0.55)
        else:
            intervals.append((round(s, 2), round(e, 2)))
            s = max(0.0, t - 0.35)
            e = min(clip_dur, t + 0.55)
    intervals.append((round(s, 2), round(e, 2)))
    if len(intervals) == 1 and intervals[0][0] <= 0.1 and intervals[0][1] >= clip_dur - 0.3:
        return [(0.0, round(clip_dur, 2))]
    return intervals


def _even(n: float, minimum: int = 16) -> int:
    val = max(minimum, int(round(n)))
    return val if val % 2 == 0 else val - 1


def track_corner_streamer_face(
    frame_faces: list[tuple[float, list[tuple[float, float, float, float, float]]]],
    src_w: int,
    src_h: int,
    requested_corner: CornerPosition,
    sampled: list[tuple[float, np.ndarray]] | None = None,
    inv_scale: float = 1.0,
) -> FaceCropTrack | None:
    """
    Locate and smoothly track the streamer's face inside the chosen corner (or AUTO-detect best corner).
    Clamps crop inside detected webcam borders so gameplay never leaks into the 9:16 PiP frame.
    """
    if not frame_faces or requested_corner == CornerPosition.OFF:
        return None

    is_auto = requested_corner == CornerPosition.AUTO
    if is_auto:
        multi_frames = sum(1 for _, bxs in frame_faces if _is_fullscreen_multi_person_frame(bxs, src_w, src_h))
        if multi_frames >= max(2, int(len(frame_faces) * 0.30)):
            return None

    corners_to_check = (
        [CornerPosition.TOP_LEFT, CornerPosition.TOP_RIGHT, CornerPosition.BOTTOM_LEFT, CornerPosition.BOTTOM_RIGHT]
        if is_auto
        else [requested_corner]
    )

    min_face_h = max(24.0, src_h * 0.075)
    best_corner = requested_corner
    best_hits: list[tuple[int, float, tuple[float, float, float, float, float]]] = []

    for corner in corners_to_check:
        hits: list[tuple[int, float, tuple[float, float, float, float, float]]] = []
        prev_cx: float | None = None
        for f_idx, (t_rel, boxes) in enumerate(frame_faces):
            roi_boxes = [
                b for b in boxes
                if _in_corner_roi(b[0] + b[2] / 2.0, b[1] + b[3] / 2.0, src_w, src_h, corner, strict_outer=True)
                and b[3] >= min_face_h
                and b[4] >= 0.50
                and (not is_auto or b[3] <= src_h * 0.30)
            ]
            if roi_boxes:
                def _rank(b: tuple[float, float, float, float, float]) -> float:
                    base = b[4] * math.sqrt(b[2] * b[3])
                    if prev_cx is None:
                        return base
                    dist = abs((b[0] + b[2] / 2.0) - prev_cx) / max(1.0, src_w)
                    return base * max(0.35, 1.0 - dist * 2.0)

                chosen = max(roi_boxes, key=_rank)
                prev_cx = chosen[0] + chosen[2] / 2.0
                hits.append((f_idx, t_rel, chosen))
        if len(hits) > len(best_hits):
            best_hits = hits
            best_corner = corner

    min_needed = max(2, int(len(frame_faces) * 0.06))
    if len(best_hits) < min_needed:
        return None

    # Median streamer face across detections
    med_fx = float(np.median([b[0] for _, _, b in best_hits]))
    med_fy = float(np.median([b[1] for _, _, b in best_hits]))
    med_fw = float(np.median([b[2] for _, _, b in best_hits]))
    med_fh = float(np.median([b[3] for _, _, b in best_hits]))
    med_box = (med_fx, med_fy, med_fw, med_fh, 0.9)

    bounds = (
        _detect_webcam_bounds(
            sampled, [h[0] for h in best_hits], inv_scale, src_w, src_h, best_corner, med_box
        )
        if sampled
        else None
    )

    if bounds is not None:
        cam_x0, cam_y0, cam_x1, cam_y1 = bounds
        cam_w = cam_x1 - cam_x0
        cam_h = cam_y1 - cam_y0
        # Keep full webcam height (no gameplay leak!) and enough width for face + headphones/shoulders
        ch = _even(min(cam_h, max(med_fh * 2.1, cam_h)), 64)
        ch = min(ch, cam_h if cam_h % 2 == 0 else cam_h - 1)
        ideal_cw = max(ch * 9.0 / 16.0, med_fw * 1.65)
        cw = _even(min(cam_w, ideal_cw), 48)
        cw = min(cw, cam_w if cam_w % 2 == 0 else cam_w - 1)
    else:
        cam_x0, cam_y0, cam_x1, cam_y1 = 0, 0, src_w, src_h
        desired_ch = max(src_h * 0.14, min(src_h * 0.40, med_fh * 2.2))
        k = max(2, int(round(desired_ch / 32.0)) * 2)
        while (16 * k > src_h or 9 * k > src_w) and k > 2:
            k -= 2
        cw, ch = 9 * k, 16 * k

    # Build trajectory strictly across hit frames, holding nearest detected face during brief blinks/cuts
    times_all = [t for t, _ in frame_faces]
    hit_map = {t: b for _, t, b in best_hits}
    raw_x: list[float] = []
    raw_y: list[float] = []
    last_box = best_hits[0][2]

    for t in times_all:
        b = hit_map.get(t, last_box)
        last_box = b
        face_cx = b[0] + b[2] / 2.0
        face_cy = b[1] + b[3] / 2.0
        crop_x = max(float(cam_x0), min(float(cam_x1 - cw), face_cx - cw / 2.0))
        crop_y = max(float(cam_y0), min(float(cam_y1 - ch), face_cy - ch * 0.44))
        raw_x.append(crop_x)
        raw_y.append(crop_y)

    smooth_x = _smooth_series(raw_x, alpha=0.25)
    smooth_y = _smooth_series(raw_y, alpha=0.25)
    keyframes = _downsample_keyframes(times_all, smooth_x, smooth_y, min_dt=1.0, max_points=32)

    clip_dur = max(0.5, times_all[-1] + 0.5) if times_all else 1.0
    active_intervals = _build_active_intervals([t for _, t, _ in best_hits], clip_dur)

    return FaceCropTrack(
        cw=cw,
        ch=ch,
        keyframes=keyframes,
        found_faces=True,
        detected_corner=best_corner,
        active_intervals=active_intervals,
    )


def track_main_speaker_9_16(
    frame_faces: list[tuple[float, list[tuple[float, float, float, float, float]]]],
    src_w: int,
    src_h: int,
    exclude_corner: CornerPosition = CornerPosition.OFF,
) -> FaceCropTrack | None:
    """Track primary speaker's horizontal X position so 9:16 main crop frames them instead of slicing faces."""
    if not frame_faces or (src_w / max(1, src_h)) <= (9.0 / 16.0):
        return None

    cw = int((src_h * 9.0 / 16.0) // 2) * 2
    ch = src_h if src_h % 2 == 0 else src_h - 1

    hits: list[tuple[float, float]] = []
    for t_rel, boxes in frame_faces:
        # Ignore small/corner gaming webcams so main 9:16 crop stays on the central action
        candidates = [
            b for b in boxes
            if b[3] >= src_h * 0.12
            and not (
                ((b[1] + b[3] / 2.0) >= src_h * 0.62 or (b[1] + b[3] / 2.0) <= src_h * 0.30)
                and ((b[0] + b[2] / 2.0) <= src_w * 0.36 or (b[0] + b[2] / 2.0) >= src_w * 0.64)
            )
        ]
        if exclude_corner not in (CornerPosition.OFF, CornerPosition.AUTO):
            non_corner = [
                b for b in candidates if not _in_corner_roi(b[0] + b[2] / 2.0, b[1] + b[3] / 2.0, src_w, src_h, exclude_corner, strict_outer=False)
            ]
            if non_corner:
                candidates = non_corner
        if candidates:
            main_face = max(candidates, key=lambda b: b[2] * b[3] * b[4])
            cx = main_face[0] + main_face[2] / 2.0
            crop_x = max(0.0, min(float(src_w - cw), cx - cw / 2.0))
            hits.append((t_rel, crop_x))

    if len(hits) < max(2, int(len(frame_faces) * 0.15)):
        return None

    hit_map = {t: x for t, x in hits}
    times_all = [t for t, _ in frame_faces]
    last_x = hits[0][1]
    raw_x: list[float] = []
    for t in times_all:
        if t in hit_map:
            last_x = hit_map[t]
        raw_x.append(last_x)

    smooth_x = _smooth_series(raw_x, alpha=0.25)
    zeros_y = [0.0] * len(times_all)
    keyframes = _downsample_keyframes(times_all, smooth_x, zeros_y, min_dt=1.0, max_points=32)
    return FaceCropTrack(cw=cw, ch=ch, keyframes=keyframes, found_faces=True)


def _downsample_keyframes(
    times: list[float],
    xs: list[float],
    ys: list[float],
    min_dt: float = 1.0,
    max_points: int = 32,
) -> list[CropKeyframe]:
    if not times:
        return []
    if len(times) <= 2:
        return [CropKeyframe(t=round(t, 2), x=round(x, 1), y=round(y, 1)) for t, x, y in zip(times, xs, ys)]

    step = max(1, int(math.ceil(len(times) / max_points)))
    kfs: list[CropKeyframe] = []
    last_t = -999.0
    for idx in range(0, len(times), step):
        t = times[idx]
        if t - last_t >= min_dt or idx == 0:
            kfs.append(CropKeyframe(t=round(t, 2), x=round(xs[idx], 1), y=round(ys[idx], 1)))
            last_t = t
    if kfs[-1].t < times[-1] - 0.1:
        kfs.append(CropKeyframe(t=round(times[-1], 2), x=round(xs[-1], 1), y=round(ys[-1], 1)))
    return kfs


def build_ffmpeg_lerp_expr(keyframes: list[CropKeyframe], axis: str = "x") -> str:
    """Build a smooth piecewise-linear FFmpeg crop expression for x(t) or y(t)."""
    if not keyframes:
        return "0"
    vals = [kf.x if axis == "x" else kf.y for kf in keyframes]
    if len(vals) == 1 or (max(vals) - min(vals) < 6.0):
        return str(int(round(float(np.median(vals)))))

    expr = f"{vals[-1]:.1f}"
    for i in range(len(keyframes) - 2, -1, -1):
        t0 = keyframes[i].t
        t1 = keyframes[i + 1].t
        v0 = vals[i]
        v1 = vals[i + 1]
        dt = max(0.05, t1 - t0)
        seg = f"({v0:.1f}+({v1 - v0:.1f})*(t-{t0:.2f})/{dt:.2f})"
        expr = f"if(lt(t,{t1:.2f}),{seg},{expr})"
    return expr


def analyze_clip_faces(
    video_path: Path,
    src_w: int,
    src_h: int,
    start: float,
    end: float,
    streamer_cam_source: CornerPosition,
    track_main: bool = True,
) -> tuple[FaceCropTrack | None, FaceCropTrack | None]:
    """
    Sample clip frames once and compute both:
      1. Streamer Corner Cam (PiP) face track (when streamer_cam_source != OFF)
      2. Main 9:16 speaker horizontal face track (when track_main is True)
    """
    need_corner = streamer_cam_source != CornerPosition.OFF
    if not need_corner and not track_main:
        return None, None

    sampled, inv_scale = sample_clip_frames(video_path, src_w, src_h, start, end, sample_fps=2.0, target_w=854)
    if not sampled:
        return None, None

    frame_faces = _detect_faces_per_frame(sampled, inv_scale)
    if not frame_faces:
        return None, None

    corner_track = (
        track_corner_streamer_face(
            frame_faces, src_w, src_h, streamer_cam_source, sampled=sampled, inv_scale=inv_scale
        )
        if need_corner
        else None
    )
    if corner_track and corner_track.found_faces:
        main_track = None
    elif track_main:
        main_track = track_main_speaker_9_16(frame_faces, src_w, src_h, exclude_corner=streamer_cam_source)
    else:
        main_track = None
    return corner_track, main_track
