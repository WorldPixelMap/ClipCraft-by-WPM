"""Lightweight offline CPU real-human face detection, anywhere-on-screen webcam fitting, & smooth tracking."""

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


@dataclass
class _CandidateTrack:
    hits: list[tuple[int, float, tuple[float, ...]]]
    last_idx: int
    med_cx: float
    med_cy: float
    med_bw: float
    med_bh: float
    bounds: tuple[int, int, int, int] | None = None
    live_motion: float = 5.0
    skin_ratio: float = 0.5
    score: float = 0.0


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


def _create_detector(width: int, height: int, score_thresh: float = 0.42):
    model_path = ensure_yunet_model()
    if not model_path:
        return None
    with _DETECTOR_LOCK:
        return cv2.FaceDetectorYN.create(str(model_path), "", (width, height), score_thresh, 0.3, 30)


def sample_clip_frames(
    video_path: Path,
    src_w: int,
    src_h: int,
    start: float,
    end: float,
    sample_fps: float = 2.5,
    target_w: int = 960,
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


def _in_region_roi(
    cx: float,
    cy: float,
    w: int,
    h: int,
    region: CornerPosition,
    strict_outer: bool = True,
) -> bool:
    """Check whether (cx, cy) lies inside a requested manual source region (generous bounds)."""
    if region in (CornerPosition.OFF, CornerPosition.AUTO):
        return True
    nx = cx / max(1, w)
    ny = cy / max(1, h)
    xlim = 0.44 if strict_outer else 0.50
    ylim = 0.46 if strict_outer else 0.52

    if region == CornerPosition.TOP_LEFT:
        return nx <= xlim and ny <= ylim
    if region == CornerPosition.TOP_RIGHT:
        return nx >= (1.0 - xlim) and ny <= ylim
    if region == CornerPosition.BOTTOM_LEFT:
        return nx <= xlim and ny >= (1.0 - ylim)
    if region == CornerPosition.BOTTOM_RIGHT:
        return nx >= (1.0 - xlim) and ny >= (1.0 - ylim)
    if region == CornerPosition.TOP_CENTER:
        return 0.25 <= nx <= 0.75 and ny <= ylim
    if region == CornerPosition.BOTTOM_CENTER:
        return 0.25 <= nx <= 0.75 and ny >= (1.0 - ylim)
    if region == CornerPosition.MID_LEFT:
        return nx <= xlim and 0.22 <= ny <= 0.78
    if region == CornerPosition.MID_RIGHT:
        return nx >= (1.0 - xlim) and 0.22 <= ny <= 0.78
    if region == CornerPosition.CENTER:
        return 0.20 <= nx <= 0.80 and 0.20 <= ny <= 0.80
    return True


def _infer_region_from_point(cx: float, cy: float, w: int, h: int) -> CornerPosition:
    """Map median face center (cx, cy) to the nearest descriptive frame region."""
    nx = cx / max(1, w)
    ny = cy / max(1, h)
    in_mid_x = 0.36 <= nx <= 0.64
    in_mid_y = 0.38 <= ny <= 0.62

    if in_mid_x and in_mid_y:
        return CornerPosition.CENTER
    if in_mid_x:
        return CornerPosition.TOP_CENTER if ny < 0.5 else CornerPosition.BOTTOM_CENTER
    if in_mid_y:
        return CornerPosition.MID_LEFT if nx < 0.5 else CornerPosition.MID_RIGHT
    if ny < 0.5:
        return CornerPosition.TOP_LEFT if nx < 0.5 else CornerPosition.TOP_RIGHT
    return CornerPosition.BOTTOM_LEFT if nx < 0.5 else CornerPosition.BOTTOM_RIGHT


def _smooth_series(vals: list[float], alpha: float = 0.30) -> list[float]:
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


def _face_landmark_quality(box: tuple[float, ...]) -> float:
    """
    Validate upright human facial landmark geometry from YuNet 15-float detections.
    Returns 0.15..1.0 (returns 0.88 neutral for 5-tuple detections without landmarks).
    """
    if len(box) < 15:
        return 0.88
    fx, fy, fw, fh = box[0], box[1], max(1.0, box[2]), max(1.0, box[3])
    rx, ry = box[5], box[6]
    lx, ly = box[7], box[8]
    nx, ny = box[9], box[10]
    rmx, rmy = box[11], box[12]
    lmx, lmy = box[13], box[14]

    eye_dx = lx - rx
    eye_dist = math.hypot(eye_dx, ly - ry)
    if eye_dist < fw * 0.16 or eye_dist > fw * 0.88:
        return 0.20

    # Eyes should be roughly level and cleanly above mouth
    tilt = abs(ly - ry) / max(1.0, eye_dist)
    if tilt > 0.65:
        return 0.25

    eye_y = (ry + ly) * 0.5
    mouth_y = (rmy + lmy) * 0.5
    if mouth_y <= eye_y + fh * 0.10 or ny < eye_y - fh * 0.08 or ny > mouth_y + fh * 0.08:
        return 0.25

    # Landmarks should lie mostly inside/near the face bounding box
    if not (fx - fw * 0.15 <= rx <= fx + fw * 1.15 and fx - fw * 0.15 <= lx <= fx + fw * 1.15):
        return 0.30
    if not (fy - fh * 0.10 <= eye_y <= mouth_y <= fy + fh * 1.12):
        return 0.30

    # Reward upright frontal / clean 3-quarter poses
    eye_mid_x = (rx + lx) * 0.5
    nose_offset = abs(nx - eye_mid_x) / max(1.0, eye_dist)
    frontal_bonus = max(0.55, 1.0 - max(0.0, nose_offset - 0.15) * 0.55)
    tilt_bonus = max(0.65, 1.0 - tilt * 0.6)
    return min(1.0, frontal_bonus * tilt_bonus)


def _detect_faces_per_frame(
    sampled: list[tuple[float, np.ndarray]],
    inv_scale: float,
) -> list[tuple[float, list[tuple[float, ...]]]]:
    """Run YuNet globally on sampled frames, returning (t, [(x, y, w, h, score, ...landmarks), ...])."""
    if not sampled:
        return []
    sh, sw = sampled[0][1].shape[:2]
    detector = _create_detector(sw, sh)
    if detector is None:
        return []

    results: list[tuple[float, list[tuple[float, ...]]]] = []
    for t_rel, frame in sampled:
        _, raw_faces = detector.detect(frame)
        boxes: list[tuple[float, ...]] = []
        if raw_faces is not None:
            for f in raw_faces:
                fx = float(f[0]) * inv_scale
                fy = float(f[1]) * inv_scale
                fw = float(f[2]) * inv_scale
                fh = float(f[3]) * inv_scale
                score = float(f[14]) if len(f) > 14 else 0.7
                if fw < 16 or fh < 18:
                    continue
                ratio = fw / max(1.0, fh)
                if ratio < 0.42 or ratio > 1.55:
                    continue
                if len(f) >= 15:
                    det: tuple[float, ...] = (
                        fx,
                        fy,
                        fw,
                        fh,
                        score,
                        float(f[4]) * inv_scale,
                        float(f[5]) * inv_scale,
                        float(f[6]) * inv_scale,
                        float(f[7]) * inv_scale,
                        float(f[8]) * inv_scale,
                        float(f[9]) * inv_scale,
                        float(f[10]) * inv_scale,
                        float(f[11]) * inv_scale,
                        float(f[12]) * inv_scale,
                        float(f[13]) * inv_scale,
                    )
                else:
                    det = (fx, fy, fw, fh, score)
                if _face_landmark_quality(det) >= 0.35:
                    boxes.append(det)
        results.append((t_rel, boxes))
    return results


def _find_edge_consistent(
    grad_line_a: np.ndarray,
    grad_line_b: np.ndarray,
    lo: int,
    hi: int,
    med_baseline: float,
    min_strength: float = 10.0,
) -> int | None:
    """Detect a crisp or mildly feathered straight webcam border shared across both halves of a scan band."""
    if hi <= lo + 3:
        return None
    seg_a = grad_line_a[lo:hi]
    seg_b = grad_line_b[lo:hi]
    combined = 0.65 * np.minimum(seg_a, seg_b) + 0.35 * np.maximum(seg_a, seg_b)
    best_rel = int(np.argmax(combined))
    peak = float(combined[best_rel])
    # Require both halves to show meaningful edge energy so diagonal game geometry doesn't fake a border
    weakest_half = float(min(seg_a[best_rel], seg_b[best_rel]))
    if peak < max(min_strength, med_baseline * 2.2) or weakest_half < min_strength * 0.45:
        return None
    return lo + best_rel + 1


def _detect_facecam_box_anywhere(
    sampled: list[tuple[float, np.ndarray]],
    hit_indices: list[int],
    inv_scale: float,
    src_w: int,
    src_h: int,
    face_box: tuple[float, ...],
) -> tuple[int, int, int, int] | None:
    """
    Detect an overlaid OBS/stream webcam rectangle (x0, y0, x1, y1) at ANY position on the frame
    (corners, mid-left/right, top/bottom-center, or floating inset) so cropped video fits the webcam
    cleanly with zero gameplay leak. Returns None if face is borderless / green-screen / full-frame.
    """
    if not sampled or not hit_indices:
        return None

    pick = hit_indices[: min(8, len(hit_indices))]
    grays = [cv2.cvtColor(sampled[i][1], cv2.COLOR_BGR2GRAY).astype(np.float32) for i in pick]
    mean_gray = np.mean(grays, axis=0)
    sh, sw = mean_gray.shape
    if sh < 24 or sw < 24:
        return None

    fx, fy, fw, fh = face_box[0], face_box[1], face_box[2], face_box[3]
    f_top = fy / inv_scale
    f_bot = (fy + fh) / inv_scale
    f_left = fx / inv_scale
    f_right = (fx + fw) / inv_scale
    sfw = max(8.0, fw / inv_scale)
    sfh = max(8.0, fh / inv_scale)

    # Search bands anchored strictly around the candidate face
    bx0 = max(0, int(round(f_left - sfw * 0.65)))
    bx1 = min(sw, int(round(f_right + sfw * 0.65)))
    bmid = (bx0 + bx1) // 2
    if bmid <= bx0 + 3 or bx1 <= bmid + 3:
        return None

    # 2-px central difference detects both crisp 1px lines and 2px feathered OBS borders
    row_grad_l = np.mean(np.abs(mean_gray[2:, bx0:bmid] - mean_gray[:-2, bx0:bmid]), axis=1)
    row_grad_r = np.mean(np.abs(mean_gray[2:, bmid:bx1] - mean_gray[:-2, bmid:bx1]), axis=1)
    row_med = float(np.median(np.minimum(row_grad_l, row_grad_r))) + 1e-3

    max_cam_sh = int(sh * 0.56)
    up_lo = max(1, int(f_bot - max_cam_sh))
    up_hi = max(up_lo + 4, int(f_top - sfh * 0.04))
    top_edge = _find_edge_consistent(row_grad_l, row_grad_r, up_lo, min(len(row_grad_l) - 1, up_hi), row_med)

    dn_lo = min(len(row_grad_l) - 4, int(f_bot + sfh * 0.04))
    dn_hi = min(len(row_grad_l) - 1, int(f_top + max_cam_sh))
    bot_edge = _find_edge_consistent(row_grad_l, row_grad_r, dn_lo, dn_hi, row_med)

    inner_edges = 0
    cam_y0: int | None = None
    cam_y1: int | None = None
    if top_edge is not None:
        cam_y0 = max(0, min(int(max(0.0, fy - 4)), int(round(top_edge * inv_scale))))
        inner_edges += 1
    elif f_top <= sh * 0.22:
        cam_y0 = 0

    if bot_edge is not None:
        cam_y1 = min(src_h, max(int(fy + fh + 4), int(round(bot_edge * inv_scale))))
        inner_edges += 1
    elif f_bot >= sh * 0.78:
        cam_y1 = src_h

    if cam_y0 is None or cam_y1 is None:
        return None
    if (cam_y1 - cam_y0) > src_h * 0.62 or (cam_y1 - cam_y0) < fh * 1.10:
        return None

    # Horizontal boundaries inside detected vertical webcam span
    sy0 = max(0, int(round(cam_y0 / inv_scale)) + 2)
    sy1 = min(sh, int(round(cam_y1 / inv_scale)) - 2)
    ymid = (sy0 + sy1) // 2
    if ymid <= sy0 + 3 or sy1 <= ymid + 3:
        return None

    col_grad_t = np.mean(np.abs(mean_gray[sy0:ymid, 2:] - mean_gray[sy0:ymid, :-2]), axis=0)
    col_grad_b = np.mean(np.abs(mean_gray[ymid:sy1, 2:] - mean_gray[ymid:sy1, :-2]), axis=0)
    col_med = float(np.median(np.minimum(col_grad_t, col_grad_b))) + 1e-3

    max_cam_sw = int(sw * 0.52)
    lf_lo = max(1, int(f_right - max_cam_sw))
    lf_hi = max(lf_lo + 4, int(f_left - sfw * 0.04))
    left_edge = _find_edge_consistent(col_grad_t, col_grad_b, lf_lo, min(len(col_grad_t) - 1, lf_hi), col_med, min_strength=9.0)

    rt_lo = min(len(col_grad_t) - 4, int(f_right + sfw * 0.04))
    rt_hi = min(len(col_grad_t) - 1, int(f_left + max_cam_sw))
    right_edge = _find_edge_consistent(col_grad_t, col_grad_b, rt_lo, rt_hi, col_med, min_strength=9.0)

    cam_x0: int | None = None
    cam_x1: int | None = None
    if left_edge is not None:
        cam_x0 = max(0, min(int(max(0.0, fx - 4)), int(round(left_edge * inv_scale))))
        inner_edges += 1
    elif f_left <= sw * 0.22:
        cam_x0 = 0

    if right_edge is not None:
        cam_x1 = min(src_w, max(int(fx + fw + 4), int(round(right_edge * inv_scale))))
        inner_edges += 1
    elif f_right >= sw * 0.78:
        cam_x1 = src_w

    if cam_x0 is None or cam_x1 is None or inner_edges < 2:
        return None

    cam_w = cam_x1 - cam_x0
    cam_h = cam_y1 - cam_y0
    if cam_w < max(56.0, fw * 1.10) or cam_h < max(56.0, fh * 1.10):
        return None
    if cam_w > src_w * 0.58 or cam_h > src_h * 0.58:
        return None
    return cam_x0, cam_y0, cam_x1, cam_y1


def _track_patch_metrics(
    sampled: list[tuple[float, np.ndarray]] | None,
    hits: list[tuple[int, float, tuple[float, ...]]],
    inv_scale: float,
    src_w: int,
    src_h: int,
) -> tuple[float, float]:
    """
    Evaluate real-camera human indicators inside the stabilized face region:
      - live_motion: pixel change across hit frames (distinguishes talking human from frozen photo/PNG avatar)
      - skin_ratio: natural human skin fraction in center cheek/nose patch (gentle tie-breaker bonus)
    """
    if not sampled or len(hits) < 2:
        return 5.0, 0.5

    med_fx = float(np.median([b[0] for _, _, b in hits])) / inv_scale
    med_fy = float(np.median([b[1] for _, _, b in hits])) / inv_scale
    med_fw = float(np.median([b[2] for _, _, b in hits])) / inv_scale
    med_fh = float(np.median([b[3] for _, _, b in hits])) / inv_scale

    sh, sw = sampled[0][1].shape[:2]
    x0 = max(0, int(round(med_fx + med_fw * 0.15)))
    x1 = min(sw, int(round(med_fx + med_fw * 0.85)))
    y0 = max(0, int(round(med_fy + med_fh * 0.20)))
    y1 = min(sh, int(round(med_fy + med_fh * 0.88)))
    if x1 <= x0 + 6 or y1 <= y0 + 6:
        return 5.0, 0.5

    step = max(1, len(hits) // 6)
    sample_hits = hits[::step][:6]
    patches_gray = [cv2.cvtColor(sampled[i][1][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32) for i, _, _ in sample_hits]
    diffs = [float(np.mean(np.abs(patches_gray[k] - patches_gray[k - 1]))) for k in range(1, len(patches_gray))]
    live_motion = float(np.median(diffs)) if diffs else 5.0

    # Generous YCrCb skin-chroma check on center cheek/nose subpatch
    skin_vals: list[float] = []
    for i, _, b in sample_hits[:4]:
        bx = b[0] / inv_scale
        by = b[1] / inv_scale
        bw = b[2] / inv_scale
        bh = b[3] / inv_scale
        sx0 = max(0, int(round(bx + bw * 0.22)))
        sx1 = min(sw, int(round(bx + bw * 0.78)))
        sy0 = max(0, int(round(by + bh * 0.28)))
        sy1 = min(sh, int(round(by + bh * 0.78)))
        if sx1 > sx0 + 4 and sy1 > sy0 + 4:
            ycrcb = cv2.cvtColor(sampled[i][1][sy0:sy1, sx0:sx1], cv2.COLOR_BGR2YCrCb)
            cr = ycrcb[:, :, 1]
            cb = ycrcb[:, :, 2]
            mask = (cr >= 126) & (cr <= 182) & (cb >= 72) & (cb <= 136)
            skin_vals.append(float(np.mean(mask)))
    skin_ratio = float(np.mean(skin_vals)) if skin_vals else 0.5
    return live_motion, skin_ratio


def _cluster_face_tracks(
    frame_faces: list[tuple[float, list[tuple[float, ...]]]],
    src_w: int,
    src_h: int,
    region_filter: CornerPosition,
) -> list[_CandidateTrack]:
    """Group face detections across sampled frames into spatially coherent face tracks anywhere on screen."""
    min_face_h = max(20.0, src_h * 0.042)
    tracks: list[_CandidateTrack] = []

    for f_idx, (t_rel, boxes) in enumerate(frame_faces):
        valid = [
            b for b in boxes
            if b[3] >= min_face_h
            and b[4] >= 0.45
            and _in_region_roi(b[0] + b[2] / 2.0, b[1] + b[3] / 2.0, src_w, src_h, region_filter, strict_outer=True)
        ]
        # Prefer best confidence boxes first when matching
        valid.sort(key=lambda b: b[4] * math.sqrt(b[2] * b[3]), reverse=True)
        used_tracks: set[int] = set()

        for b in valid:
            cx = b[0] + b[2] / 2.0
            cy = b[1] + b[3] / 2.0
            bw, bh = b[2], b[3]

            best_ti = -1
            best_dist = 1e12
            for ti, tr in enumerate(tracks):
                if ti in used_tracks:
                    continue
                if (f_idx - tr.last_idx) > 6:
                    continue
                if bh < tr.med_bh * 0.52 or bh > tr.med_bh * 1.90:
                    continue
                ref_span = max(bw, tr.med_bw, bh, tr.med_bh)
                dist = math.hypot(cx - tr.med_cx, cy - tr.med_cy)
                if dist <= ref_span * 1.85 and dist < best_dist:
                    best_dist = dist
                    best_ti = ti

            if best_ti >= 0:
                tr = tracks[best_ti]
                tr.hits.append((f_idx, t_rel, b))
                tr.last_idx = f_idx
                recent = tr.hits[-5:]
                tr.med_cx = float(np.median([h[2][0] + h[2][2] / 2.0 for h in recent]))
                tr.med_cy = float(np.median([h[2][1] + h[2][3] / 2.0 for h in recent]))
                tr.med_bw = float(np.median([h[2][2] for h in recent]))
                tr.med_bh = float(np.median([h[2][3] for h in recent]))
                used_tracks.add(best_ti)
            else:
                tracks.append(
                    _CandidateTrack(
                        hits=[(f_idx, t_rel, b)],
                        last_idx=f_idx,
                        med_cx=cx,
                        med_cy=cy,
                        med_bw=bw,
                        med_bh=bh,
                    )
                )
    return tracks


def _score_candidate_tracks(
    tracks: list[_CandidateTrack],
    n_frames: int,
    src_w: int,
    src_h: int,
    sampled: list[tuple[float, np.ndarray]] | None,
    inv_scale: float,
) -> list[_CandidateTrack]:
    """Score candidate face tracks to separate real human streamer faces from game NPCs & static posters."""
    min_needed = max(2, int(n_frames * 0.06))
    scored: list[_CandidateTrack] = []

    for tr in tracks:
        if len(tr.hits) < min_needed:
            continue

        presence = len(tr.hits) / max(1, n_frames)
        conf_mean = float(np.mean([h[2][4] for h in tr.hits]))
        lm_mean = float(np.mean([_face_landmark_quality(h[2]) for h in tr.hits]))

        med_fx = float(np.median([h[2][0] for h in tr.hits]))
        med_fy = float(np.median([h[2][1] for h in tr.hits]))
        med_fw = float(np.median([h[2][2] for h in tr.hits]))
        med_fh = float(np.median([h[2][3] for h in tr.hits]))
        med_box = (med_fx, med_fy, med_fw, med_fh, conf_mean)

        # Positional stability across detections
        cxs = [h[2][0] + h[2][2] / 2.0 for h in tr.hits]
        cys = [h[2][1] + h[2][3] / 2.0 for h in tr.hits]
        if len(cxs) >= 2:
            steps = [math.hypot(cxs[i] - cxs[i - 1], cys[i] - cys[i - 1]) for i in range(1, len(cxs))]
            med_step = float(np.median(steps)) / max(1.0, med_fw)
            spread = (float(np.std(cxs)) + float(np.std(cys))) / max(1.0, med_fw + med_fh)
        else:
            med_step = 0.0
            spread = 0.0
        stability_mult = max(0.55, min(1.45, 1.38 - med_step * 0.85 - spread * 0.35))

        bounds = (
            _detect_facecam_box_anywhere(sampled, [h[0] for h in tr.hits], inv_scale, src_w, src_h, med_box)
            if sampled
            else None
        )
        live_motion, skin_ratio = _track_patch_metrics(sampled, tr.hits, inv_scale, src_w, src_h)

        # Strongly reject frozen PNG avatars / printed wall posters when enough frames exist
        if sampled and len(tr.hits) >= 4 and live_motion < 1.1:
            motion_mult = 0.15
        else:
            motion_mult = 1.0 + min(0.20, max(0.0, (live_motion - 1.5) * 0.03))

        # Overlay border evidence is the strongest real-streamer signal over mid-screen game NPCs
        overlay_mult = 1.65 if bounds is not None else 1.0
        skin_mult = 0.92 + 0.20 * skin_ratio

        # Mild streamer-zone tie-breaker (outer edges/corners slightly favored ONLY on near-ties, never exclusive)
        nx = (med_fx + med_fw / 2.0) / max(1, src_w)
        ny = (med_fy + med_fh / 2.0) / max(1, src_h)
        in_center_crosshair = (0.38 <= nx <= 0.62) and (0.34 <= ny <= 0.66)
        zone_mult = 0.88 if in_center_crosshair else 1.08

        size_rel = min(1.18, max(0.85, 0.85 + (med_fh / max(1, src_h)) * 1.4))

        total_score = (
            (presence ** 0.72)
            * conf_mean
            * lm_mean
            * stability_mult
            * overlay_mult
            * motion_mult
            * skin_mult
            * zone_mult
            * size_rel
        )
        tr.bounds = bounds
        tr.live_motion = live_motion
        tr.skin_ratio = skin_ratio
        tr.score = float(total_score)
        scored.append(tr)

    scored.sort(key=lambda t: t.score, reverse=True)
    return scored


def _build_active_intervals(hit_times: list[float], clip_dur: float, max_gap: float = 1.2) -> list[tuple[float, float]]:
    """Group face-present timestamps into continuous intervals so PiP hides cleanly during cutscenes."""
    if not hit_times:
        return []
    ordered = sorted(set(hit_times))
    intervals: list[tuple[float, float]] = []
    s = max(0.0, ordered[0] - 0.35)
    e = min(clip_dur, ordered[0] + 0.55)
    for t in ordered[1:]:
        if t - e <= max_gap:
            e = min(clip_dur, t + 0.55)
        else:
            intervals.append((round(s, 2), round(e, 2)))
            s = max(0.0, t - 0.35)
            e = min(clip_dur, t + 0.55)
    intervals.append((round(s, 2), round(e, 2)))
    if len(intervals) == 1 and intervals[0][0] <= 0.15 and intervals[0][1] >= clip_dur - 0.35:
        return [(0.0, round(clip_dur, 2))]
    return intervals


def _even(n: float, minimum: int = 16) -> int:
    val = max(minimum, int(round(n)))
    return val if val % 2 == 0 else val - 1


def _compute_fitted_crop_size(
    med_fx: float,
    med_fy: float,
    med_fw: float,
    med_fh: float,
    src_w: int,
    src_h: int,
    bounds: tuple[int, int, int, int] | None,
) -> tuple[int, int, tuple[int, int, int, int]]:
    """
    Compute tightly face-fitted crop (cw, ch) and clamp container (x0, y0, x1, y1) so the
    real human face + hair/headphones/collar fills the Cam Frame cleanly with zero dead filler bars.
    """
    if bounds is not None:
        cam_x0, cam_y0, cam_x1, cam_y1 = bounds
        cam_w = cam_x1 - cam_x0
        cam_h = cam_y1 - cam_y0
        need_cw = med_fw * 2.05
        need_ch = med_fh * 2.15
        # Keep tight face framing while obeying webcam box edges (never leak gameplay)
        if cam_h <= med_fh * 2.5:
            ch = _even(cam_h, 48)
        else:
            ch = _even(min(cam_h, need_ch), 48)
        ch = min(ch, cam_h if cam_h % 2 == 0 else max(48, cam_h - 1))

        ideal_cw = max(need_cw, ch * 0.78)
        max_cw = max(need_cw, ch * 1.32)
        cw = _even(min(cam_w, min(ideal_cw, max_cw)), 48)
        cw = min(cw, cam_w if cam_w % 2 == 0 else max(48, cam_w - 1))
        return cw, ch, (cam_x0, cam_y0, cam_x1, cam_y1)

    # Borderless / green-screen / full-frame human anywhere in the video:
    # Sleek ~3:4 portrait card centered comfortably around face + hair + collar
    need_cw = max(64.0, med_fw * 2.10)
    need_ch = max(84.0, med_fh * 2.25, need_cw / 0.76)

    # If face sits near the very bottom/top wall with no headroom/footroom on one side, avoid grabbing dead wall
    footroom = src_h - (med_fy + med_fh)
    headroom = med_fy
    if footroom < med_fh * 0.25 or headroom < med_fh * 0.20:
        need_ch = min(need_ch, max(med_fh * 1.90, need_cw / 0.92))

    ch = _even(min(src_h, need_ch), 64)
    cw = _even(min(src_w, max(need_cw, ch * 0.58), ch * 1.25), 48)
    # Clamp minimum portrait ratio so extreme narrow slices don't cut ears
    if cw / max(1, ch) < 0.5625:
        ch = _even(min(src_h, cw / 0.5625), 64)
    return cw, ch, (0, 0, src_w, src_h)


def track_corner_streamer_face(
    frame_faces: list[tuple[float, list[tuple[float, ...]]]],
    src_w: int,
    src_h: int,
    requested_corner: CornerPosition,
    sampled: list[tuple[float, np.ndarray]] | None = None,
    inv_scale: float = 1.0,
) -> FaceCropTrack | None:
    """
    Locate the real human streamer face anywhere in the source video (corners, edges, center,
    top/bottom-center, or moving across scene cuts) and compute a smoothly face-fitted Cam Frame track.
    """
    if not frame_faces or requested_corner == CornerPosition.OFF:
        return None

    tracks = _cluster_face_tracks(frame_faces, src_w, src_h, requested_corner)
    ranked = _score_candidate_tracks(tracks, len(frame_faces), src_w, src_h, sampled, inv_scale)
    if not ranked:
        return None

    primary = ranked[0]
    med_fx = float(np.median([h[2][0] for h in primary.hits]))
    med_fy = float(np.median([h[2][1] for h in primary.hits]))
    med_fw = float(np.median([h[2][2] for h in primary.hits]))
    med_fh = float(np.median([h[2][3] for h in primary.hits]))

    cw, ch, primary_bounds = _compute_fitted_crop_size(
        med_fx, med_fy, med_fw, med_fh, src_w, src_h, primary.bounds
    )

    # Assemble hit timeline: include compatible non-overlapping human streamer segments if cam moved mid-clip
    timeline_hits: dict[int, tuple[tuple[float, ...], tuple[int, int, int, int]]] = {
        f_idx: (box, primary_bounds) for f_idx, _, box in primary.hits
    }
    if requested_corner == CornerPosition.AUTO:
        for sec in ranked[1:]:
            if len(sec.hits) < 3 or sec.score < primary.score * 0.40:
                continue
            sec_fh = float(np.median([h[2][3] for h in sec.hits]))
            # Allow seamless follow across scene cuts ONLY when scale fits cw/ch and isn't an NPC crowd
            if sec.bounds is None and not (0.60 <= sec_fh / max(1.0, med_fh) <= 1.50):
                continue
            sec_bounds = sec.bounds if sec.bounds is not None else (0, 0, src_w, src_h)
            if (sec_bounds[2] - sec_bounds[0]) < cw or (sec_bounds[3] - sec_bounds[1]) < ch:
                continue
            for f_idx, _, box in sec.hits:
                if f_idx not in timeline_hits:
                    timeline_hits[f_idx] = (box, sec_bounds)

    times_all = [t for t, _ in frame_faces]
    ordered_indices = sorted(timeline_hits.keys())
    if not ordered_indices:
        return None

    raw_x: list[float] = []
    raw_y: list[float] = []
    last_box, last_bounds = timeline_hits[ordered_indices[0]]
    hit_times: list[float] = []

    for f_idx, t_rel in enumerate(times_all):
        if f_idx in timeline_hits:
            last_box, last_bounds = timeline_hits[f_idx]
            hit_times.append(t_rel)
        cx0, cy0, cx1, cy1 = last_bounds
        face_cx = last_box[0] + last_box[2] / 2.0
        face_cy = last_box[1] + last_box[3] / 2.0
        crop_x = max(float(cx0), min(float(cx1 - cw), face_cx - cw / 2.0))
        crop_y = max(float(cy0), min(float(cy1 - ch), face_cy - ch * 0.45))
        raw_x.append(crop_x)
        raw_y.append(crop_y)

    smooth_x = _smooth_series(raw_x, alpha=0.38)
    smooth_y = _smooth_series(raw_y, alpha=0.38)
    keyframes = _downsample_keyframes(times_all, smooth_x, smooth_y, min_dt=0.5, max_points=48)

    detected_region = _infer_region_from_point(med_fx + med_fw / 2.0, med_fy + med_fh / 2.0, src_w, src_h)
    clip_dur = max(0.5, times_all[-1] + 0.4) if times_all else 1.0
    active_intervals = _build_active_intervals(hit_times, clip_dur)

    return FaceCropTrack(
        cw=cw,
        ch=ch,
        keyframes=keyframes,
        found_faces=True,
        detected_corner=detected_region,
        active_intervals=active_intervals,
    )


def track_main_speaker_9_16(
    frame_faces: list[tuple[float, list[tuple[float, ...]]]],
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
        candidates = [
            b for b in boxes
            if b[3] >= src_h * 0.11
            and not (
                ((b[1] + b[3] / 2.0) >= src_h * 0.64 or (b[1] + b[3] / 2.0) <= src_h * 0.28)
                and ((b[0] + b[2] / 2.0) <= src_w * 0.34 or (b[0] + b[2] / 2.0) >= src_w * 0.66)
            )
        ]
        if exclude_corner not in (CornerPosition.OFF, CornerPosition.AUTO):
            non_corner = [
                b for b in candidates
                if not _in_region_roi(b[0] + b[2] / 2.0, b[1] + b[3] / 2.0, src_w, src_h, exclude_corner, strict_outer=False)
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
    min_dt: float = 0.8,
    max_points: int = 36,
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


def build_ffmpeg_lerp_expr(keyframes: list[CropKeyframe], axis: str = "x", jump_thresh: float = 0.0) -> str:
    """Build a smooth piecewise-linear FFmpeg crop expression for x(t) or y(t), snapping cleanly on jump cuts."""
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
        if jump_thresh > 0.0 and abs(v1 - v0) >= jump_thresh:
            t_mid = (t0 + t1) * 0.5
            expr = f"if(lt(t,{t_mid:.2f}),{v0:.1f},{expr})"
        else:
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
      1. Real human streamer Cam Frame track (anywhere on source video when streamer_cam_source != OFF)
      2. Main 9:16 speaker horizontal face track (when track_main is True)
    """
    need_corner = streamer_cam_source != CornerPosition.OFF
    if not need_corner and not track_main:
        return None, None

    sampled, inv_scale = sample_clip_frames(video_path, src_w, src_h, start, end, sample_fps=2.5, target_w=960)
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
