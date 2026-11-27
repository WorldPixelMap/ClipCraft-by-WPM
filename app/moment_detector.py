"""100% Non-AI Best Moment Detector using YouTube Heatmap, Audio DSP, and Deterministic Transcript Heuristics."""

from __future__ import annotations

import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

from app.models import MomentCandidate, TranscriptSegment
from app.youtube_service import find_ffmpeg_dir

# Deterministic high-retention rhetorical, hook, and storytelling patterns (zero AI)
HOOK_PATTERNS = [
    re.compile(r"\b(here'?s why|the truth is|the secret|the problem is|most people|nobody talks|what happens)\b", re.I),
    re.compile(r"\b(mistake|never|always|actually|literally|biggest|best|worst|insane|crazy|unbelievable)\b", re.I),
    re.compile(r"\b(number one|top \d+|step \d+|first thing|rule number|three things|two things)\b", re.I),
    re.compile(r"\b(how to|why you|stop doing|don't ever|watch this|listen carefully|think about)\b", re.I),
]
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b|\$\d+")


@dataclass
class AudioAnalysis:
    rms_10hz: np.ndarray        # 10 samples/sec normalized RMS energy
    flux_10hz: np.ndarray       # 10 samples/sec onset/energy surge
    silence_mask_10hz: np.ndarray  # True where frame is near-silent/pause
    sample_rate_hz: float = 10.0


def _ffmpeg_exe() -> str:
    ffdir = find_ffmpeg_dir()
    if ffdir:
        cand = Path(ffdir) / "ffmpeg.exe"
        if cand.exists():
            return str(cand)
    return "ffmpeg"


def analyze_audio_signal(video_path: Path, total_duration: float) -> AudioAnalysis:
    """Decode audio via ffmpeg into 8 kHz mono PCM and extract 10 Hz energy + onset envelope (Zero AI)."""
    sample_rate = 8000
    hop_samples = sample_rate // 10  # 100ms frames -> 10 Hz
    cmd = [
        _ffmpeg_exe(),
        "-v", "error",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "s16le",
        "pipe:1",
    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    expected_frames = max(10, int(math.ceil(total_duration * 10.0)))

    if proc.returncode != 0 or len(proc.stdout) < hop_samples * 2:
        # Silent or audio-less fallback
        zeros = np.zeros(expected_frames, dtype=np.float32)
        return AudioAnalysis(rms_10hz=zeros, flux_10hz=zeros, silence_mask_10hz=np.ones(expected_frames, dtype=bool))

    pcm = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    n_frames = len(pcm) // hop_samples
    if n_frames < 5:
        zeros = np.zeros(expected_frames, dtype=np.float32)
        return AudioAnalysis(rms_10hz=zeros, flux_10hz=zeros, silence_mask_10hz=np.ones(expected_frames, dtype=bool))

    frames = pcm[: n_frames * hop_samples].reshape(n_frames, hop_samples)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-10)

    # Positive energy flux (sudden excitement / emphasis / speech onset)
    diff = np.diff(rms, prepend=rms[0])
    flux = np.maximum(diff, 0.0)

    # Robust normalization relative to 95th percentile (immune to single explosive spike)
    p95 = float(np.percentile(rms, 95)) or 1e-4
    rms_norm = np.clip(rms / max(p95, 1e-4), 0.0, 1.5)
    flux_p95 = float(np.percentile(flux, 95)) or 1e-4
    flux_norm = np.clip(flux / max(flux_p95, 1e-4), 0.0, 1.5)

    # Adaptive pause/silence detection
    active_vals = rms_norm[rms_norm > 0.03]
    noise_floor = float(np.percentile(active_vals, 15)) if len(active_vals) > 10 else 0.05
    silence_thresh = max(0.03, min(0.22, noise_floor * 0.75))
    silence_mask = rms_norm < silence_thresh

    return AudioAnalysis(
        rms_10hz=rms_norm.astype(np.float32),
        flux_10hz=flux_norm.astype(np.float32),
        silence_mask_10hz=silence_mask,
    )


def _build_heatmap_curve(heatmap: list[dict[str, float]], n_bins_1hz: int, duration: float) -> np.ndarray | None:
    """Resample YouTube Most-Replayed heatmap onto a 1 Hz grid."""
    if not heatmap or n_bins_1hz <= 0:
        return None
    curve = np.zeros(n_bins_1hz, dtype=np.float32)
    counts = np.zeros(n_bins_1hz, dtype=np.float32)
    for pt in heatmap:
        s = max(0, int(math.floor(pt["start_time"])))
        e = min(n_bins_1hz, int(math.ceil(min(duration, pt["end_time"]))))
        val = max(0.0, min(1.0, pt["value"]))
        if e <= s:
            e = min(n_bins_1hz, s + 1)
        curve[s:e] += val
        counts[s:e] += 1.0
    valid = counts > 0
    if not np.any(valid):
        return None
    curve[valid] /= counts[valid]
    return curve


def _compute_audio_1hz_scores(audio: AudioAnalysis, n_bins_1hz: int) -> tuple[np.ndarray, np.ndarray]:
    """Summarize 10 Hz audio into 1 Hz energy excess and speech-activity ratio."""
    rms = audio.rms_10hz
    flux = audio.flux_10hz
    n_10 = len(rms)

    # Local 30s baseline so quiet segments in loud videos or quiet podcasts scale fairly
    win_10hz = min(max(31, n_10 // 4), 301)
    if win_10hz % 2 == 0:
        win_10hz += 1
    baseline = median_filter(rms, size=win_10hz, mode="nearest")
    dynamic_excess = np.clip((rms - baseline * 0.85) / (baseline + 0.12), 0.0, 2.0)

    energy_1hz = np.zeros(n_bins_1hz, dtype=np.float32)
    active_1hz = np.zeros(n_bins_1hz, dtype=np.float32)
    for sec in range(n_bins_1hz):
        i0 = sec * 10
        i1 = min(n_10, (sec + 1) * 10)
        if i0 >= n_10:
            break
        seg_rms = rms[i0:i1]
        seg_exc = dynamic_excess[i0:i1]
        seg_flux = flux[i0:i1]
        seg_sil = audio.silence_mask_10hz[i0:i1]
        energy_1hz[sec] = float(0.45 * np.mean(seg_rms) + 0.35 * np.mean(seg_exc) + 0.20 * np.mean(seg_flux))
        active_1hz[sec] = float(1.0 - np.mean(seg_sil))

    # Normalize energy curve to [0, 1] via 98th percentile
    p98 = float(np.percentile(energy_1hz, 98)) if n_bins_1hz > 0 else 1.0
    if p98 > 1e-5:
        energy_1hz = np.clip(energy_1hz / p98, 0.0, 1.0)
    return energy_1hz, active_1hz


def _score_transcript_window(
    segments: list[TranscriptSegment],
    start: float,
    end: float,
) -> tuple[float, list[str], str]:
    """Score a time window deterministically using caption density, hooks, punctuation, and clean starts."""
    win_segs = [s for s in segments if s.end > start and s.start < end]
    if not win_segs:
        return 0.0, [], ""

    dur = max(1.0, end - start)
    # Count only words/text proportionally overlapping [start, end] so shorter sub-windows don't double-count
    effective_texts: list[str] = []
    effective_word_count = 0.0
    for s in win_segs:
        if s.has_word_timing and s.words:
            in_words = [w.word for w in s.words if w.end > start and w.start < end]
            if in_words:
                effective_texts.append(" ".join(in_words))
                effective_word_count += len(in_words)
        else:
            seg_len = max(0.2, s.end - s.start)
            ov = max(0.0, min(end, s.end) - max(start, s.start))
            frac = min(1.0, ov / seg_len)
            seg_words = re.findall(r"[A-Za-z0-9']+", s.text)
            if frac >= 0.45:
                effective_texts.append(s.text)
            effective_word_count += len(seg_words) * frac

    joined = " ".join(effective_texts)
    words = re.findall(r"[A-Za-z0-9']+", joined)
    word_count = int(round(effective_word_count))
    if word_count < 6 or not words:
        return 0.08, [], joined[:120]

    wps = effective_word_count / dur
    # Sweet spot for engaging spoken content: 2.0 to 4.2 words/sec
    if wps < 1.2:
        cadence_score = wps / 1.2 * 0.4
    elif wps <= 4.3:
        cadence_score = 0.65 + 0.35 * min(1.0, (wps - 1.2) / 1.8)
    else:
        cadence_score = max(0.5, 1.0 - (wps - 4.3) * 0.15)

    # Lexical diversity penalizes repetitive lyrics/filler
    unique_ratio = len({w.lower() for w in words}) / max(1, len(words))
    diversity_score = min(1.0, unique_ratio / 0.62)

    # Hook & rhetorical signals
    hook_hits = sum(len(pat.findall(joined)) for pat in HOOK_PATTERNS)
    q_exclaim = joined.count("?") + joined.count("!")
    num_hits = len(NUMBER_RE.findall(joined))

    # Bonus if first 8 seconds contain a hook/question AND window starts cleanly near a segment start
    first_seg = win_segs[0]
    clean_start_bonus = 0.15 if abs(first_seg.start - start) <= 1.5 else 0.0
    opening_text = " ".join(s.text for s in win_segs if s.start <= start + 8.0 and s.end >= start + 0.5)
    opening_hook = 1.0 if ("?" in opening_text or any(p.search(opening_text) for p in HOOK_PATTERNS)) else 0.0

    rhetorical_score = min(
        1.0,
        hook_hits * 0.22 + q_exclaim * 0.18 + num_hits * 0.12 + opening_hook * 0.28 + clean_start_bonus,
    )

    total = 0.38 * cadence_score + 0.22 * diversity_score + 0.40 * rhetorical_score
    tags: list[str] = []
    if opening_hook > 0 or hook_hits >= 2:
        tags.append("Strong Opening Hook")
    if q_exclaim >= 1:
        tags.append("High Engagement Cue")
    if 2.1 <= wps <= 4.3 and diversity_score >= 0.72:
        tags.append("Crisp Speech Cadence")

    preview = joined.strip()
    if len(preview) > 130:
        preview = preview[:127].rsplit(" ", 1)[0] + "..."
    return float(np.clip(total, 0.0, 1.0)), tags, preview


def _find_pause_points(audio: AudioAnalysis, segments: list[TranscriptSegment], duration: float) -> list[float]:
    """Collect natural cut points from transcript gaps and 10 Hz audio silence valleys."""
    points: set[float] = {0.0, round(duration, 2)}

    # 1. Caption segment gaps & sentence endings
    for idx, seg in enumerate(segments):
        if seg.text.rstrip().endswith((".", "?", "!")):
            points.add(round(seg.end, 2))
        if idx + 1 < len(segments):
            gap = segments[idx + 1].start - seg.end
            if gap >= 0.18:
                points.add(round((seg.end + segments[idx + 1].start) / 2.0, 2))
        else:
            points.add(round(seg.end, 2))

    # 2. Audio silence valleys (>= 250ms contiguous low energy)
    sil = audio.silence_mask_10hz
    i = 0
    n = len(sil)
    while i < n:
        if sil[i]:
            j = i
            while j < n and sil[j]:
                j += 1
            if j - i >= 2:
                mid = ((i + j) / 2.0) / audio.sample_rate_hz
                points.add(round(mid, 2))
            i = j
        else:
            i += 1

    return sorted(points)


def _snap_boundary(
    target: float,
    pause_points: list[float],
    min_bound: float,
    max_bound: float,
    max_shift: float = 3.5,
) -> float:
    """Snap target timestamp to the nearest natural pause inside [min_bound, max_bound]."""
    best = target
    best_dist = max_shift + 0.01
    for p in pause_points:
        if p < min_bound or p > max_bound:
            continue
        dist = abs(p - target)
        if dist < best_dist:
            best_dist = dist
            best = p
    return round(float(np.clip(best, min_bound, max_bound)), 2)


def detect_best_moments(
    video_path: Path,
    duration: float,
    num_clips: int,
    target_duration: int = 35,
    heatmap: list[dict[str, float]] | None = None,
    segments: list[TranscriptSegment] | None = None,
) -> tuple[list[MomentCandidate], AudioAnalysis]:
    """
    Pick top N non-overlapping highlight moments using strictly deterministic (Zero-AI) signals:
    YouTube Most-Replayed heatmap + audio DSP energy/flux + transcript hook heuristics + pause snapping.
    """
    segments = segments or []
    heatmap = heatmap or []

    audio = analyze_audio_signal(video_path, duration)
    n_bins = max(1, int(math.ceil(duration)))
    energy_1hz, active_1hz = _compute_audio_1hz_scores(audio, n_bins)
    heatmap_1hz = _build_heatmap_curve(heatmap, n_bins, duration)
    pause_points = _find_pause_points(audio, segments, duration)

    # Adapt clip duration cleanly for short videos so N non-overlapping clips remain possible
    max_feasible_dur = max(6.0, duration / max(1, num_clips) * 0.92)
    base_dur = float(min(target_duration, max_feasible_dur)) if duration < num_clips * (target_duration + 3) else float(target_duration)
    min_clip_len = max(5.0, min(14.0, base_dur * 0.65))
    max_clip_len = min(duration, max(min_clip_len + 2.0, base_dur * 1.25))

    if duration <= 8.0:
        # Ultra-short video fallback: return a single full clip
        return [
            MomentCandidate(
                rank=1,
                start=0.0,
                end=round(duration, 2),
                duration=round(duration, 2),
                score=90.0,
                audio_score=85.0,
                reasons=["Full Video Highlight"],
                title_snippet=" ".join(s.text for s in segments)[:120],
            )
        ], audio

    # Multi-scale candidate lengths around base_dur
    win_lengths = sorted({
        max(5, int(round(base_dur * 0.8))),
        max(6, int(round(base_dur))),
        min(int(duration), max(7, int(round(base_dur * 1.2)))),
    })

    step = 1 if duration < 150 else 2
    has_heatmap = heatmap_1hz is not None and float(np.max(heatmap_1hz)) > 0.05
    has_transcript = len(segments) >= 3

    # Smoothed audio energy envelope for hook boost
    smooth_energy = uniform_filter1d(energy_1hz, size=min(5, len(energy_1hz)))

    raw_candidates: list[dict] = []
    for w_len in win_lengths:
        if w_len > duration:
            continue
        for s_int in range(0, max(1, n_bins - w_len + 1), step):
            e_int = min(n_bins, s_int + w_len)
            if e_int - s_int < 5:
                continue

            # 1. Audio score (peak surges + mean energy + opening hook + speech coverage)
            win_e = energy_1hz[s_int:e_int]
            win_act = active_1hz[s_int:e_int]
            active_ratio = float(np.mean(win_act))
            if active_ratio < 0.20 and duration > 30:
                # Skip mostly dead silence
                continue

            peak_e = float(np.percentile(win_e, 88))
            mean_e = float(np.mean(win_e))
            dyn_std = float(np.std(win_e))
            open_e = float(np.mean(smooth_energy[s_int : min(e_int, s_int + 5)]))
            a_score = float(np.clip(0.35 * peak_e + 0.30 * mean_e + 0.20 * min(1.0, dyn_std * 3.0) + 0.15 * open_e, 0.0, 1.0))
            if active_ratio < 0.45:
                a_score *= 0.75

            # 2. Heatmap score (audience replay intensity)
            h_score = 0.0
            if has_heatmap and heatmap_1hz is not None:
                win_h = heatmap_1hz[s_int:e_int]
                h_score = float(np.clip(0.55 * np.max(win_h) + 0.45 * np.mean(win_h), 0.0, 1.0))

            # 3. Transcript heuristic score
            t_score, t_tags, preview = (0.0, [], "")
            if has_transcript:
                t_score, t_tags, preview = _score_transcript_window(segments, float(s_int), float(e_int))

            # Adaptive weighting across available non-AI signals
            if has_heatmap and has_transcript:
                combined = 0.42 * h_score + 0.32 * a_score + 0.26 * t_score
            elif has_heatmap:
                combined = 0.55 * h_score + 0.45 * a_score
            elif has_transcript:
                combined = 0.56 * a_score + 0.44 * t_score
            else:
                combined = a_score

            # Slight penalty for very first 2 seconds of video if it's an intro bumper unless heatmap/transcript loves it
            if s_int == 0 and duration > 120 and h_score < 0.5:
                combined *= 0.92

            raw_candidates.append(
                {
                    "start": float(s_int),
                    "end": float(e_int),
                    "combined": combined,
                    "heatmap_score": h_score,
                    "audio_score": a_score,
                    "transcript_score": t_score,
                    "t_tags": t_tags,
                    "preview": preview,
                    "open_e": open_e,
                    "peak_e": peak_e,
                }
            )

    if not raw_candidates:
        # Uniform split fallback for unusual videos
        seg_len = duration / max(1, num_clips)
        return [
            MomentCandidate(
                rank=i + 1,
                start=round(i * seg_len, 2),
                end=round(min(duration, (i + 1) * seg_len), 2),
                duration=round(min(duration, (i + 1) * seg_len) - i * seg_len, 2),
                score=70.0,
                reasons=["Balanced Timeline Split"],
            )
            for i in range(num_clips)
        ], audio

    # Sort descending by score and apply greedy NMS + pause snapping
    raw_candidates.sort(key=lambda c: c["combined"], reverse=True)
    selected: list[MomentCandidate] = []
    min_gap = 2.0 if duration > 90 else 0.5

    for cand in raw_candidates:
        if len(selected) >= num_clips:
            break

        # Snap to natural pause points without exceeding bounds or hitting existing clips
        s_snap = _snap_boundary(
            cand["start"],
            pause_points,
            min_bound=max(0.0, cand["start"] - 3.5),
            max_bound=min(cand["end"] - min_clip_len, cand["start"] + 3.5),
        )
        e_snap = _snap_boundary(
            cand["end"],
            pause_points,
            min_bound=max(s_snap + min_clip_len, cand["end"] - 3.5),
            max_bound=min(duration, s_snap + max_clip_len, cand["end"] + 3.5),
        )

        # Ensure non-overlapping with already chosen top moments
        overlaps = any(not (e_snap + min_gap <= ex.start or s_snap >= ex.end + min_gap) for ex in selected)
        if overlaps:
            continue

        reasons: list[str] = []
        if cand["heatmap_score"] >= 0.55:
            reasons.append("YouTube Most-Replayed Peak")
        if cand["peak_e"] >= 0.65:
            reasons.append("High Vocal Energy")
        elif cand["open_e"] >= 0.50:
            reasons.append("Strong Audio Opening")
        for tag in cand["t_tags"]:
            if tag not in reasons:
                reasons.append(tag)
        if not reasons:
            reasons.append("Top Dynamic Moment")

        selected.append(
            MomentCandidate(
                rank=len(selected) + 1,
                start=s_snap,
                end=e_snap,
                duration=round(e_snap - s_snap, 2),
                score=round(min(99.0, max(50.0, 52.0 + cand["combined"] * 47.0)), 1),
                heatmap_score=round(cand["heatmap_score"] * 100, 1),
                audio_score=round(cand["audio_score"] * 100, 1),
                transcript_score=round(cand["transcript_score"] * 100, 1),
                reasons=reasons[:3],
                title_snippet=cand["preview"],
            )
        )

    # If video was tight and strict min_gap prevented filling num_clips, relax overlap slightly
    if len(selected) < num_clips:
        for cand in raw_candidates:
            if len(selected) >= num_clips:
                break
            s_c, e_c = cand["start"], min(duration, cand["end"])
            overlap_amt = max(
                (max(0.0, min(e_c, ex.end) - max(s_c, ex.start)) for ex in selected),
                default=0.0,
            )
            if overlap_amt > (e_c - s_c) * 0.25:
                continue
            selected.append(
                MomentCandidate(
                    rank=len(selected) + 1,
                    start=round(s_c, 2),
                    end=round(e_c, 2),
                    duration=round(e_c - s_c, 2),
                    score=round(min(99.0, max(50.0, 50.0 + cand["combined"] * 45.0)), 1),
                    heatmap_score=round(cand["heatmap_score"] * 100, 1),
                    audio_score=round(cand["audio_score"] * 100, 1),
                    transcript_score=round(cand["transcript_score"] * 100, 1),
                    reasons=cand["t_tags"][:2] or ["Additional Highlight"],
                    title_snippet=cand["preview"],
                )
            )

    return selected, audio
