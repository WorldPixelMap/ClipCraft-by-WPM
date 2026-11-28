"""100% Non-AI Best Moment Detector using YouTube Heatmap, Audio DSP, Climax Arc, and Bilingual Transcript Heuristics."""

from __future__ import annotations

import bisect
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

from app.models import MomentCandidate, TranscriptSegment
from app.youtube_service import find_ffmpeg_dir

# Tier-1 strong curiosity, storytelling, and opening-hook cues (English + Hindi + Hinglish)
STRONG_HOOK_PATTERNS = [
    re.compile(
        r"\b(here'?s why|the truth is|the secret|the problem is|the reason|most people|nobody talks|"
        r"what happens|you won'?t believe|turns out|wait until|watch this|listen carefully|"
        r"stop doing|don'?t ever|biggest mistake|how to|why you|did you know)\b",
        re.I,
    ),
    re.compile(r"\b(number one|top \d+|step \d+|first thing|rule number|three things|two things)\b", re.I),
    re.compile(
        r"\b(sach batau|asli baat|asli sach|sabse badi|yeh galti|mat karna|kaise|kyon|kyun|"
        r"dekhna|dhyan se|sunno|suno|kabhi mat|isliye|raaz)\b",
        re.I,
    ),
    re.compile(r"(सबसे\s*बड़ी|असली\s*बात|सच\s*बताऊँ|ये\s*गलती|मत\s*करना|कैसे|क्यों|देखो|सुनो|ध्यान\s*से)"),
]

# Tier-2 intensity, contrast, and payoff markers
INTENSITY_PATTERNS = [
    re.compile(
        r"\b(mistake|never|always|actually|literally|biggest|best|worst|insane|crazy|unbelievable|"
        r"instead|however|suddenly|finally|realized|proved|wrong|secret|truth|shocking|broken|"
        r"galti|lekin|magar|bilkul|asli|sach|khatam|fayda|nuksaan)\b",
        re.I,
    ),
    re.compile(r"(गलती|लेकिन|मगर|बिल्कुल|असली|सच|ख़त्म|खत्म|फायदा|नुकसान|ज़्यादा)"),
]

REACTION_TAG_RE = re.compile(r"\[(?:laughter|applause|cheering|cheers|crowd|laughing)\]", re.I)
REACTION_WORD_RE = re.compile(r"\b(haha+|lmao|omg|wtf|whoa|woah|no way|oh my god|crazy|insane)\b", re.I)
NON_SPEECH_TAG_RE = re.compile(r"\[[^\]]*(?:music|♪|silence|noise|inaudible|instrumental)[^\]]*\]", re.I)
ANY_BRACKET_TAG_RE = re.compile(r"\[[^\]]+\]")
MUSIC_SYMBOL_RE = re.compile(r"[♪♫]+")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b|\$\d+")
WORD_TOKEN_RE = re.compile(r"[A-Za-z0-9'\u0900-\u097F]+")

CONTINUATION_STARTERS = {
    "and", "but", "or", "so", "because", "which", "that", "then", "also",
    "though", "however", "plus", "aur", "lekin", "magar", "par", "kyunki", "toh",
}
DANGLING_ENDINGS = {
    "the", "a", "an", "and", "but", "or", "to", "of", "in", "on", "at", "for",
    "with", "from", "by", "is", "are", "was", "were", "that", "which", "who",
    "if", "when", "because", "as", "into", "ka", "ki", "ke", "se", "mein", "ko", "aur",
}


@dataclass
class AudioAnalysis:
    rms_10hz: np.ndarray            # 10 samples/sec normalized RMS energy
    flux_10hz: np.ndarray           # 10 samples/sec onset/energy surge
    silence_mask_10hz: np.ndarray   # True where frame is near-silent/pause
    sample_rate_hz: float = 10.0
    speech_band_10hz: np.ndarray | None = None  # 10 Hz ratio of 300-3400Hz speech-band energy
    modulation_10hz: np.ndarray | None = None   # 10 Hz syllable/laughter envelope modulation


@dataclass
class _TranscriptGrid:
    spoken_words_1hz: np.ndarray
    t1_hooks_1hz: np.ndarray
    t2_hooks_1hz: np.ndarray
    q_exclaim_1hz: np.ndarray
    nums_1hz: np.ndarray
    reactions_1hz: np.ndarray
    lyric_1hz: np.ndarray
    sent_start_1hz: np.ndarray
    sent_end_1hz: np.ndarray
    next_speech_sec: np.ndarray
    spoken_segments: list[TranscriptSegment]
    seg_starts: list[float]


def _ffmpeg_exe() -> str:
    ffdir = find_ffmpeg_dir()
    if ffdir:
        cand = Path(ffdir) / "ffmpeg.exe"
        if cand.exists():
            return str(cand)
    return "ffmpeg"


def _clean_transcript_text(text: str) -> tuple[str, int, bool]:
    """Strip non-speech tags/symbols while extracting audience reaction hits and lyric flag."""
    reaction_hits = len(REACTION_TAG_RE.findall(text))
    has_music = bool(MUSIC_SYMBOL_RE.search(text) or NON_SPEECH_TAG_RE.search(text))
    cleaned = ANY_BRACKET_TAG_RE.sub(" ", text)
    cleaned = MUSIC_SYMBOL_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    reaction_hits += len(REACTION_WORD_RE.findall(cleaned))
    return cleaned, reaction_hits, has_music


def analyze_audio_signal(video_path: Path, total_duration: float) -> AudioAnalysis:
    """Decode audio via ffmpeg into 8 kHz mono PCM and extract 10 Hz energy, flux, speech-band & modulation (Zero AI)."""
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
        zeros = np.zeros(expected_frames, dtype=np.float32)
        return AudioAnalysis(
            rms_10hz=zeros,
            flux_10hz=zeros,
            silence_mask_10hz=np.ones(expected_frames, dtype=bool),
            speech_band_10hz=zeros,
            modulation_10hz=zeros,
        )

    pcm = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    n_frames = len(pcm) // hop_samples
    if n_frames < 5:
        zeros = np.zeros(expected_frames, dtype=np.float32)
        return AudioAnalysis(
            rms_10hz=zeros,
            flux_10hz=zeros,
            silence_mask_10hz=np.ones(expected_frames, dtype=bool),
            speech_band_10hz=zeros,
            modulation_10hz=zeros,
        )

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

    # Chunked FFT speech-band energy ratio (300-3400 Hz vs 60+ Hz) to separate speech/reactions from bass rumble
    freqs = np.fft.rfftfreq(hop_samples, d=1.0 / sample_rate)
    speech_bins = (freqs >= 300.0) & (freqs <= 3400.0)
    total_bins = freqs >= 60.0
    hann = np.hanning(hop_samples).astype(np.float32)
    speech_band = np.zeros(n_frames, dtype=np.float32)
    chunk_sz = 4096
    for i0 in range(0, n_frames, chunk_sz):
        i1 = min(n_frames, i0 + chunk_sz)
        mag2 = np.abs(np.fft.rfft(frames[i0:i1] * hann, axis=1)) ** 2
        sp_e = np.sum(mag2[:, speech_bins], axis=1)
        tot_e = np.sum(mag2[:, total_bins], axis=1) + 1e-10
        speech_band[i0:i1] = np.clip(sp_e / tot_e, 0.0, 1.0)
    speech_band[silence_mask] = 0.0

    # Short-window syllable/laughter envelope modulation (700ms window around 10 Hz abs-diff)
    env_diff = np.abs(np.diff(rms_norm, prepend=rms_norm[0]))
    mod_raw = uniform_filter1d(env_diff, size=min(7, n_frames))
    mod_p95 = float(np.percentile(mod_raw, 95)) or 1e-4
    modulation = np.clip(mod_raw / max(mod_p95, 1e-4), 0.0, 1.3)

    return AudioAnalysis(
        rms_10hz=rms_norm.astype(np.float32),
        flux_10hz=flux_norm.astype(np.float32),
        silence_mask_10hz=silence_mask,
        speech_band_10hz=speech_band.astype(np.float32),
        modulation_10hz=modulation.astype(np.float32),
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


def _compute_audio_1hz_scores(audio: AudioAnalysis, n_bins_1hz: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Summarize 10 Hz audio into 1 Hz energy excess, speech-activity ratio, and speech-clarity envelope."""
    rms = audio.rms_10hz
    flux = audio.flux_10hz
    speech_10 = audio.speech_band_10hz if audio.speech_band_10hz is not None else np.ones_like(rms)
    mod_10 = audio.modulation_10hz if audio.modulation_10hz is not None else flux
    n_10 = len(rms)

    # Local 30s baseline so quiet segments in loud videos or quiet podcasts scale fairly
    win_10hz = min(max(31, n_10 // 4), 301)
    if win_10hz % 2 == 0:
        win_10hz += 1
    baseline = median_filter(rms, size=win_10hz, mode="nearest")
    dynamic_excess = np.clip((rms - baseline * 0.85) / (baseline + 0.12), 0.0, 2.0)

    energy_1hz = np.zeros(n_bins_1hz, dtype=np.float32)
    active_1hz = np.zeros(n_bins_1hz, dtype=np.float32)
    speech_1hz = np.zeros(n_bins_1hz, dtype=np.float32)
    for sec in range(n_bins_1hz):
        i0 = sec * 10
        i1 = min(n_10, (sec + 1) * 10)
        if i0 >= n_10:
            break
        seg_rms = rms[i0:i1]
        seg_exc = dynamic_excess[i0:i1]
        seg_flux = flux[i0:i1]
        seg_sil = audio.silence_mask_10hz[i0:i1]
        seg_sp = speech_10[i0:i1]
        seg_mod = mod_10[i0:i1]

        clarity = float(0.65 * np.mean(seg_sp) + 0.35 * min(1.0, float(np.mean(seg_mod))))
        energy_1hz[sec] = float(
            0.38 * np.mean(seg_rms)
            + 0.32 * np.mean(seg_exc)
            + 0.18 * np.mean(seg_flux)
            + 0.12 * clarity
        )
        active_1hz[sec] = float(1.0 - np.mean(seg_sil))
        speech_1hz[sec] = clarity

    # Normalize energy curve to [0, 1] via 98th percentile
    p98 = float(np.percentile(energy_1hz, 98)) if n_bins_1hz > 0 else 1.0
    if p98 > 1e-5:
        energy_1hz = np.clip(energy_1hz / p98, 0.0, 1.0)
    return energy_1hz, active_1hz, speech_1hz


def _build_transcript_grid(segments: list[TranscriptSegment], n_bins_1hz: int) -> _TranscriptGrid:
    """Pre-index transcript cues onto a 1 Hz grid for O(1) window scanning even on multi-hour streams."""
    spoken_words_1hz = np.zeros(n_bins_1hz, dtype=np.float32)
    t1_hooks_1hz = np.zeros(n_bins_1hz, dtype=np.float32)
    t2_hooks_1hz = np.zeros(n_bins_1hz, dtype=np.float32)
    q_exclaim_1hz = np.zeros(n_bins_1hz, dtype=np.float32)
    nums_1hz = np.zeros(n_bins_1hz, dtype=np.float32)
    reactions_1hz = np.zeros(n_bins_1hz, dtype=np.float32)
    lyric_1hz = np.zeros(n_bins_1hz, dtype=np.float32)
    sent_start_1hz = np.zeros(n_bins_1hz, dtype=np.float32)
    sent_end_1hz = np.zeros(n_bins_1hz, dtype=np.float32)

    spoken_segments: list[TranscriptSegment] = []
    prev_ended_sentence = True

    for seg in segments:
        cleaned, react_hits, has_music = _clean_transcript_text(seg.text)
        tokens = WORD_TOKEN_RE.findall(cleaned)
        s0 = max(0, min(n_bins_1hz - 1, int(math.floor(seg.start))))
        s1 = max(s0 + 1, min(n_bins_1hz, int(math.ceil(seg.end))))
        span_bins = max(1, s1 - s0)

        if react_hits > 0:
            reactions_1hz[s0:s1] += react_hits / span_bins

        if not tokens:
            if seg.text.rstrip().endswith((".", "?", "!", "।")):
                prev_ended_sentence = True
            continue

        spoken_segments.append(seg)
        spoken_words_1hz[s0:s1] += len(tokens) / span_bins
        if has_music:
            lyric_1hz[s0:s1] += 1.0

        t1 = sum(len(pat.findall(cleaned)) for pat in STRONG_HOOK_PATTERNS)
        t2 = sum(len(pat.findall(cleaned)) for pat in INTENSITY_PATTERNS)
        if has_music:
            t2 *= 0.25
        qe = cleaned.count("?") * 1.35 + cleaned.count("!") * 0.85
        nm = len(NUMBER_RE.findall(cleaned))

        if t1 > 0:
            t1_hooks_1hz[s0:s1] += t1 / span_bins
        if t2 > 0:
            t2_hooks_1hz[s0:s1] += t2 / span_bins
        if qe > 0:
            q_exclaim_1hz[s0:s1] += qe / span_bins
        if nm > 0:
            nums_1hz[s0:s1] += nm / span_bins

        first_tok = tokens[0].lower()
        if prev_ended_sentence and first_tok not in CONTINUATION_STARTERS:
            sent_start_1hz[s0] = 1.0
        ends_sent = cleaned.rstrip().endswith((".", "?", "!", "।"))
        if ends_sent:
            sent_end_1hz[min(n_bins_1hz - 1, s1 - 1)] = 1.0
        prev_ended_sentence = ends_sent

    # Precompute next spoken second index for every second t -> instant opening-delay lookup
    next_speech_sec = np.full(n_bins_1hz, n_bins_1hz, dtype=np.int32)
    nxt = n_bins_1hz
    for sec in range(n_bins_1hz - 1, -1, -1):
        if spoken_words_1hz[sec] > 0.05:
            nxt = sec
        next_speech_sec[sec] = nxt

    spoken_segments.sort(key=lambda s: s.start)
    seg_starts = [s.start for s in spoken_segments]
    return _TranscriptGrid(
        spoken_words_1hz=spoken_words_1hz,
        t1_hooks_1hz=t1_hooks_1hz,
        t2_hooks_1hz=t2_hooks_1hz,
        q_exclaim_1hz=q_exclaim_1hz,
        nums_1hz=nums_1hz,
        reactions_1hz=reactions_1hz,
        lyric_1hz=lyric_1hz,
        sent_start_1hz=sent_start_1hz,
        sent_end_1hz=sent_end_1hz,
        next_speech_sec=next_speech_sec,
        spoken_segments=spoken_segments,
        seg_starts=seg_starts,
    )


def _prefix(arr: np.ndarray) -> np.ndarray:
    out = np.zeros(len(arr) + 1, dtype=np.float64)
    np.cumsum(arr, out=out[1:])
    return out


def _score_transcript_window(
    segments: list[TranscriptSegment],
    start: float,
    end: float,
    grid: _TranscriptGrid | None = None,
) -> tuple[float, list[str], str]:
    """Score a time window deterministically using spoken cadence, bilingual hooks, reactions, cohesion & clean starts."""
    if grid is not None and grid.spoken_segments:
        lo = max(0, bisect.bisect_left(grid.seg_starts, start) - 2)
        win_segs = []
        for s in grid.spoken_segments[lo:]:
            if s.start >= end:
                break
            if s.end > start:
                win_segs.append(s)
    else:
        win_segs = [s for s in segments if s.end > start and s.start < end]

    if not win_segs:
        return 0.0, [], ""

    dur = max(1.0, end - start)
    effective_texts: list[str] = []
    opening_texts: list[str] = []
    effective_word_count = 0.0
    reaction_hits = 0
    lyric_segs = 0
    first_spoken_t: float | None = None
    last_spoken_seg: TranscriptSegment | None = None
    first_spoken_seg: TranscriptSegment | None = None

    for s in win_segs:
        cleaned_seg, seg_react, has_music = _clean_transcript_text(s.text)
        reaction_hits += seg_react
        seg_tokens = WORD_TOKEN_RE.findall(cleaned_seg)
        if not seg_tokens:
            continue

        if has_music:
            lyric_segs += 1
        if first_spoken_seg is None:
            first_spoken_seg = s
        last_spoken_seg = s

        if s.has_word_timing and s.words:
            in_words = []
            for w in s.words:
                if w.end <= start or w.start >= end:
                    continue
                cw = ANY_BRACKET_TAG_RE.sub("", MUSIC_SYMBOL_RE.sub("", w.word)).strip()
                if WORD_TOKEN_RE.search(cw):
                    in_words.append(cw)
                    if first_spoken_t is None or w.start < first_spoken_t:
                        first_spoken_t = max(start, w.start)
            if in_words:
                txt_piece = " ".join(in_words)
                effective_texts.append(txt_piece)
                effective_word_count += len(in_words)
                if s.start <= start + 7.5:
                    opening_texts.append(txt_piece)
        else:
            seg_len = max(0.2, s.end - s.start)
            ov = max(0.0, min(end, s.end) - max(start, s.start))
            frac = min(1.0, ov / seg_len)
            if first_spoken_t is None:
                first_spoken_t = max(start, s.start)
            if frac >= 0.45:
                effective_texts.append(cleaned_seg)
                if s.start <= start + 7.5:
                    opening_texts.append(cleaned_seg)
            effective_word_count += len(seg_tokens) * frac

    joined = " ".join(effective_texts).strip()
    words = WORD_TOKEN_RE.findall(joined)
    word_count = int(round(effective_word_count))
    if word_count < 6 or not words:
        return 0.06, [], joined[:120]

    wps = effective_word_count / dur
    # Sweet spot for engaging spoken content: 1.9 to 4.3 words/sec
    if wps < 1.1:
        cadence_score = (wps / 1.1) * 0.35
    elif wps <= 4.3:
        cadence_score = 0.62 + 0.38 * min(1.0, (wps - 1.1) / 1.7)
    else:
        cadence_score = max(0.5, 1.0 - (wps - 4.3) * 0.14)

    # Lexical diversity + bigram anti-loop check penalizes repetitive lyrics/filler
    lower_words = [w.lower() for w in words]
    unique_ratio = len(set(lower_words)) / max(1, len(lower_words))
    diversity_score = min(1.0, unique_ratio / 0.60)
    if len(lower_words) >= 8:
        bigrams = [(lower_words[i], lower_words[i + 1]) for i in range(len(lower_words) - 1)]
        bigram_ratio = len(set(bigrams)) / max(1, len(bigrams))
        if bigram_ratio < 0.70:
            diversity_score *= max(0.45, bigram_ratio / 0.70)

    mostly_lyrics = lyric_segs >= max(1, len(effective_texts) * 0.6)
    t1_hits = sum(len(pat.findall(joined)) for pat in STRONG_HOOK_PATTERNS)
    t2_hits = sum(len(pat.findall(joined)) for pat in INTENSITY_PATTERNS)
    if mostly_lyrics:
        t2_hits *= 0.25

    q_count = joined.count("?")
    ex_count = joined.count("!")
    num_hits = len(NUMBER_RE.findall(joined))

    opening_joined = " ".join(opening_texts).strip()
    opening_hook = 1.0 if (
        "?" in opening_joined
        or any(p.search(opening_joined) for p in STRONG_HOOK_PATTERNS)
        or (not mostly_lyrics and any(p.search(opening_joined) for p in INTENSITY_PATTERNS))
    ) else 0.0

    # Leading dead-air / instrumental penalty & clean-thought start/end bonuses
    opening_delay = max(0.0, (first_spoken_t - start) if first_spoken_t is not None else dur)
    delay_penalty = max(0.0, (opening_delay - 1.8) * 0.10)
    first_word_low = words[0].lower()
    clean_start_bonus = (
        0.14
        if (
            first_spoken_seg is not None
            and abs(first_spoken_seg.start - start) <= 1.5
            and first_word_low not in CONTINUATION_STARTERS
        )
        else 0.0
    )
    clean_end_bonus = (
        0.10
        if (last_spoken_seg is not None and last_spoken_seg.text.rstrip().endswith((".", "?", "!", "।")))
        else 0.0
    )

    rhetorical_raw = (
        t1_hits * 0.26
        + t2_hits * 0.14
        + q_count * 0.22
        + ex_count * 0.14
        + num_hits * 0.12
        + reaction_hits * 0.16
        + opening_hook * 0.26
        + clean_start_bonus
        + clean_end_bonus
        - delay_penalty
    )
    rhetorical_score = max(0.0, min(1.0, rhetorical_raw))

    total = 0.34 * cadence_score + 0.20 * diversity_score + 0.46 * rhetorical_score
    if opening_delay > 3.5:
        total *= max(0.50, 1.0 - (opening_delay - 3.0) * 0.06)

    tags: list[str] = []
    if opening_hook > 0 or t1_hits >= 1:
        tags.append("Strong Opening Hook")
    if reaction_hits >= 1:
        tags.append("Audience Reaction Peak")
    elif q_count >= 1 or ex_count >= 1:
        tags.append("High Engagement Cue")
    if 2.0 <= wps <= 4.3 and diversity_score >= 0.72:
        tags.append("Crisp Speech Cadence")

    preview = joined
    if len(preview) > 130:
        preview = preview[:127].rsplit(" ", 1)[0] + "..."
    return float(np.clip(total, 0.0, 1.0)), tags, preview


def compute_speech_keep_intervals(
    audio: AudioAnalysis,
    clip_start: float,
    clip_end: float,
    min_silence_dur: float = 0.65,
    pad: float = 0.14,
) -> tuple[list[tuple[float, float]], float]:
    """
    Identify clip-relative [0, clip_dur] intervals to KEEP when trimming long dead-air pauses.
    Returns (intervals, trimmed_ratio). Falls back to full clip if no significant silence exists.
    """
    clip_dur = max(0.5, clip_end - clip_start)
    if len(audio.silence_mask_10hz) < 5:
        return [(0.0, round(clip_dur, 3))], 0.0

    i0 = max(0, int(math.floor(clip_start * audio.sample_rate_hz)))
    i1 = min(len(audio.silence_mask_10hz), int(math.ceil(clip_end * audio.sample_rate_hz)))
    if i1 - i0 < 6:
        return [(0.0, round(clip_dur, 3))], 0.0

    sil = audio.silence_mask_10hz[i0:i1]
    if bool(np.all(sil)):
        return [(0.0, round(clip_dur, 3))], 0.0

    raw_active: list[tuple[float, float]] = []
    idx = 0
    n = len(sil)
    while idx < n:
        if not sil[idx]:
            j = idx
            while j < n and not sil[j]:
                j += 1
            s_rel = max(0.0, (idx / audio.sample_rate_hz) - pad)
            e_rel = min(clip_dur, (j / audio.sample_rate_hz) + pad)
            if e_rel > s_rel + 0.05:
                raw_active.append((s_rel, e_rel))
            idx = j
        else:
            idx += 1

    if not raw_active:
        return [(0.0, round(clip_dur, 3))], 0.0

    merged: list[list[float]] = [[raw_active[0][0], raw_active[0][1]]]
    for s_rel, e_rel in raw_active[1:]:
        if s_rel - merged[-1][1] < min_silence_dur:
            merged[-1][1] = max(merged[-1][1], e_rel)
        else:
            merged.append([s_rel, e_rel])

    keep = [(round(a, 3), round(b, 3)) for a, b in merged if b - a >= 0.25]
    kept_total = sum(b - a for a, b in keep)
    removed = clip_dur - kept_total
    if not keep or removed < 0.45 or kept_total < max(1.2, clip_dur * 0.2):
        return [(0.0, round(clip_dur, 3))], 0.0

    return keep, round(max(0.0, min(0.85, removed / clip_dur)), 3)


def _find_semantic_boundaries(
    audio: AudioAnalysis,
    segments: list[TranscriptSegment],
    duration: float,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Collect scored opening cut points (t, quality_bonus) and closing cut points (t, quality_bonus)
    using sentence punctuation, segment/word gaps, and audio speech onsets/releases.
    """
    openings: dict[float, float] = {0.0: 0.4}
    closings: dict[float, float] = {round(duration, 2): 0.4}

    def _add_open(t: float, bonus: float) -> None:
        key = round(max(0.0, min(duration, t)), 2)
        if bonus > openings.get(key, -99.0):
            openings[key] = bonus

    def _add_close(t: float, bonus: float) -> None:
        key = round(max(0.0, min(duration, t)), 2)
        if bonus > closings.get(key, -99.0):
            closings[key] = bonus

    for idx, seg in enumerate(segments):
        cleaned, _, _ = _clean_transcript_text(seg.text)
        tokens = WORD_TOKEN_RE.findall(cleaned)
        if not tokens:
            continue

        first_tok = tokens[0].lower()
        last_tok = tokens[-1].lower()
        prev_ended = idx == 0 or segments[idx - 1].text.rstrip().endswith((".", "?", "!", "।"))
        gap_before = (seg.start - segments[idx - 1].end) if idx > 0 else seg.start

        open_bonus = 0.0
        if prev_ended:
            open_bonus += 2.2
        if gap_before >= 0.25:
            open_bonus += 1.1
        if first_tok in CONTINUATION_STARTERS and not prev_ended:
            open_bonus -= 1.2
        _add_open(max(0.0, seg.start - 0.16), open_bonus)

        ends_sent = cleaned.rstrip().endswith((".", "?", "!", "।"))
        gap_after = (segments[idx + 1].start - seg.end) if idx + 1 < len(segments) else 1.0
        close_bonus = 0.0
        if ends_sent:
            close_bonus += 2.4
        if gap_after >= 0.25:
            close_bonus += 1.1
        if last_tok in DANGLING_ENDINGS and not ends_sent:
            close_bonus -= 1.4
        _add_close(min(duration, seg.end + 0.18), close_bonus)

        # Intra-segment word-level sentence endings & natural pauses (crucial for rolling auto-captions)
        if seg.has_word_timing and seg.words:
            for w_idx, w in enumerate(seg.words):
                w_clean = w.word.strip()
                if not w_clean:
                    continue
                w_tok = re.sub(r"[^\w]", "", w_clean).lower()
                w_ends = w_clean.endswith((".", "?", "!", "।"))
                w_gap = (seg.words[w_idx + 1].start - w.end) if w_idx + 1 < len(seg.words) else 0.0
                if w_ends or w_gap >= 0.28:
                    c_b = (2.3 if w_ends else 0.9) - (1.2 if (w_tok in DANGLING_ENDINGS and not w_ends) else 0.0)
                    _add_close(min(duration, w.end + 0.16), c_b)
                    if w_idx + 1 < len(seg.words):
                        nxt_tok = re.sub(r"[^\w]", "", seg.words[w_idx + 1].word).lower()
                        o_b = (2.1 if w_ends else 0.9) - (1.1 if (nxt_tok in CONTINUATION_STARTERS and not w_ends) else 0.0)
                        _add_open(max(0.0, seg.words[w_idx + 1].start - 0.15), o_b)

    # Audio silence valleys -> speech-onset (openings) & speech-release (closings)
    sil = audio.silence_mask_10hz
    i = 0
    n = len(sil)
    while i < n:
        if sil[i]:
            j = i
            while j < n and sil[j]:
                j += 1
            if j - i >= 2:
                sil_start = i / audio.sample_rate_hz
                sil_end = j / audio.sample_rate_hz
                _add_close(min(duration, sil_start + 0.18), 0.7)
                _add_open(max(0.0, sil_end - 0.14), 0.7)
            i = j
        else:
            i += 1

    return sorted(openings.items()), sorted(closings.items())


def _find_pause_points(audio: AudioAnalysis, segments: list[TranscriptSegment], duration: float) -> list[float]:
    """Compatibility helper returning sorted natural cut timestamps."""
    openings, closings = _find_semantic_boundaries(audio, segments, duration)
    pts = {p for p, _ in openings} | {p for p, _ in closings}
    return sorted(pts)


def _snap_scored_boundary(
    target: float,
    candidates: list[tuple[float, float]],
    min_bound: float,
    max_bound: float,
    max_shift: float = 3.5,
) -> float:
    """Pick boundary inside [min_bound, max_bound] balancing proximity to target and semantic sentence quality."""
    if min_bound >= max_bound:
        return round(float(min_bound), 2)
    best = float(np.clip(target, min_bound, max_bound))
    best_cost = 999.0
    for pt, bonus in candidates:
        if pt < min_bound or pt > max_bound:
            continue
        dist = abs(pt - target)
        if dist > max_shift:
            continue
        # Lower cost is better: 1s distance costs 1.0, sentence boundary bonus saves up to ~2.4
        cost = dist - bonus * 0.9
        if cost < best_cost:
            best_cost = cost
            best = pt
    return round(float(np.clip(best, min_bound, max_bound)), 2)


def _refine_clip_bounds(
    cand_start: float,
    cand_end: float,
    duration: float,
    min_clip_len: float,
    max_clip_len: float,
    opening_points: list[tuple[float, float]],
    closing_points: list[tuple[float, float]],
    grid: _TranscriptGrid | None,
) -> tuple[float, float]:
    """Advance past dead leading non-speech intros and snap start/end onto clean sentence/pause boundaries."""
    s_target = cand_start
    e_target = cand_end

    if grid is not None and grid.spoken_segments:
        lo = max(0, bisect.bisect_left(grid.seg_starts, cand_start) - 1)
        first_spoken_t: float | None = None
        for seg in grid.spoken_segments[lo:]:
            if seg.start >= cand_end:
                break
            if seg.end > cand_start + 0.35:
                first_spoken_t = max(cand_start, seg.start)
                break
        if first_spoken_t is not None and (first_spoken_t - cand_start) > 1.8:
            desired_len = cand_end - cand_start
            shifted_s = max(0.0, first_spoken_t - 0.22)
            shifted_e = min(duration, max(cand_end, shifted_s + desired_len))
            if shifted_e - shifted_s >= min_clip_len:
                s_target = shifted_s
                e_target = shifted_e

    s_snap = _snap_scored_boundary(
        s_target,
        opening_points,
        min_bound=max(0.0, s_target - 3.5),
        max_bound=min(e_target - min_clip_len, s_target + 3.5),
    )
    e_snap = _snap_scored_boundary(
        e_target,
        closing_points,
        min_bound=max(s_snap + min_clip_len, e_target - 3.5),
        max_bound=min(duration, s_snap + max_clip_len, e_target + 3.5),
    )
    return s_snap, e_snap


def _extract_content_word_set(segments: list[TranscriptSegment], start: float, end: float) -> set[str]:
    words: set[str] = set()
    for s in segments:
        if s.end <= start or s.start >= end:
            continue
        cleaned, _, _ = _clean_transcript_text(s.text)
        for tok in WORD_TOKEN_RE.findall(cleaned):
            low = tok.lower()
            if len(low) >= 3:
                words.add(low)
    return words


def _text_overlap_ratio(a: set[str], b: set[str]) -> float:
    if len(a) < 4 or len(b) < 4:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / max(1, union)


def detect_best_moments(
    video_path: Path,
    duration: float,
    num_clips: int,
    target_duration: int = 35,
    heatmap: list[dict[str, float]] | None = None,
    segments: list[TranscriptSegment] | None = None,
    ollama_model: str = "",
    ollama_topic: str = "",
    ai_provider: str = "auto",
) -> tuple[list[MomentCandidate], AudioAnalysis]:
    """
    Pick top N non-overlapping highlight moments using strictly deterministic (Zero-AI) signals:
    YouTube Most-Replayed heatmap prominence + speech-band audio DSP + climax arc +
    bilingual transcript hooks + clean sentence boundary snapping (with optional local AI boost).
    """
    segments = segments or []
    heatmap = heatmap or []

    audio = analyze_audio_signal(video_path, duration)
    n_bins = max(1, int(math.ceil(duration)))
    energy_1hz, active_1hz, speech_1hz = _compute_audio_1hz_scores(audio, n_bins)
    heatmap_1hz = _build_heatmap_curve(heatmap, n_bins, duration)
    opening_points, closing_points = _find_semantic_boundaries(audio, segments, duration)

    max_feasible_dur = max(6.0, duration / max(1, num_clips) * 0.92)
    base_dur = float(min(target_duration, max_feasible_dur)) if duration < num_clips * (target_duration + 3) else float(target_duration)
    min_clip_len = max(5.0, min(14.0, base_dur * 0.65))
    max_clip_len = min(duration, max(min_clip_len + 2.0, base_dur * 1.25))

    if duration <= 8.0:
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

    win_lengths = sorted({
        max(5, int(round(base_dur * 0.78))),
        max(6, int(round(base_dur))),
        min(int(duration), max(7, int(round(base_dur * 1.2)))),
    })

    step = 1 if duration < 240 else (2 if duration < 1800 else 3)
    has_heatmap = heatmap_1hz is not None and float(np.max(heatmap_1hz)) > 0.05
    grid = _build_transcript_grid(segments, n_bins) if segments else None
    has_transcript = grid is not None and len(grid.spoken_segments) >= 3

    smooth_energy = uniform_filter1d(energy_1hz, size=min(5, len(energy_1hz)))
    heatmap_baseline = (
        median_filter(heatmap_1hz, size=min(max(15, n_bins // 6), 61), mode="nearest")
        if has_heatmap and heatmap_1hz is not None
        else None
    )

    # Precompute 1 Hz prefix sums for fast O(1) multi-scale scanning
    p_e = _prefix(energy_1hz)
    p_e2 = _prefix(energy_1hz * energy_1hz)
    p_act = _prefix(active_1hz)
    p_sp = _prefix(speech_1hz)
    p_h = _prefix(heatmap_1hz) if heatmap_1hz is not None else None
    if grid is not None:
        p_w = _prefix(grid.spoken_words_1hz)
        p_t1 = _prefix(grid.t1_hooks_1hz)
        p_t2 = _prefix(grid.t2_hooks_1hz)
        p_qe = _prefix(grid.q_exclaim_1hz)
        p_num = _prefix(grid.nums_1hz)
        p_react = _prefix(grid.reactions_1hz)
        p_lyric = _prefix(grid.lyric_1hz)
        p_s0 = _prefix(grid.sent_start_1hz)
        p_s1 = _prefix(grid.sent_end_1hz)

    def _fast_score_window(s_int: int, e_int: int) -> dict | None:
        w_dur = max(1, e_int - s_int)
        active_ratio = float((p_act[e_int] - p_act[s_int]) / w_dur)
        if active_ratio < 0.20 and duration > 30:
            return None

        win_e = energy_1hz[s_int:e_int]
        peak_e = float(np.percentile(win_e, 88))
        mean_e = float((p_e[e_int] - p_e[s_int]) / w_dur)
        var_e = max(0.0, float((p_e2[e_int] - p_e2[s_int]) / w_dur) - mean_e * mean_e)
        dyn_std = math.sqrt(var_e)
        open_e = float(np.mean(smooth_energy[s_int : min(e_int, s_int + 5)]))
        open_act = float(np.mean(active_1hz[s_int : min(e_int, s_int + 4)]))
        tail_act = float(np.mean(active_1hz[max(s_int, e_int - 3) : e_int]))
        mean_sp = float((p_sp[e_int] - p_sp[s_int]) / w_dur)

        # Climax arc: reward peaks framed cleanly inside 18%..88% rather than cut right at edges
        peak_rel = float(np.argmax(win_e)) / max(1, len(win_e) - 1) if len(win_e) > 1 else 0.5
        if 0.18 <= peak_rel <= 0.86:
            arc_factor = 1.0
        elif peak_rel < 0.08 or peak_rel > 0.94:
            arc_factor = 0.78
        else:
            arc_factor = 0.90

        a_score = float(
            np.clip(
                0.32 * peak_e
                + 0.26 * mean_e
                + 0.18 * min(1.0, dyn_std * 3.0)
                + 0.14 * open_e
                + 0.10 * mean_sp,
                0.0,
                1.0,
            )
        )
        a_score *= (0.85 + 0.15 * arc_factor)
        if active_ratio < 0.45:
            a_score *= 0.76
        if open_act < 0.25 and duration > 45:
            a_score *= 0.85
        if tail_act < 0.15 and duration > 45:
            a_score *= 0.92

        h_score = 0.0
        if has_heatmap and heatmap_1hz is not None and p_h is not None and heatmap_baseline is not None:
            win_h = heatmap_1hz[s_int:e_int]
            win_hb = heatmap_baseline[s_int:e_int]
            h_max = float(np.max(win_h))
            h_mean = float((p_h[e_int] - p_h[s_int]) / w_dur)
            prominence = float(np.clip(np.max(win_h - win_hb) * 1.6, 0.0, 1.0))
            h_idx = int(np.argmax(win_h))
            h_rel = h_idx / max(1, len(win_h) - 1) if len(win_h) > 1 else 0.5
            h_frame = 1.0 if 0.15 <= h_rel <= 0.85 else 0.82
            h_score = float(np.clip((0.46 * h_max + 0.30 * h_mean + 0.24 * prominence) * h_frame, 0.0, 1.0))

        t_approx = 0.0
        if has_transcript and grid is not None:
            w_cnt = float(p_w[e_int] - p_w[s_int])
            if w_cnt >= 6:
                wps = w_cnt / w_dur
                cad = (wps / 1.1) * 0.35 if wps < 1.1 else (0.62 + 0.38 * min(1.0, (wps - 1.1) / 1.7) if wps <= 4.3 else max(0.5, 1.0 - (wps - 4.3) * 0.14))
                open_end = min(e_int, s_int + 7)
                open_t1 = float(p_t1[open_end] - p_t1[s_int])
                open_qe = float(p_qe[open_end] - p_qe[s_int])
                t1_c = float(p_t1[e_int] - p_t1[s_int])
                t2_c = float(p_t2[e_int] - p_t2[s_int])
                qe_c = float(p_qe[e_int] - p_qe[s_int])
                num_c = float(p_num[e_int] - p_num[s_int])
                react_c = float(p_react[e_int] - p_react[s_int])
                clean_s = 0.14 if grid.sent_start_1hz[s_int] > 0 else 0.0
                clean_e = 0.10 if grid.sent_end_1hz[min(n_bins - 1, e_int - 1)] > 0 else 0.0
                open_hook = 1.0 if (open_t1 > 0 or open_qe >= 0.9) else 0.0
                first_sp = int(grid.next_speech_sec[s_int])
                delay = max(0, min(w_dur, first_sp - s_int))
                delay_pen = max(0.0, (delay - 1.8) * 0.10)
                rhet = max(
                    0.0,
                    min(1.0, t1_c * 0.26 + t2_c * 0.14 + qe_c * 0.18 + num_c * 0.12 + react_c * 0.16 + open_hook * 0.26 + clean_s + clean_e - delay_pen),
                )
                t_approx = float(np.clip(0.38 * cad + 0.16 + 0.46 * rhet, 0.0, 1.0))
                if delay > 3.5:
                    t_approx *= max(0.50, 1.0 - (delay - 3.0) * 0.06)

        if has_heatmap and has_transcript:
            combined = 0.38 * h_score + 0.30 * a_score + 0.32 * t_approx
        elif has_heatmap:
            combined = 0.52 * h_score + 0.48 * a_score
        elif has_transcript:
            combined = 0.48 * a_score + 0.52 * t_approx
        else:
            combined = a_score

        if s_int == 0 and duration > 90 and h_score < 0.55:
            combined *= 0.90

        return {
            "start": float(s_int),
            "end": float(e_int),
            "combined": combined,
            "heatmap_score": h_score,
            "audio_score": a_score,
            "transcript_score": t_approx,
            "t_tags": [],
            "preview": "",
            "open_e": open_e,
            "peak_e": peak_e,
        }

    raw_candidates: list[dict] = []
    for w_len in win_lengths:
        if w_len > duration:
            continue
        for s_int in range(0, max(1, n_bins - w_len + 1), step):
            e_int = min(n_bins, s_int + w_len)
            if e_int - s_int < 5:
                continue
            cand = _fast_score_window(s_int, e_int)
            if cand is not None:
                raw_candidates.append(cand)

    if not raw_candidates:
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

    # Select top well-spaced candidate windows for exact transcript + lexical diversity + 1s local refinement
    raw_candidates.sort(key=lambda c: c["combined"], reverse=True)
    shortlist: list[dict] = []
    min_probe_sep = max(3.0, min(10.0, base_dur * 0.22))
    target_shortlist = max(24, num_clips * 8)

    for c in raw_candidates:
        if any(abs(c["start"] - ex["start"]) < min_probe_sep and abs(c["end"] - ex["end"]) < min_probe_sep for ex in shortlist):
            continue
        shortlist.append(c)
        if len(shortlist) >= target_shortlist:
            break

    # Fine 1s boundary refinement around top peaks when step > 1
    refined_pool: list[dict] = list(shortlist)
    if step > 1:
        seen_pairs = {(int(c["start"]), int(c["end"])) for c in refined_pool}
        for c in shortlist[:14]:
            base_s = int(c["start"])
            w_len = int(c["end"] - c["start"])
            for offset in (-1, 1):
                ns = max(0, base_s + offset)
                ne = min(n_bins, ns + w_len)
                if ne - ns >= 5 and (ns, ne) not in seen_pairs:
                    seen_pairs.add((ns, ne))
                    rcand = _fast_score_window(ns, ne)
                    if rcand is not None:
                        refined_pool.append(rcand)

    # Exact transcript evaluation (diversity, bigram anti-loop, exact opening delay) on shortlist
    for c in refined_pool:
        if has_transcript:
            t_exact, t_tags, preview = _score_transcript_window(segments, c["start"], c["end"], grid=grid)
            c["transcript_score"] = t_exact
            c["t_tags"] = t_tags
            c["preview"] = preview
            if has_heatmap:
                c["combined"] = 0.38 * c["heatmap_score"] + 0.30 * c["audio_score"] + 0.32 * t_exact
            else:
                c["combined"] = 0.48 * c["audio_score"] + 0.52 * t_exact
            if c["start"] == 0.0 and duration > 90 and c["heatmap_score"] < 0.55:
                c["combined"] *= 0.90

    refined_pool.sort(key=lambda c: c["combined"], reverse=True)

    # Optional local AI re-ranking (topic match or general viral hook ranking)
    if ollama_model.strip() and has_transcript:
        from app.ollama_service import score_moments_with_local_ai

        spaced: list[dict] = []
        for c in refined_pool:
            if not c["preview"]:
                continue
            if any(abs(c["start"] - ex["start"]) < max(8.0, base_dur * 0.5) for ex in spaced):
                continue
            spaced.append(c)
            if len(spaced) >= 8:
                break
        if spaced:
            ai_scores = score_moments_with_local_ai(
                ollama_model,
                spaced,
                topic=ollama_topic,
                provider=ai_provider,
            )
            has_topic = bool(ollama_topic.strip())
            ai_wt = 0.58 if has_topic else 0.35
            for idx, score_val in ai_scores.items():
                if 0 <= idx < len(spaced):
                    c_ref = spaced[idx]
                    c_ref["combined"] = (1.0 - ai_wt) * c_ref["combined"] + ai_wt * score_val
                    if score_val >= 0.45:
                        tag = f"Topic: {ollama_topic.strip()[:22]}" if has_topic else "Local AI Viral Pick"
                        c_ref["t_tags"] = [tag] + [t for t in c_ref["t_tags"] if t != tag]
            refined_pool.sort(key=lambda c: c["combined"], reverse=True)

    selected: list[MomentCandidate] = []
    selected_word_sets: list[set[str]] = []
    if duration > max(90.0, num_clips * base_dur * 2.5):
        min_gap = min(20.0, max(2.0, duration * 0.012))
    else:
        min_gap = 2.0 if duration > 90 else 0.5

    for cand in refined_pool:
        if len(selected) >= num_clips:
            break

        s_snap, e_snap = _refine_clip_bounds(
            cand_start=cand["start"],
            cand_end=cand["end"],
            duration=duration,
            min_clip_len=min_clip_len,
            max_clip_len=max_clip_len,
            opening_points=opening_points,
            closing_points=closing_points,
            grid=grid,
        )

        overlaps = any(not (e_snap + min_gap <= ex.start or s_snap >= ex.end + min_gap) for ex in selected)
        if overlaps:
            continue

        cand_words = _extract_content_word_set(segments, s_snap, e_snap) if has_transcript else set()
        if any(_text_overlap_ratio(cand_words, prev_w) > 0.62 for prev_w in selected_word_sets):
            continue

        # Re-evaluate exact transcript preview/tags on snapped bounds for crisp metadata
        t_tags = list(cand["t_tags"])
        preview = cand["preview"]
        t_score_final = cand["transcript_score"]
        if has_transcript:
            t_score_final, snapped_tags, snapped_preview = _score_transcript_window(segments, s_snap, e_snap, grid=grid)
            if snapped_preview:
                preview = snapped_preview
            for tg in snapped_tags:
                if tg not in t_tags:
                    t_tags.append(tg)

        reasons: list[str] = []
        if cand["heatmap_score"] >= 0.55:
            reasons.append("YouTube Most-Replayed Peak")
        if cand["peak_e"] >= 0.65:
            reasons.append("High Vocal Energy")
        elif cand["open_e"] >= 0.50:
            reasons.append("Strong Audio Opening")
        for tag in t_tags:
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
                transcript_score=round(t_score_final * 100, 1),
                reasons=reasons[:3],
                title_snippet=preview,
            )
        )
        selected_word_sets.append(cand_words)

    # Fallback pass: if tight duration or text diversity prevented filling num_clips, relax cleanly
    if len(selected) < num_clips:
        for cand in refined_pool:
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


def find_clip_hook_interval(
    audio: AudioAnalysis,
    clip_start: float,
    clip_end: float,
    duration: float,
    heatmap: list[dict[str, float]] | None = None,
    segments: list[TranscriptSegment] | None = None,
    min_hook_dur: float = 2.4,
    max_hook_dur: float = 3.6,
) -> tuple[float, float] | None:
    """
    Find a punchy 1.8-3.6s micro-moment later inside [clip_start, clip_end] to prepend
    as a video+audio attention-grabbing teaser hook before the main clip plays.
    """
    clip_dur = clip_end - clip_start
    if clip_dur < 7.5:
        return None

    eff_min_hook = 1.8 if clip_dur < 12.0 else min_hook_dur
    offset = min(clip_dur * 0.45, max(3.2 if clip_dur < 12.0 else 4.5, clip_dur * 0.24))
    search_start = clip_start + offset
    search_end = min(duration, clip_end - 0.20)
    if search_end - search_start < eff_min_hook:
        return None

    target_len = min(3.2, max(eff_min_hook, min(max_hook_dur, clip_dur * 0.15)))
    sr = max(1.0, float(audio.sample_rate_hz))
    n_10 = len(audio.rms_10hz)
    speech_10 = audio.speech_band_10hz if audio.speech_band_10hz is not None else np.ones_like(audio.rms_10hz)
    mod_10 = audio.modulation_10hz if audio.modulation_10hz is not None else audio.flux_10hz

    n_bins = max(1, int(math.ceil(duration)))
    heatmap_1hz = _build_heatmap_curve(heatmap or [], n_bins, duration)
    segments = segments or []

    best_score = -1.0
    best_s = search_start
    best_e = min(search_end, search_start + target_len)

    t = search_start
    while t + eff_min_hook <= search_end + 0.05:
        w_end = min(search_end, t + target_len)
        i0 = max(0, int(math.floor(t * sr)))
        i1 = min(n_10, max(i0 + 2, int(math.ceil(w_end * sr))))
        if i1 > i0 and n_10 > 0:
            sil_slice = audio.silence_mask_10hz[i0:i1]
            act_ratio = float(1.0 - np.mean(sil_slice)) if len(sil_slice) else 0.5
            if act_ratio >= 0.30 or not np.any(~audio.silence_mask_10hz):
                rms_sl = audio.rms_10hz[i0:i1]
                flux_sl = audio.flux_10hz[i0:i1]
                sp_sl = speech_10[i0:i1]
                mod_sl = mod_10[i0:i1]
                a_val = (
                    0.40 * float(np.percentile(rms_sl, 85))
                    + 0.25 * float(np.mean(flux_sl))
                    + 0.20 * float(np.mean(sp_sl))
                    + 0.15 * float(np.mean(mod_sl))
                )
            else:
                a_val = 0.05
        else:
            a_val = 0.3

        h_val = 0.0
        if heatmap_1hz is not None:
            s_sec = max(0, int(math.floor(t)))
            e_sec = min(n_bins, max(s_sec + 1, int(math.ceil(w_end))))
            h_val = float(np.max(heatmap_1hz[s_sec:e_sec]))

        t_val = 0.0
        if segments:
            win_txt_parts = []
            for seg in segments:
                if seg.end > t and seg.start < w_end:
                    cleaned, react_hits, _ = _clean_transcript_text(seg.text)
                    if cleaned:
                        win_txt_parts.append(cleaned)
                    if react_hits > 0:
                        t_val += 0.35
            joined = " ".join(win_txt_parts)
            if joined:
                if "?" in joined or "!" in joined:
                    t_val += 0.35
                if any(p.search(joined) for p in STRONG_HOOK_PATTERNS):
                    t_val += 0.40
                elif any(p.search(joined) for p in INTENSITY_PATTERNS):
                    t_val += 0.20
                t_val = min(1.0, t_val)

        score = a_val + 0.45 * h_val + 0.55 * t_val
        if score > best_score:
            best_score = score
            best_s = t
            best_e = w_end
        t += 0.4

    # Snap hook micro-boundaries cleanly to word/segment boundaries so speech isn't chopped mid-word
    word_starts: list[float] = []
    word_ends: list[float] = []
    for seg in segments:
        if seg.end < best_s - 1.0 or seg.start > best_e + 1.0:
            continue
        if seg.has_word_timing and seg.words:
            for w in seg.words:
                word_starts.append(w.start)
                word_ends.append(w.end)
        else:
            word_starts.append(seg.start)
            word_ends.append(seg.end)

    h_start = best_s
    h_end = best_e
    if word_starts:
        near_s = min(word_starts, key=lambda ws: abs(ws - best_s))
        if abs(near_s - best_s) <= 0.75 and near_s >= search_start - 0.5:
            h_start = max(clip_start + 2.5, near_s - 0.08)
        near_e = min(word_ends, key=lambda we: abs(we - (h_start + target_len)))
        if abs(near_e - (h_start + target_len)) <= 0.85 and (near_e - h_start) >= eff_min_hook:
            h_end = min(duration, clip_end, near_e + 0.12)

    if h_end - h_start < eff_min_hook:
        h_end = min(duration, clip_end, h_start + eff_min_hook)
    if h_end - h_start > max_hook_dur:
        h_end = h_start + max_hook_dur
    if h_end - h_start < 1.6:
        return None
    return round(h_start, 2), round(h_end, 2)
