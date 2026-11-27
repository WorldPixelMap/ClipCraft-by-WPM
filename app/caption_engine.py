"""Highlighted Captions Engine: Zero-AI platform captions + offline Whisper-tiny + Hindi/Hinglish support."""

from __future__ import annotations

import os
import re
import subprocess
import threading
from pathlib import Path
from typing import Literal

from app.models import (
    AspectRatio,
    CaptionEngineMode,
    CaptionLanguage,
    HighlightColor,
    TranscriptSegment,
    WordTimestamp,
)
from app.moment_detector import AudioAnalysis
from app.youtube_service import find_ffmpeg_dir

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_WHISPER_LOCK = threading.Lock()
_WHISPER_MODEL = None

HIGHLIGHT_COLORS: dict[HighlightColor, str] = {
    HighlightColor.YELLOW: "&H0000E6FF&",
    HighlightColor.CYAN: "&H00FFF000&",
    HighlightColor.LIME: "&H0032FF55&",
    HighlightColor.CORAL: "&H006D4DFF&",
}

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

# Common Hinglish lexical overrides for natural readable Roman Hindi (Zero AI)
HINGLISH_WORDS: dict[str, str] = {
    "है": "hai", "हैं": "hain", "हूँ": "hoon", "हूं": "hoon", "हो": "ho",
    "मैं": "main", "में": "mein", "हम": "hum", "तुम": "tum", "आप": "aap",
    "नहीं": "nahi", "नही": "nahi", "ना": "na", "क्या": "kya", "क्यों": "kyon",
    "कैसे": "kaise", "कब": "kab", "कहाँ": "kahan", "कहां": "kahan", "कौन": "kaun",
    "और": "aur", "या": "ya", "पर": "par", "लेकिन": "lekin", "मगर": "magar",
    "अगर": "agar", "तो": "toh", "भी": "bhi", "ही": "hi", "से": "se",
    "को": "ko", "का": "ka", "की": "ki", "के": "ke", "ने": "ne",
    "यह": "yeh", "ये": "ye", "वह": "woh", "वो": "wo", "इस": "is", "उस": "us",
    "था": "tha", "थी": "thi", "थे": "the", "हुआ": "hua", "हुई": "hui", "हुए": "hue",
    "गया": "gaya", "गई": "gayi", "गए": "gaye", "होगा": "hoga", "होगी": "hogi", "होंगे": "honge",
    "करना": "karna", "करते": "karte", "करता": "karta", "करती": "karti", "करो": "karo",
    "रहा": "raha", "रही": "rahi", "रहे": "rahe", "वाला": "wala", "वाले": "wale", "वाली": "wali",
    "बहुत": "bahut", "अच्छा": "accha", "अच्छे": "acche", "अच्छी": "acchi",
    "सब": "sab", "कुछ": "kuch", "सिर्फ": "sirf", "अभी": "abhi", "आज": "aaj", "कल": "kal",
    "जब": "jab", "तब": "tab", "यहाँ": "yahan", "वहाँ": "wahan", "भाई": "bhai", "दोस्त": "dost",
    "यार": "yaar", "लोग": "log", "बात": "baat", "देखो": "dekho", "सुनो": "suno",
    "बताओ": "batao", "समझ": "samajh", "क्योंकि": "kyunki", "इसलिए": "isliye",
    "थोड़ा": "thoda", "ज़्यादा": "zyada", "ज्यादा": "zyada", "एक": "ek", "दो": "do", "तीन": "teen",
}

_DEVA_VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo",
    "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o",
}
_DEVA_MATRAS = {
    "ा": "a", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "ृ": "ri",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॉ": "o", "ॅ": "e",
}
_DEVA_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "क़": "q", "ख़": "kh", "ग़": "gh", "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f", "य़": "y",
    "ळ": "l",
}
_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def _transliterate_deva_word(word: str) -> str:
    """Phonetic Devanagari-to-Hinglish transliterator for a single word."""
    stripped = word.strip(".,!?;:\"'()[]")
    if stripped in HINGLISH_WORDS:
        return word.replace(stripped, HINGLISH_WORDS[stripped])

    # Normalize nukta pairs before iterating
    w = word.translate(_DEVA_DIGITS)
    nukta_map = {"क\u093c": "क़", "ख\u093c": "ख़", "ग\u093c": "ग़", "ज\u093c": "ज़", "ड\u093c": "ड़", "ढ\u093c": "ढ़", "फ\u093c": "फ़", "य\u093c": "य़"}
    for seq, comp in nukta_map.items():
        w = w.replace(seq, comp)

    out: list[str] = []
    i = 0
    n = len(w)
    while i < n:
        ch = w[i]
        if ch in _DEVA_CONSONANTS:
            base = _DEVA_CONSONANTS[ch]
            nxt = w[i + 1] if i + 1 < n else ""
            if nxt == "्":  # Halant suppresses inherent vowel
                out.append(base)
                i += 2
            elif nxt in _DEVA_MATRAS:
                out.append(base + _DEVA_MATRAS[nxt])
                i += 2
            else:
                # Apply natural Hindi schwa deletion at word end (when preceded by another consonant/vowel)
                is_word_end = i + 1 >= n or not ("\u0900" <= nxt <= "\u097F")
                if is_word_end and len(out) > 0:
                    out.append(base)
                else:
                    out.append(base + "a")
                i += 1
        elif ch in _DEVA_VOWELS:
            out.append(_DEVA_VOWELS[ch])
            i += 1
        elif ch in ("ं", "ँ"):
            out.append("n")
            i += 1
        elif ch == "ः":
            out.append("h")
            i += 1
        elif ch in ("।", "॥"):
            out.append(".")
            i += 1
        elif ch in _DEVA_MATRAS or ch == "्" or ch == "\u093c":
            i += 1
        else:
            out.append(ch)
            i += 1

    return "".join(out)


def devanagari_to_hinglish(text: str) -> str:
    """Convert Devanagari text spans inside a string to natural Hinglish (Latin script)."""
    if not DEVANAGARI_RE.search(text):
        return text
    tokens = re.split(r"(\s+)", text)
    return "".join(_transliterate_deva_word(tok) if DEVANAGARI_RE.search(tok) else tok for tok in tokens)


def _ffmpeg_exe() -> str:
    ffdir = find_ffmpeg_dir()
    if ffdir:
        cand = Path(ffdir) / "ffmpeg.exe"
        if cand.exists():
            return str(cand)
    return "ffmpeg"


def is_offline_model_cached() -> bool:
    """Check whether the inbuilt tiny faster-whisper model exists locally in models/."""
    model_repo = MODELS_DIR / "models--Systran--faster-whisper-tiny" / "snapshots"
    if not model_repo.exists():
        return False
    return any(p.is_dir() and (p / "model.bin").exists() for p in model_repo.iterdir())


def _get_offline_whisper():
    """Load the minimal CPU int8 Whisper-tiny model as a thread-safe singleton."""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL

    with _WHISPER_LOCK:
        if _WHISPER_MODEL is not None:
            return _WHISPER_MODEL
        from faster_whisper import WhisperModel

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cached = is_offline_model_cached()
        cpu_threads = max(1, min(4, os.cpu_count() or 2))
        _WHISPER_MODEL = WhisperModel(
            "tiny",
            device="cpu",
            compute_type="int8",
            cpu_threads=cpu_threads,
            download_root=str(MODELS_DIR),
            local_files_only=cached,
        )
        return _WHISPER_MODEL


def _pace_segment_words_by_audio(
    text: str,
    seg_start: float,
    seg_end: float,
    audio: AudioAnalysis | None,
) -> list[WordTimestamp]:
    """Zero-AI word timing: weight words by length and place across active audio energy frames."""
    raw_words = text.split()
    if not raw_words:
        return []
    dur = max(0.2, seg_end - seg_start)
    weights = [max(2.0, len(re.sub(r"[^\w]", "", w)) + 1.4) for w in raw_words]
    total_w = sum(weights)

    if audio is None or len(audio.rms_10hz) == 0:
        cur = seg_start
        out: list[WordTimestamp] = []
        for w, wt in zip(raw_words, weights):
            w_dur = dur * (wt / total_w)
            out.append(WordTimestamp(word=w, start=round(cur, 3), end=round(cur + w_dur, 3)))
            cur += w_dur
        return out

    i0 = max(0, int(seg_start * 10))
    i1 = min(len(audio.rms_10hz), max(i0 + 1, int(round(seg_end * 10))))
    active_bins = [idx for idx in range(i0, i1) if not audio.silence_mask_10hz[idx]]
    if len(active_bins) < 2:
        active_bins = list(range(i0, i1))

    bin_times = [(b / 10.0, min(seg_end, (b + 1) / 10.0)) for b in active_bins]
    total_active = sum(max(0.05, bt1 - bt0) for bt0, bt1 in bin_times)
    if total_active <= 0.1:
        return _pace_segment_words_by_audio(text, seg_start, seg_end, None)

    def _map_fraction(frac: float) -> float:
        target = frac * total_active
        acc = 0.0
        for bt0, bt1 in bin_times:
            span = max(0.05, bt1 - bt0)
            if acc + span >= target:
                local = (target - acc) / span
                return bt0 + local * span
            acc += span
        return bin_times[-1][1]

    out = []
    cum = 0.0
    for w, wt in zip(raw_words, weights):
        f0 = cum / total_w
        cum += wt
        f1 = cum / total_w
        t0 = max(seg_start, _map_fraction(f0))
        t1 = min(seg_end, max(t0 + 0.06, _map_fraction(f1)))
        out.append(WordTimestamp(word=w, start=round(t0, 3), end=round(t1, 3)))
    return out


def extract_clip_words_from_youtube(
    segments: list[TranscriptSegment],
    clip_start: float,
    clip_end: float,
    audio: AudioAnalysis | None = None,
) -> tuple[list[WordTimestamp], Literal["youtube_word_sync", "youtube_audio_paced", "none"]]:
    """Slice and normalize platform captions into clip-relative [0, duration] word timestamps (Zero AI)."""
    clip_dur = clip_end - clip_start
    words: list[WordTimestamp] = []
    used_exact = False
    used_paced = False

    for seg in segments:
        if seg.end <= clip_start or seg.start >= clip_end:
            continue
        if seg.has_word_timing and seg.words:
            used_exact = True
            for w in seg.words:
                if w.end <= clip_start or w.start >= clip_end:
                    continue
                ws = max(0.0, w.start - clip_start)
                we = min(clip_dur, w.end - clip_start)
                if we > ws + 0.02:
                    words.append(WordTimestamp(word=w.word, start=round(ws, 3), end=round(we, 3)))
        else:
            used_paced = True
            s_clamped = max(clip_start, seg.start)
            e_clamped = min(clip_end, seg.end)
            if e_clamped - s_clamped < 0.1:
                continue
            paced = _pace_segment_words_by_audio(seg.text, s_clamped, e_clamped, audio)
            for w in paced:
                ws = max(0.0, w.start - clip_start)
                we = min(clip_dur, w.end - clip_start)
                if we > ws + 0.02:
                    words.append(WordTimestamp(word=w.word, start=round(ws, 3), end=round(we, 3)))

    if not words:
        return [], "none"

    source: Literal["youtube_word_sync", "youtube_audio_paced"] = (
        "youtube_word_sync" if (used_exact and not used_paced) else "youtube_audio_paced"
    )
    return words, source


def transcribe_clip_offline_tiny(
    video_path: Path,
    clip_start: float,
    clip_end: float,
    caption_language: CaptionLanguage = CaptionLanguage.AUTO,
) -> list[WordTimestamp]:
    """Run the inbuilt minimal CPU int8 Whisper-tiny model ONLY on the selected clip interval."""
    import numpy as np

    clip_dur = max(0.5, clip_end - clip_start)
    cmd = [
        _ffmpeg_exe(),
        "-v", "error",
        "-ss", f"{clip_start:.3f}",
        "-i", str(video_path),
        "-t", f"{clip_dur:.3f}",
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-f", "s16le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if len(proc.stdout) < 3200:
        return []

    audio_np = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    model = _get_offline_whisper()

    whisper_lang: str | None = None
    if caption_language == CaptionLanguage.ENGLISH:
        whisper_lang = "en"
    elif caption_language in (CaptionLanguage.HINDI, CaptionLanguage.HINGLISH):
        whisper_lang = "hi"

    seg_iter, _ = model.transcribe(
        audio_np,
        language=whisper_lang,
        beam_size=1,
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    words: list[WordTimestamp] = []
    for seg in seg_iter:
        if not seg.words:
            continue
        for w in seg.words:
            token = (w.word or "").strip()
            if not token:
                continue
            ws = max(0.0, float(w.start))
            we = min(clip_dur, max(ws + 0.06, float(w.end)))
            words.append(WordTimestamp(word=token, start=round(ws, 3), end=round(we, 3)))
    return words


def _apply_language_format(words: list[WordTimestamp], caption_language: CaptionLanguage) -> list[WordTimestamp]:
    """Convert Devanagari tokens to Hinglish when Hinglish mode is requested."""
    if caption_language != CaptionLanguage.HINGLISH:
        return words
    converted: list[WordTimestamp] = []
    for w in words:
        tok = devanagari_to_hinglish(w.word)
        converted.append(WordTimestamp(word=tok, start=w.start, end=w.end))
    return converted


def get_clip_words(
    video_path: Path,
    segments: list[TranscriptSegment],
    clip_start: float,
    clip_end: float,
    audio: AudioAnalysis | None,
    mode: CaptionEngineMode,
    work_dir: Path,
    rank: int,
    caption_language: CaptionLanguage = CaptionLanguage.AUTO,
) -> tuple[list[WordTimestamp], Literal["youtube_word_sync", "youtube_audio_paced", "offline_whisper_tiny", "none"], bool]:
    """Resolve word-level timestamps with Zero-AI captions first and inbuilt offline tiny fallback."""
    if mode != CaptionEngineMode.OFFLINE_AI and segments:
        yt_words, yt_source = extract_clip_words_from_youtube(segments, clip_start, clip_end, audio)
        if yt_words:
            return _apply_language_format(yt_words, caption_language), yt_source, False

    if mode == CaptionEngineMode.NO_AI:
        return [], "none", False

    ai_words = transcribe_clip_offline_tiny(video_path, clip_start, clip_end, caption_language)
    if ai_words:
        return _apply_language_format(ai_words, caption_language), "offline_whisper_tiny", True
    return [], "none", True


SRT_TIME_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[.,](\d{3})"
)


def _srt_ts_to_sec(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_custom_script_or_srt(
    script_text: str,
    duration: float,
    audio: AudioAnalysis | None = None,
    caption_language: CaptionLanguage = CaptionLanguage.AUTO,
) -> list[WordTimestamp]:
    """
    Convert a user-provided .SRT or plain script into audio-paced word timestamps (100% Zero AI).
    """
    raw = script_text.strip()
    if not raw:
        return []

    segments: list[TranscriptSegment] = []
    if SRT_TIME_RE.search(raw):
        blocks = re.split(r"\r?\n\s*\r?\n", raw)
        for blk in blocks:
            lines = [ln.strip() for ln in blk.splitlines() if ln.strip()]
            ts_idx = next((i for i, ln in enumerate(lines) if SRT_TIME_RE.search(ln)), -1)
            if ts_idx < 0:
                continue
            m = SRT_TIME_RE.search(lines[ts_idx])
            assert m is not None
            s0 = _srt_ts_to_sec(*m.groups()[:4])
            s1 = _srt_ts_to_sec(*m.groups()[4:])
            text = " ".join(lines[ts_idx + 1 :]).strip()
            if text and s1 > s0:
                segments.append(TranscriptSegment(text=text, start=s0, end=min(duration, s1)))

    if segments:
        words: list[WordTimestamp] = []
        for seg in segments:
            words.extend(_pace_segment_words_by_audio(seg.text, seg.start, seg.end, audio))
        return _apply_language_format(words, caption_language)

    # Plain script: distribute cleanly across active speech audio span (or full reel duration)
    clean_text = re.sub(r"\s+", " ", raw).strip()
    s_start, s_end = 0.15, max(0.5, duration - 0.15)
    if audio is not None and len(audio.silence_mask_10hz) > 4:
        active_idxs = [idx for idx, is_sil in enumerate(audio.silence_mask_10hz) if not is_sil]
        if active_idxs:
            s_start = max(0.0, active_idxs[0] / 10.0)
            s_end = min(duration, max(s_start + 0.5, (active_idxs[-1] + 1) / 10.0))

    words = _pace_segment_words_by_audio(clean_text, s_start, s_end, audio)
    return _apply_language_format(words, caption_language)


def _group_words_into_lines(words: list[WordTimestamp], max_words: int = 4, max_chars: int = 24) -> list[list[WordTimestamp]]:
    """Group timed words into punchy, readable 3-4 word caption phrases."""
    groups: list[list[WordTimestamp]] = []
    cur: list[WordTimestamp] = []
    cur_chars = 0

    for w in words:
        clean = w.word.strip()
        if not clean:
            continue
        gap = (w.start - cur[-1].end) if cur else 0.0
        would_exceed = len(cur) >= max_words or (cur_chars + len(clean) + 1 > max_chars and len(cur) >= 2)
        if cur and (would_exceed or gap > 0.45):
            groups.append(cur)
            cur = []
            cur_chars = 0

        cur.append(w)
        cur_chars += len(clean) + 1
        if clean.endswith((".", "?", "!", ",")) and len(cur) >= 2:
            groups.append(cur)
            cur = []
            cur_chars = 0

    if cur:
        groups.append(cur)
    return groups


def _format_ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    cs = int(round(seconds * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs_rem = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs_rem:02d}"


def _format_srt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms_rem = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms_rem:03d}"


def _escape_ass(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", r"\{").replace("}", r"\}")


def write_highlighted_ass_and_srt(
    words: list[WordTimestamp],
    ass_path: Path,
    srt_path: Path | None,
    video_w: int,
    video_h: int,
    clip_duration: float,
    aspect_ratio: AspectRatio,
    highlight_color: HighlightColor,
    caption_language: CaptionLanguage = CaptionLanguage.AUTO,
    watermark_text: str = "",
) -> str:
    """Write word-by-word highlighted .ass subtitle (plus optional watermark) and .srt file."""
    is_vertical = (video_h >= video_w) if aspect_ratio == AspectRatio.ORIGINAL else (aspect_ratio != AspectRatio.LANDSCAPE)
    font_size = int(round(video_h * (0.044 if is_vertical else 0.068)))
    margin_v = int(round(video_h * (0.18 if is_vertical else 0.075)))
    margin_lr = int(round(video_w * 0.07))
    outline = max(3, int(round(font_size * 0.09)))
    shadow = max(2, int(round(font_size * 0.045)))
    accent_bgr = HIGHLIGHT_COLORS[highlight_color]

    has_devanagari = caption_language == CaptionLanguage.HINDI or any(DEVANAGARI_RE.search(w.word) for w in words)
    font_name = "Nirmala UI" if has_devanagari else "Arial"

    wm_size = max(18, int(round(video_h * 0.022)))
    wm_margin_v = max(18, int(round(video_h * 0.03)))

    ass_header = f"""[Script Info]
Title: Local ClipCraft by WPM
ScriptType: v4.00+
WrapStyle: 2
ScaledBorderAndShadow: yes
PlayResX: {video_w}
PlayResY: {video_h}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},&H00FFFFFF&,&H00FFFFFF&,&H00000000&,&H96000000&,-1,0,0,0,100,100,0,0,1,{outline},{shadow},2,{margin_lr},{margin_lr},{margin_v},1
Style: Watermark,Arial,{wm_size},&H55FFFFFF&,&H55FFFFFF&,&H66000000&,&H00000000&,-1,0,0,0,100,100,0,0,1,2,1,8,20,20,{wm_margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    dialogue_lines: list[str] = []
    srt_entries: list[str] = []

    clean_wm = watermark_text.strip()
    if clean_wm:
        dialogue_lines.append(
            f"Dialogue: 1,0:00:00.00,{_format_ass_time(clip_duration)},Watermark,,0,0,0,,{_escape_ass(clean_wm)}"
        )

    max_words = 4 if is_vertical else 6
    max_chars = 22 if is_vertical else 38
    groups = _group_words_into_lines(words, max_words=max_words, max_chars=max_chars)
    prev_cs_end = 0

    for grp_idx, grp in enumerate(groups, start=1):
        grp_start = max(prev_cs_end / 100.0, grp[0].start)
        grp_end = max(grp_start + 0.1, grp[-1].end)
        plain_line = " ".join(w.word.strip() for w in grp)
        srt_entries.append(
            f"{grp_idx}\n{_format_srt_time(grp_start)} --> {_format_srt_time(grp_end)}\n{plain_line}\n"
        )

        # Keep Devanagari unmodified; uppercase Latin English/Hinglish for crisp viral captions
        display_tokens = [
            _escape_ass(w.word.strip() if DEVANAGARI_RE.search(w.word) else w.word.strip().upper())
            for w in grp
        ]
        for idx, w in enumerate(grp):
            cs_start = max(prev_cs_end, int(round(max(0.0, w.start) * 100)))
            raw_end = grp[idx + 1].start if idx + 1 < len(grp) else w.end
            cs_end = max(cs_start + 5, int(round(raw_end * 100)))
            prev_cs_end = cs_end

            styled_parts = []
            for j, tok in enumerate(display_tokens):
                if j == idx:
                    styled_parts.append(f"{{\\c{accent_bgr}\\fscx109\\fscy109}}{tok}{{\\r}}")
                else:
                    styled_parts.append(tok)
            line_text = " ".join(styled_parts)
            dialogue_lines.append(
                f"Dialogue: 0,{_format_ass_time(cs_start / 100.0)},{_format_ass_time(cs_end / 100.0)},Default,,0,0,0,,{line_text}"
            )

    ass_path.write_text(ass_header + "\n".join(dialogue_lines) + "\n", encoding="utf-8")
    if srt_path and srt_entries:
        srt_path.write_text("\n".join(srt_entries), encoding="utf-8")

    preview = " ".join(w.word.strip() for w in words)
    if len(preview) > 160:
        preview = preview[:157].rsplit(" ", 1)[0] + "..."
    return preview
