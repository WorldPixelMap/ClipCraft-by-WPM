"""Highlighted Captions Engine: Zero-AI platform captions + offline Whisper-tiny + Hindi/Hinglish support."""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Literal

from app.models import (
    AspectRatio,
    CaptionEngineMode,
    CaptionLanguage,
    CaptionPosition,
    CaptionStyle,
    HighlightColor,
    SpeechModelPreset,
    TranscriptSegment,
    WordTimestamp,
)
from app.moment_detector import AudioAnalysis
from app.youtube_service import find_ffmpeg_dir

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_WHISPER_LOCK = threading.Lock()
_WHISPER_MODELS: dict[str, object] = {}

HIGHLIGHT_COLORS: dict[HighlightColor, str] = {
    HighlightColor.YELLOW: "&H0000E6FF&",
    HighlightColor.CYAN: "&H00FFF000&",
    HighlightColor.LIME: "&H0032FF55&",
    HighlightColor.CORAL: "&H006D4DFF&",
}

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
ARABIC_SCRIPT_RE = re.compile(r"[\u0600-\u06FF]")

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
    "सब": "sab", "सबसे": "sabse", "कुछ": "kuch", "सिर्फ": "sirf", "अभी": "abhi", "आज": "aaj", "कल": "kal",
    "जब": "jab", "तब": "tab", "यहाँ": "yahan", "वहाँ": "wahan", "भाई": "bhai", "दोस्त": "dost", "दोस्तों": "doston",
    "यार": "yaar", "लोग": "log", "लोगों": "logon", "बात": "baat", "बातें": "baatein", "देखो": "dekho", "सुनो": "suno",
    "बताओ": "batao", "समझ": "samajh", "समझो": "samjho", "क्योंकि": "kyunki", "इसलिए": "isliye",
    "थोड़ा": "thoda", "ज़्यादा": "zyada", "ज्यादा": "zyada", "एक": "ek", "दो": "do", "तीन": "teen",
    "बड़ा": "bada", "बड़ी": "badi", "बड़े": "bade", "छोटा": "chhota", "छोटी": "chhoti",
    "गलत": "galat", "गलती": "galti", "सही": "sahi", "मज़ा": "maza", "मजा": "maza", "हाल": "haal",
    "वीडियो": "video", "चैनल": "channel", "सब्सक्राइब": "subscribe", "लाइक": "like", "शेयर": "share",
    "कमेंट": "comment", "बिल्कुल": "bilkul", "असली": "asli", "नकली": "nakli", "जल्दी": "jaldi",
    "शुरू": "shuru", "ख़त्म": "khatam", "खत्म": "khatam", "मतलब": "matlab", "चलो": "chalo", "चलिए": "chaliye",
    "हमेशा": "hamesha", "कभी": "kabhi", "फिर": "phir", "फ़िर": "phir", "कैसे": "kaise", "ऐसे": "aise",
    "वैसे": "waise", "जैसे": "jaise", "कितना": "kitna", "कितने": "kitne", "पहला": "pehla", "पहले": "pehle",
    "दूसरा": "doosra", "दूसरे": "doosre", "नया": "naya", "नई": "nayi", "नयी": "nayi", "पुराना": "purana",
    "दुनिया": "duniya", "ज़िंदगी": "zindagi", "जिंदगी": "zindagi", "पैसे": "paise", "पैसा": "paisa",
    "काम": "kaam", "नाम": "naam", "समय": "samay", "वक्त": "waqt", "वक़्त": "waqt", "दिन": "din", "रात": "raat",
    "साथ": "saath", "पास": "paas", "दूर": "door", "अंदर": "andar", "बाहर": "bahar", "ऊपर": "upar", "नीचे": "neeche",
    "सामने": "saamne", "पीछे": "peeche", "बिना": "bina", "जरूरी": "zaroori", "ज़रूरी": "zaroori",
    "ज़रूर": "zaroor", "जरूर": "zaroor", "शायद": "shayad", "पता": "pata", "लगता": "lagta", "मिलता": "milta",
    "मुझको": "mujhko", "मुझे": "mujhe", "तुम्हें": "tumhein", "आपको": "aapko", "सबको": "sabko",
    "हमारा": "hamara", "हमारे": "hamare", "हमारी": "hamari", "तुम्हारा": "tumhara", "मेरा": "mera",
    "मेरे": "mere", "मेरी": "meri", "अपना": "apna", "अपने": "apne", "अपनी": "apni",
    "करेंगे": "karenge", "करूंगा": "karunga", "करूँगा": "karunga", "आया": "aaya", "दिया": "diya", "लिया": "liya", "किया": "kiya",
    "फ़ायदा": "fayda", "फायदा": "fayda", "नुक़सान": "nuksaan", "नुकसान": "nuksaan", "आसान": "aasaan", "मुश्किल": "mushkil",
    # Common live-stream & English loanwords spoken in Hinglish
    "लाइव": "live", "स्ट्रीम": "stream", "स्ट्रीमिंग": "streaming", "गेम": "game", "गेमिंग": "gaming", "गेमप्ले": "gameplay",
    "गाइस": "guys", "गाइज़": "guys", "ब्रो": "bro", "क्लच": "clutch", "चैट": "chat", "सुपरचैट": "superchat",
    "टीम": "team", "मैच": "match", "रैंक": "rank", "किल": "kill", "हेडशॉट": "headshot", "विन": "win",
    "प्ले": "play", "प्लेयर": "player", "प्लेयर्स": "players", "लिंक": "link", "मॉड": "mod", "स्पैम": "spam",
    "क्लिप": "clip", "क्लिप्स": "clips", "शॉर्ट्स": "shorts", "रील": "reel", "रील्स": "reels",
    "स्क्रीन": "screen", "माइक": "mic", "कैमरा": "camera", "फेस": "face", "साउंड": "sound", "पावर": "power",
    "फॉलो": "follow", "थैंक्स": "thanks", "थैंक्यू": "thank you", "ओके": "okay", "सॉरी": "sorry",
    "प्लीज": "please", "प्लीज़": "please", "वेलकम": "welcome", "हेलो": "hello", "हैलो": "hello", "हाय": "hi", "बाय": "bye",
    "टाइम": "time", "स्टार्ट": "start", "एंड": "end", "लेवल": "level", "फाइनली": "finally",
    "सीरियसली": "seriously", "एक्चुअली": "actually", "लिटरली": "literally",
    "क्रेजी": "crazy", "क्रेज़ी": "crazy", "इंसेन": "insane", "सर": "sir", "मैम": "ma'am",
    "यूट्यूब": "YouTube", "ट्विच": "Twitch", "किक": "Kick", "इंस्टाग्राम": "Instagram", "फेसबुक": "Facebook",
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
    "\u0958": "q", "\u0959": "kh", "\u095a": "gh", "\u095b": "z", "\u095c": "r", "\u095d": "rh", "\u095e": "f", "\u095f": "y",
    "ळ": "l",
}
_NUKTA_PRECOMPOSE = {
    "\u0915\u093c": "\u0958",
    "\u0916\u093c": "\u0959",
    "\u0917\u093c": "\u095a",
    "\u091c\u093c": "\u095b",
    "\u0921\u093c": "\u095c",
    "\u0922\u093c": "\u095d",
    "\u092b\u093c": "\u095e",
    "\u092f\u093c": "\u095f",
}
_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def _transliterate_deva_word(word: str) -> str:
    """Phonetic Devanagari-to-Hinglish transliterator for a single word."""
    norm_word = unicodedata.normalize("NFC", word)
    stripped = norm_word.strip(".,!?;:\"'()[]")
    if stripped in HINGLISH_WORDS:
        return norm_word.replace(stripped, HINGLISH_WORDS[stripped])

    w = norm_word.translate(_DEVA_DIGITS)
    for seq, comp in _NUKTA_PRECOMPOSE.items():
        w = w.replace(seq, comp)
    # Strip any rare non-standard nukta after consonants so attached matras still match cleanly
    w = w.replace("\u093c", "")

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
                matra = "i" if (nxt == "ी" and i + 2 >= n) else _DEVA_MATRAS[nxt]
                out.append(base + matra)
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


def _is_model_cached(model_name: str) -> bool:
    """Return True when a specific faster-whisper model snapshot exists locally in models/."""
    model_repo = MODELS_DIR / f"models--Systran--faster-whisper-{model_name}" / "snapshots"
    if not model_repo.exists():
        return False
    return any(p.is_dir() and (p / "model.bin").exists() for p in model_repo.iterdir())


def is_offline_model_cached() -> bool:
    """Check whether any inbuilt faster-whisper model exists locally in models/."""
    return any(_is_model_cached(name) for name in ("small", "base", "tiny"))


def _load_whisper_instance(model_name: str):
    """Load and cache a specific CPU int8 WhisperModel instance safely."""
    with _WHISPER_LOCK:
        if model_name in _WHISPER_MODELS:
            return _WHISPER_MODELS[model_name]
        from faster_whisper import WhisperModel

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cached = _is_model_cached(model_name)
        cpu_threads = max(1, min(4, os.cpu_count() or 2))
        inst = WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8",
            cpu_threads=cpu_threads,
            download_root=str(MODELS_DIR),
            local_files_only=cached,
        )
        _WHISPER_MODELS[model_name] = inst
        return inst


def _get_offline_whisper(
    speech_model: SpeechModelPreset = SpeechModelPreset.AUTO,
    caption_language: CaptionLanguage = CaptionLanguage.AUTO,
):
    """Resolve and load the most accurate available offline Whisper model with clean offline fallback."""
    if speech_model == SpeechModelPreset.TINY:
        candidates = ["tiny"]
    elif speech_model == SpeechModelPreset.BASE:
        candidates = ["base", "tiny"]
    elif speech_model == SpeechModelPreset.SMALL:
        candidates = ["small", "base", "tiny"]
    else:
        if _is_model_cached("small"):
            candidates = ["small", "base", "tiny"]
        elif caption_language in (CaptionLanguage.HINDI, CaptionLanguage.HINGLISH):
            candidates = ["small", "base", "tiny"]
        else:
            candidates = ["base", "tiny"]

    last_exc: Exception | None = None
    for name in candidates:
        try:
            return _load_whisper_instance(name), name
        except Exception as exc:
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("No offline Whisper model could be loaded.")


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


HINDI_DEVANAGARI_PROMPT = "नमस्ते दोस्तों, आज की लाइव स्ट्रीम में आपका स्वागत है। हिंदी में बिल्कुल सही और स्पष्ट शब्द।"
ENGLISH_PROMPT = "Clear, accurate live stream commentary and conversation in natural English."
_DEVA_LEADING_MARK_RE = re.compile(r"^[\u0900-\u0903\u093a-\u094f\u0951-\u0957\u0962-\u0963\u0971]+")
_INDIC_LANGS = {"hi", "ur", "pa", "bn", "mr", "ne"}
_NOISY_EN_FALSE_POSITIVES = {"cy", "nn", "la", "haw", "yo", "sn", "jw", "su", "mi", "fo", "mg", "tt"}
_WHISPER_HALLUCINATION_RE = re.compile(
    r"^(?:subtitles\s+by|amara\.org|transcribed\s+by|translated\s+by|copyright\b|mbc\b|all\s+rights\s+reserved|thank\s+you\s+for\s+watching\.?|thanks\s+for\s+watching\.?|please\s+subscribe\.?|प्रस्तुति\s*[:\-]?)\s*$",
    re.I,
)


def _merge_indic_word_tokens(raw_words: list[WordTimestamp]) -> list[WordTimestamp]:
    """Attach split Devanagari combining marks/matras back onto the preceding word cleanly."""
    merged: list[WordTimestamp] = []
    for w in raw_words:
        tok = (w.word or "").strip()
        if not tok:
            continue
        if merged and _DEVA_LEADING_MARK_RE.match(tok):
            prev = merged[-1]
            merged[-1] = WordTimestamp(
                word=prev.word + tok,
                start=prev.start,
                end=max(prev.end, w.end),
            )
        else:
            merged.append(WordTimestamp(word=tok, start=w.start, end=w.end))
    return merged


def _translate_phrases(
    phrases: list[str],
    target_lang: str = "hi",
    source_lang: str = "auto",
    ollama_model: str = "",
    ai_provider: str = "auto",
) -> list[str] | None:
    """Translate short caption phrases via local Ollama/LM Studio or free zero-key web translation."""
    clean_phrases = [re.sub(r"\s+", " ", p).strip() for p in phrases]
    if not clean_phrases or not any(clean_phrases):
        return None

    target_label = "natural conversational Hindi in Devanagari script" if target_lang == "hi" else "clear natural English"
    if ollama_model.strip():
        try:
            from app.ollama_service import _local_ai_generate_json

            numbered = "\n".join(f"{i}: {p}" for i, p in enumerate(clean_phrases) if p)
            prompt = (
                f"Translate each numbered line accurately into {target_label}.\n"
                "Return JSON with key 'lines' mapping each string index to its translation.\n"
                f"Lines:\n{numbered}"
            )
            res = _local_ai_generate_json(ollama_model, prompt, provider=ai_provider, timeout=22.0)
            if res and isinstance(res.get("lines"), dict):
                out = []
                ok = True
                for i, orig in enumerate(clean_phrases):
                    tr = str(res["lines"].get(str(i)) or "").strip()
                    if orig and target_lang == "hi" and not DEVANAGARI_RE.search(tr):
                        ok = False
                        break
                    out.append(tr or orig)
                if ok and len(out) == len(clean_phrases):
                    return out
        except Exception:
            pass

    def _gtx_batch(items: list[str]) -> list[str] | None:
        if not items:
            return []
        joined = "\n".join(items)
        query = urllib.parse.urlencode(
            {"client": "gtx", "sl": source_lang, "tl": target_lang, "dt": "t", "q": joined}
        )
        req = urllib.request.Request(
            f"https://translate.googleapis.com/translate_a/single?{query}",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
            full = "".join(str(seg[0]) for seg in (data[0] or []) if seg and seg[0])
            parts = [ln.strip() for ln in full.splitlines()]
            if len(parts) == len(items) and all(parts):
                return parts
        except Exception:
            return None
        return None

    translated: list[str] = []
    chunk_size = 12
    for i0 in range(0, len(clean_phrases), chunk_size):
        chunk = clean_phrases[i0 : i0 + chunk_size]
        res_chunk = _gtx_batch(chunk)
        if res_chunk is None:
            res_chunk = []
            for item in chunk:
                if not item:
                    res_chunk.append("")
                    continue
                single = _gtx_batch([item])
                if not single:
                    return None
                res_chunk.append(single[0])
        translated.extend(res_chunk)

    return translated if len(translated) == len(clean_phrases) else None


def translate_phrases_to_hindi(
    phrases: list[str],
    ollama_model: str = "",
    ai_provider: str = "auto",
) -> list[str] | None:
    """Translate short caption phrases to Hindi (Devanagari)."""
    return _translate_phrases(
        phrases,
        target_lang="hi",
        source_lang="auto",
        ollama_model=ollama_model,
        ai_provider=ai_provider,
    )


def translate_phrases_to_english(
    phrases: list[str],
    ollama_model: str = "",
    ai_provider: str = "auto",
) -> list[str] | None:
    """Translate short caption phrases to natural English."""
    return _translate_phrases(
        phrases,
        target_lang="en",
        source_lang="auto",
        ollama_model=ollama_model,
        ai_provider=ai_provider,
    )


def _translate_words(
    words: list[WordTimestamp],
    target_lang: str,
    source_lang: str = "auto",
    audio: AudioAnalysis | None = None,
    ollama_model: str = "",
    ai_provider: str = "auto",
) -> list[WordTimestamp] | None:
    """Group timed words into natural phrases, translate to target_lang, and re-pace word timings."""
    if not words:
        return None

    phrase_groups: list[list[WordTimestamp]] = []
    cur: list[WordTimestamp] = []
    for w in words:
        tok = w.word.strip()
        if not tok:
            continue
        gap = (w.start - cur[-1].end) if cur else 0.0
        if cur and (len(cur) >= 10 or gap > 0.45):
            phrase_groups.append(cur)
            cur = []
        cur.append(w)
        if tok.endswith((".", "?", "!", "।")) and len(cur) >= 3:
            phrase_groups.append(cur)
            cur = []
    if cur:
        phrase_groups.append(cur)

    if not phrase_groups:
        return None

    src_phrases = [" ".join(w.word.strip() for w in g) for g in phrase_groups]
    tr_phrases = _translate_phrases(
        src_phrases,
        target_lang=target_lang,
        source_lang=source_lang,
        ollama_model=ollama_model,
        ai_provider=ai_provider,
    )
    if not tr_phrases:
        return None

    out_words: list[WordTimestamp] = []
    for grp, tr_txt in zip(phrase_groups, tr_phrases):
        p_start = grp[0].start
        p_end = max(p_start + 0.25, grp[-1].end)
        text_to_pace = tr_txt.strip() or " ".join(w.word.strip() for w in grp)
        out_words.extend(_pace_segment_words_by_audio(text_to_pace, p_start, p_end, audio))
    return out_words or None


def translate_words_to_hindi(
    words: list[WordTimestamp],
    audio: AudioAnalysis | None = None,
    ollama_model: str = "",
    ai_provider: str = "auto",
    source_lang: str = "auto",
) -> list[WordTimestamp] | None:
    """Group timed words into natural phrases, translate to Hindi Devanagari, and re-pace word timings."""
    return _translate_words(
        words,
        target_lang="hi",
        source_lang=source_lang,
        audio=audio,
        ollama_model=ollama_model,
        ai_provider=ai_provider,
    )


def translate_words_to_english(
    words: list[WordTimestamp],
    audio: AudioAnalysis | None = None,
    ollama_model: str = "",
    ai_provider: str = "auto",
) -> list[WordTimestamp] | None:
    """Group timed words into natural phrases, translate to English, and re-pace word timings."""
    return _translate_words(
        words,
        target_lang="en",
        source_lang="auto",
        audio=audio,
        ollama_model=ollama_model,
        ai_provider=ai_provider,
    )


def _words_match_requested_language(words: list[WordTimestamp], caption_language: CaptionLanguage) -> bool:
    """Ensure cached/downloaded platform captions actually match the user's chosen language script."""
    if not words or caption_language == CaptionLanguage.AUTO:
        return bool(words)
    joined = " ".join(w.word for w in words)
    if caption_language in (CaptionLanguage.HINDI, CaptionLanguage.HINGLISH) and ARABIC_SCRIPT_RE.search(joined):
        return False
    alpha_chars = [ch for ch in joined if ch.isalpha()]
    if not alpha_chars:
        return True
    deva_count = sum(1 for ch in alpha_chars if "\u0900" <= ch <= "\u097F")
    latin_count = sum(1 for ch in alpha_chars if ("a" <= ch.lower() <= "z"))
    total = len(alpha_chars)

    if caption_language == CaptionLanguage.HINDI:
        return (deva_count / total) >= 0.25
    if caption_language == CaptionLanguage.ENGLISH:
        return (latin_count / total) >= 0.55 and deva_count == 0
    if caption_language == CaptionLanguage.HINGLISH:
        return ((deva_count + latin_count) / total) >= 0.45
    return True


def _extract_whisper_words(seg_iter, clip_dur: float) -> list[WordTimestamp]:
    """Collect word timestamps from faster-whisper segments while filtering silence hallucinations."""
    words: list[WordTimestamp] = []
    for seg in seg_iter:
        seg_txt = re.sub(r"\s+", " ", (getattr(seg, "text", "") or "")).strip()
        if not seg_txt or _WHISPER_HALLUCINATION_RE.match(seg_txt):
            continue
        no_speech = float(getattr(seg, "no_speech_prob", 0.0) or 0.0)
        avg_lp = float(getattr(seg, "avg_logprob", 0.0) or 0.0)
        if no_speech > 0.85 and avg_lp < -1.15:
            continue
        if not getattr(seg, "words", None):
            continue
        for w in seg.words:
            token = (w.word or "").strip()
            if not token:
                continue
            # Collapse runaway 4+ identical consecutive single-word hallucinations
            if len(words) >= 3 and all(prev.word.lower() == token.lower() for prev in words[-3:]):
                continue
            ws = max(0.0, float(w.start))
            we = min(clip_dur, max(ws + 0.06, float(w.end)))
            words.append(WordTimestamp(word=token, start=round(ws, 3), end=round(we, 3)))
    return _merge_indic_word_tokens(words)


def transcribe_clip_offline_tiny(
    video_path: Path,
    clip_start: float,
    clip_end: float,
    caption_language: CaptionLanguage = CaptionLanguage.AUTO,
    speech_model: SpeechModelPreset = SpeechModelPreset.AUTO,
) -> list[WordTimestamp]:
    """Run offline CPU int8 Whisper (small/base/tiny) on normalized audio for the selected clip interval."""
    import numpy as np

    clip_dur = max(0.5, clip_end - clip_start)
    cmd = [
        _ffmpeg_exe(),
        "-v", "error",
        "-ss", f"{clip_start:.3f}",
        "-i", str(video_path),
        "-t", f"{clip_dur:.3f}",
        "-vn",
        "-af", "highpass=f=70,aresample=async=1:first_pts=0",
        "-ac", "1",
        "-ar", "16000",
        "-f", "s16le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if len(proc.stdout) < 3200:
        return []

    audio_np = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if len(audio_np) == 0:
        return []
    audio_np = audio_np - float(np.mean(audio_np))
    peak = float(np.max(np.abs(audio_np)))
    if 0.002 < peak < 0.75:
        audio_np = np.clip(audio_np * (0.85 / peak), -1.0, 1.0)

    model, model_name = _get_offline_whisper(speech_model=speech_model, caption_language=caption_language)

    vad_params = {
        "threshold": 0.35,
        "min_speech_duration_ms": 150,
        "min_silence_duration_ms": 450,
        "speech_pad_ms": 220,
    }
    detected_lang, detected_prob = "", 0.0
    try:
        detected_lang, detected_prob, _ = model.detect_language(
            audio_np,
            vad_filter=True,
            vad_parameters=vad_params,
            language_detection_segments=3,
        )
    except Exception:
        try:
            detected_lang, detected_prob, _ = model.detect_language(
                audio_np,
                vad_filter=False,
                language_detection_segments=3,
            )
        except Exception:
            detected_lang, detected_prob = "", 0.0

    whisper_lang: str | None = None
    whisper_task: str = "transcribe"
    initial_prompt: str | None = None

    if caption_language == CaptionLanguage.ENGLISH:
        if (
            not detected_lang
            or detected_lang == "en"
            or detected_prob < 0.72
            or detected_lang in _NOISY_EN_FALSE_POSITIVES
        ):
            whisper_lang = "en"
            whisper_task = "transcribe"
            initial_prompt = ENGLISH_PROMPT
        else:
            whisper_lang = detected_lang
            whisper_task = "translate"
    elif caption_language == CaptionLanguage.HINDI:
        if detected_lang == "en" and detected_prob >= 0.88:
            whisper_lang = "en"
            whisper_task = "transcribe"
            initial_prompt = ENGLISH_PROMPT
        else:
            whisper_lang = "hi"
            whisper_task = "transcribe"
            initial_prompt = HINDI_DEVANAGARI_PROMPT
    elif caption_language == CaptionLanguage.HINGLISH:
        if detected_lang == "en" and detected_prob >= 0.88:
            whisper_lang = "en"
            whisper_task = "transcribe"
            initial_prompt = ENGLISH_PROMPT
        else:
            whisper_lang = "hi"
            whisper_task = "transcribe"
            initial_prompt = HINDI_DEVANAGARI_PROMPT
    else:
        if detected_lang in {"hi", "ur"}:
            whisper_lang = "hi"
            initial_prompt = HINDI_DEVANAGARI_PROMPT
        elif detected_lang and detected_prob >= 0.45 and detected_lang not in _NOISY_EN_FALSE_POSITIVES:
            whisper_lang = detected_lang

    beam_sz = 2 if model_name == "small" else 3
    transcribe_kwargs = dict(
        language=whisper_lang,
        task=whisper_task,
        beam_size=beam_sz,
        word_timestamps=True,
        condition_on_previous_text=False,
        repetition_penalty=1.12,
        no_repeat_ngram_size=3,
        hallucination_silence_threshold=1.5,
        initial_prompt=initial_prompt,
    )
    seg_iter, _ = model.transcribe(
        audio_np,
        vad_filter=True,
        vad_parameters=vad_params,
        **transcribe_kwargs,
    )
    words = _extract_whisper_words(seg_iter, clip_dur)

    # If VAD filtered out quiet/music-backed live speech, retry without VAD before giving up
    if len(words) < 2 and peak >= 0.015:
        seg_iter_raw, _ = model.transcribe(audio_np, vad_filter=False, **transcribe_kwargs)
        words_raw = _extract_whisper_words(seg_iter_raw, clip_dur)
        if len(words_raw) > len(words):
            words = words_raw

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
    ollama_model: str = "",
    ai_provider: str = "auto",
    speech_model: SpeechModelPreset = SpeechModelPreset.AUTO,
) -> tuple[list[WordTimestamp], Literal["youtube_word_sync", "youtube_audio_paced", "offline_whisper_tiny", "none"], bool]:
    """Resolve word-level timestamps with Zero-AI captions first, high-accuracy offline Whisper fallback & clean translation."""
    yt_words: list[WordTimestamp] = []
    yt_source: Literal["youtube_word_sync", "youtube_audio_paced", "none"] = "none"

    if mode != CaptionEngineMode.OFFLINE_AI and segments:
        yt_words, yt_source = extract_clip_words_from_youtube(segments, clip_start, clip_end, audio)
        if yt_words and (mode == CaptionEngineMode.NO_AI or _words_match_requested_language(yt_words, caption_language)):
            return _apply_language_format(yt_words, caption_language), yt_source, False

    if mode == CaptionEngineMode.NO_AI:
        return [], "none", False

    # If platform gave clean captions in another language (e.g. English when tlang=hi hit 429), translate cleanly
    if yt_words and caption_language in (CaptionLanguage.HINDI, CaptionLanguage.HINGLISH):
        translated_hi = translate_words_to_hindi(
            yt_words, audio=audio, ollama_model=ollama_model, ai_provider=ai_provider
        )
        if translated_hi:
            return _apply_language_format(translated_hi, caption_language), yt_source, False
        return _apply_language_format(yt_words, caption_language), yt_source, False
    if yt_words and caption_language == CaptionLanguage.ENGLISH:
        translated_en = translate_words_to_english(
            yt_words, audio=audio, ollama_model=ollama_model, ai_provider=ai_provider
        )
        if translated_en:
            return translated_en, yt_source, False

    ai_words = transcribe_clip_offline_tiny(
        video_path,
        clip_start,
        clip_end,
        caption_language,
        speech_model=speech_model,
    )
    if ai_words:
        joined_ai = " ".join(w.word for w in ai_words)
        has_arabic = bool(ARABIC_SCRIPT_RE.search(joined_ai))
        has_deva = bool(DEVANAGARI_RE.search(joined_ai))

        if caption_language == CaptionLanguage.ENGLISH and not _words_match_requested_language(
            ai_words, CaptionLanguage.ENGLISH
        ):
            translated_en = translate_words_to_english(
                ai_words, audio=audio, ollama_model=ollama_model, ai_provider=ai_provider
            )
            if translated_en:
                return translated_en, "offline_whisper_tiny", True
        elif caption_language == CaptionLanguage.HINDI and (
            has_arabic or not _words_match_requested_language(ai_words, CaptionLanguage.HINDI)
        ):
            translated_hi = translate_words_to_hindi(
                ai_words,
                audio=audio,
                ollama_model=ollama_model,
                ai_provider=ai_provider,
                source_lang="ur" if has_arabic else "auto",
            )
            if translated_hi:
                return translated_hi, "offline_whisper_tiny", True
        elif caption_language == CaptionLanguage.HINGLISH:
            if has_arabic or not has_deva:
                # Only translate when Urdu/Arabic script slipped in or audio was purely non-Hindi/non-mixed
                if has_arabic or not _words_match_requested_language(ai_words, CaptionLanguage.HINGLISH):
                    translated_hi = translate_words_to_hindi(
                        ai_words,
                        audio=audio,
                        ollama_model=ollama_model,
                        ai_provider=ai_provider,
                        source_lang="ur" if has_arabic else "auto",
                    )
                    if translated_hi:
                        return _apply_language_format(translated_hi, CaptionLanguage.HINGLISH), "offline_whisper_tiny", True
                elif not has_deva and _words_match_requested_language(ai_words, CaptionLanguage.ENGLISH):
                    translated_hi = translate_words_to_hindi(
                        ai_words, audio=audio, ollama_model=ollama_model, ai_provider=ai_provider
                    )
                    if translated_hi:
                        return _apply_language_format(translated_hi, CaptionLanguage.HINGLISH), "offline_whisper_tiny", True
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


def _group_words_into_lines(
    words: list[WordTimestamp],
    max_words: int = 4,
    max_chars: int = 24,
    hook_split_sec: float | None = None,
) -> list[list[WordTimestamp]]:
    """Group timed words into punchy, readable 3-4 word caption phrases."""
    groups: list[list[WordTimestamp]] = []
    cur: list[WordTimestamp] = []
    cur_chars = 0

    for w in words:
        clean = w.word.strip()
        if not clean:
            continue
        gap = (w.start - cur[-1].end) if cur else 0.0
        crossed_hook = bool(
            hook_split_sec is not None
            and cur
            and cur[-1].end <= hook_split_sec + 0.05
            and w.start >= hook_split_sec - 0.05
        )
        would_exceed = len(cur) >= max_words or (cur_chars + len(clean) + 1 > max_chars and len(cur) >= 2)
        if cur and (would_exceed or gap > 0.45 or crossed_hook):
            groups.append(cur)
            cur = []
            cur_chars = 0

        cur.append(w)
        cur_chars += len(clean) + 1
        if clean.endswith((".", "?", "!", ",", "।")) and len(cur) >= 2:
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


def remap_words_after_jump_cuts(
    words: list[WordTimestamp],
    keep_intervals: list[tuple[float, float]],
) -> list[WordTimestamp]:
    """Remap clip-relative [0, dur] word timestamps onto a jump-cut compacted timeline."""
    if not words or not keep_intervals:
        return words

    # Precompute cumulative output offsets for each kept interval
    mapped_intervals: list[tuple[float, float, float]] = []
    acc = 0.0
    for a, b in keep_intervals:
        if b > a + 0.02:
            mapped_intervals.append((a, b, acc))
            acc += b - a
    if not mapped_intervals:
        return words

    remapped: list[WordTimestamp] = []
    for w in words:
        best_iv: tuple[float, float, float] | None = None
        best_overlap = -1.0
        for a, b, off in mapped_intervals:
            ov = min(w.end, b) - max(w.start, a)
            if ov > best_overlap:
                best_overlap = ov
                best_iv = (a, b, off)
        if best_iv is None or best_overlap <= 0.01:
            # Attach to nearest kept interval edge if word landed right on a cut pad
            mid = (w.start + w.end) / 2.0
            nearest = min(mapped_intervals, key=lambda iv: min(abs(mid - iv[0]), abs(mid - iv[1])))
            a, b, off = nearest
            t0 = off + max(0.0, min(b - a - 0.06, mid - a))
            t1 = min(off + (b - a), t0 + max(0.06, min(0.35, w.end - w.start)))
        else:
            a, b, off = best_iv
            t0 = off + max(0.0, w.start - a)
            t1 = off + min(b - a, w.end - a)
        if t1 > t0 + 0.02:
            remapped.append(WordTimestamp(word=w.word, start=round(t0, 3), end=round(t1, 3)))
    return remapped


def align_edited_text_to_words(
    edited_text: str,
    existing_words: list[WordTimestamp],
    clip_duration: float,
    audio: AudioAnalysis | None = None,
    caption_language: CaptionLanguage = CaptionLanguage.AUTO,
) -> list[WordTimestamp]:
    """Align user-edited caption text onto existing word timings (preserving cadence when counts match)."""
    tokens = [tok for tok in re.split(r"\s+", edited_text.strip()) if tok]
    if not tokens:
        return []
    if existing_words and len(tokens) == len(existing_words):
        out = [
            WordTimestamp(word=tok, start=orig.start, end=orig.end)
            for tok, orig in zip(tokens, existing_words)
        ]
        return _apply_language_format(out, caption_language)

    s_start = existing_words[0].start if existing_words else 0.1
    s_end = existing_words[-1].end if existing_words else max(0.5, clip_duration - 0.1)
    s_end = max(s_start + 0.3, min(clip_duration, s_end))
    paced = _pace_segment_words_by_audio(" ".join(tokens), s_start, s_end, audio)
    return _apply_language_format(paced, caption_language)


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
    caption_style: CaptionStyle = CaptionStyle.CLASSIC,
    caption_position: CaptionPosition = CaptionPosition.BOTTOM,
    caption_scale: float = 1.0,
    hook_banner: str = "",
    hook_split_sec: float | None = None,
) -> str:
    """Write word-by-word highlighted .ass subtitle (plus optional watermark & hook banner) and .srt file."""
    is_vertical = (video_h >= video_w) if aspect_ratio == AspectRatio.ORIGINAL else (aspect_ratio != AspectRatio.LANDSCAPE)
    scale = max(0.75, min(1.4, float(caption_scale)))
    base_ratio = 0.044 if is_vertical else 0.068
    if caption_style == CaptionStyle.MINIMAL:
        base_ratio *= 0.86
    elif caption_style == CaptionStyle.POP:
        base_ratio *= 1.08

    font_size = int(round(video_h * base_ratio * scale))
    clean_hook = re.sub(r"\s+", " ", hook_banner).strip()
    if caption_position == CaptionPosition.CENTER:
        alignment = 5
        margin_v = 0
    elif caption_position == CaptionPosition.TOP:
        alignment = 8
        margin_v = int(round(video_h * (0.24 if (clean_hook and is_vertical) else (0.13 if is_vertical else 0.075))))
    else:
        alignment = 2
        margin_v = int(round(video_h * (0.18 if is_vertical else 0.075)))

    margin_lr = int(round(video_w * 0.07))
    accent_bgr = HIGHLIGHT_COLORS[highlight_color]

    if caption_style == CaptionStyle.BOXED:
        bold_flag = -1
        border_style = 3
        outline = max(7, int(round(font_size * 0.22)))
        shadow = 0
        outline_col = "&HC8000000&"
        back_col = "&HC8000000&"
    elif caption_style == CaptionStyle.MINIMAL:
        bold_flag = 0
        border_style = 1
        outline = max(2, int(round(font_size * 0.06)))
        shadow = 1
        outline_col = "&H00000000&"
        back_col = "&H66000000&"
    else:
        bold_flag = -1
        border_style = 1
        outline = max(3, int(round(font_size * (0.105 if caption_style == CaptionStyle.POP else 0.09))))
        shadow = max(2, int(round(font_size * 0.045)))
        outline_col = "&H00000000&"
        back_col = "&H96000000&"

    has_devanagari = (
        caption_language == CaptionLanguage.HINDI
        or bool(DEVANAGARI_RE.search(clean_hook))
        or any(DEVANAGARI_RE.search(w.word) for w in words)
    )
    font_name = "Nirmala UI" if has_devanagari else "Arial"

    wm_size = max(18, int(round(video_h * 0.022)))
    wm_margin_v = max(18, int(round(video_h * 0.03)))
    hook_size = max(22, int(round(video_h * (0.031 if is_vertical else 0.046))))
    if aspect_ratio == AspectRatio.SPLIT_SCREEN:
        hook_margin_v = max(36, int(round(video_h * 0.055)))
    elif is_vertical:
        hook_margin_v = max(54, int(round(video_h * 0.115)))
    else:
        hook_margin_v = max(42, int(round(video_h * 0.065)))
    hook_margin_lr = int(round(video_w * (0.10 if is_vertical else 0.08)))
    hook_pad = max(8, int(round(hook_size * 0.24)))

    ass_header = f"""[Script Info]
Title: Local ClipCraft by WPM
ScriptType: v4.00+
WrapStyle: 2
ScaledBorderAndShadow: yes
PlayResX: {video_w}
PlayResY: {video_h}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},&H00FFFFFF&,&H00FFFFFF&,{outline_col},{back_col},{bold_flag},0,0,0,100,100,0,0,{border_style},{outline},{shadow},{alignment},{margin_lr},{margin_lr},{margin_v},1
Style: Watermark,Arial,{wm_size},&H55FFFFFF&,&H55FFFFFF&,&H66000000&,&H00000000&,-1,0,0,0,100,100,0,0,1,2,1,8,20,20,{wm_margin_v},1
Style: HookBanner,{font_name},{hook_size},&H0000E6FF&,&H0000E6FF&,&HD80F172A&,&HD80F172A&,-1,0,0,0,100,100,0,0,3,{hook_pad},0,8,{hook_margin_lr},{hook_margin_lr},{hook_margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    dialogue_lines: list[str] = []
    srt_entries: list[str] = []

    clean_wm = watermark_text.strip()
    if clean_wm:
        dialogue_lines.append(
            f"Dialogue: 2,0:00:00.00,{_format_ass_time(clip_duration)},Watermark,,0,0,0,,{_escape_ass(clean_wm)}"
        )

    if clean_hook:
        hook_end = min(4.5, max(1.0, clip_duration))
        dialogue_lines.append(
            f"Dialogue: 1,0:00:00.00,{_format_ass_time(hook_end)},HookBanner,,0,0,0,,{{\\fad(120,240)}}{_escape_ass(clean_hook)}"
        )

    if caption_style == CaptionStyle.POP:
        max_words = 2 if is_vertical else 3
        max_chars = 14 if is_vertical else 24
    else:
        max_words = 4 if is_vertical else 6
        max_chars = 22 if is_vertical else 38
    groups = _group_words_into_lines(
        words,
        max_words=max_words,
        max_chars=max_chars,
        hook_split_sec=hook_split_sec,
    )
    prev_cs_end = 0

    for grp_idx, grp in enumerate(groups, start=1):
        grp_start = max(prev_cs_end / 100.0, grp[0].start)
        grp_end = max(grp_start + 0.1, grp[-1].end)
        plain_line = " ".join(w.word.strip() for w in grp)
        srt_entries.append(
            f"{grp_idx}\n{_format_srt_time(grp_start)} --> {_format_srt_time(grp_end)}\n{plain_line}\n"
        )

        # Keep Devanagari and Minimal style unmodified; uppercase Latin English/Hinglish for punchy captions
        display_tokens = [
            _escape_ass(
                w.word.strip()
                if (DEVANAGARI_RE.search(w.word) or caption_style == CaptionStyle.MINIMAL)
                else w.word.strip().upper()
            )
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
                    if caption_style == CaptionStyle.POP:
                        styled_parts.append(f"{{\\c{accent_bgr}\\fscx118\\fscy118\\t(0,90,\\fscx106\\fscy106)}}{tok}{{\\r}}")
                    elif caption_style == CaptionStyle.MINIMAL:
                        styled_parts.append(f"{{\\c{accent_bgr}\\b1}}{tok}{{\\r}}")
                    else:
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
