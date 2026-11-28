"""Zero-AI Best Hook Generator, Social Title, Caption, and Hashtag Suggester for Shorts & Reels."""

from __future__ import annotations

import re
from collections import Counter

from app.caption_engine import DEVANAGARI_RE, devanagari_to_hinglish, translate_phrases_to_hindi
from app.models import CaptionLanguage, MomentCandidate, SocialSuggestion, WordTimestamp

_DEVA_LEADING_MARK_RE = re.compile(r"^[\u0900-\u0903\u093a-\u094f\u0951-\u0957\u0962-\u0963\u0971]+")

STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "you", "your", "are", "was", "were",
    "have", "has", "had", "not", "but", "what", "when", "where", "who", "why", "how",
    "all", "any", "can", "did", "does", "just", "like", "more", "most", "from", "into",
    "out", "about", "they", "them", "their", "there", "then", "than", "here", "one",
    "two", "also", "very", "really", "actually", "going", "gonna", "want", "know",
    "think", "mean", "right", "well", "yeah", "okay", "look", "said", "says", "she",
    "him", "his", "her", "our", "its", "been", "being", "will", "would", "could",
    "should", "because", "while", "some", "much", "many", "even", "only", "back",
    "time", "make", "made", "take", "get", "got", "see", "way", "thing", "things",
    "hai", "hain", "nahi", "kya", "aur", "mein", "main", "tum", "aap", "yeh", "woh",
    "tha", "thi", "the", "kar", "raha", "rahi", "rahe", "toh", "bhi", "par", "liye",
    "है", "हैं", "हूँ", "हूं", "हो", "मैं", "में", "हम", "तुम", "आप", "नहीं", "नही",
    "क्या", "क्यों", "कैसे", "कब", "कहाँ", "कौन", "और", "या", "पर", "लेकिन", "मगर",
    "अगर", "तो", "भी", "ही", "से", "को", "का", "की", "के", "ने", "यह", "ये", "वह",
    "वो", "इस", "उस", "था", "थी", "थे", "हुआ", "हुई", "हुए", "गया", "गई", "गए",
    "होगा", "होगी", "करना", "करते", "करता", "रहा", "रही", "रहे", "वाला", "वाले",
    "वाली", "बहुत", "सब", "कुछ", "सिर्फ", "अभी", "आज", "जब", "तब", "एक", "दो",
    "लिए", "साथ", "कर", "दिया", "लिया", "किया", "बाद", "पास", "अपने", "अपना", "अपनी",
}

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।])\s+|\n+")
CLAUSE_SPLIT_RE = re.compile(r"(?:\s*[;—–:]\s*|,\s+|\s+(?:because|until|unless|before|instead)\s+)", re.I)
BRACKET_TAG_RE = re.compile(r"\[[^\]]+\]")
MUSIC_NOTES_RE = re.compile(r"[♪♫]+")
SPEAKER_PREFIX_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 _-]{0,18}:\s+")
FILLER_START_RE = re.compile(
    r"^(?:(?:so|well|yeah|yes|okay|ok|um+|uh+|ah+|oh|like|basically|literally|"
    r"actually|you know|i mean|listen|look|alright|anyway|toh|achha|dekho|haan)\b[\s,:-]*)+",
    re.I,
)

HOOK_CUES_STRONG = re.compile(
    r"\b(why|how|secret|truth|mistake|never|stop|don'?t|nobody|problem|reason|"
    r"turns out|what happens|wait|watch|biggest|worst|best|hack|rule|warning|"
    r"kyon|kyun|kaise|galti|asli|sach|raaz|sabse|kabhi|mat)\b|"
    r"(क्यों|कैसे|गलती|असली|सच|सबसे|कभी|मत)",
    re.I,
)
HOOK_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b|\$\d+")
WEAK_HOOK_WORDS = {"yeah", "okay", "ok", "yes", "no", "hmm", "uh", "um", "hello", "hi", "thanks", "thank"}


def _clean_clip_transcript(raw: str, caption_language: CaptionLanguage = CaptionLanguage.AUTO) -> str:
    """Strip subtitle noise tags ([♪♪♪], [Music], speaker prefixes) and normalize Hinglish if requested."""
    if not raw:
        return ""
    txt = BRACKET_TAG_RE.sub(" ", raw)
    # Treat musical symbols as line breaks so adjacent sung lines stay separated cleanly
    txt = MUSIC_NOTES_RE.sub("\n", txt)
    txt = re.sub(r"[ \t\r\f\v]+", " ", txt)
    txt = re.sub(r"\n+", "\n", txt).strip()
    if caption_language == CaptionLanguage.HINGLISH and DEVANAGARI_RE.search(txt):
        txt = devanagari_to_hinglish(txt)
    return txt


def _strip_filler_start(text: str) -> str:
    cleaned = SPEAKER_PREFIX_RE.sub("", text.strip())
    # Only strip filler prefixes when substantial content remains afterward
    stripped = FILLER_START_RE.sub("", cleaned).strip(" ,:-")
    if len(stripped.split()) >= 3:
        return stripped
    return cleaned.strip(" ,:-")


def _extract_hook_candidates(
    clip_text: str,
    caption_language: CaptionLanguage = CaptionLanguage.AUTO,
) -> list[tuple[str, float]]:
    """Extract and rank clean hook phrases from clip transcript (Zero AI)."""
    cleaned = _clean_clip_transcript(clip_text, caption_language=caption_language)
    if not cleaned:
        return []

    raw_sents = [s.strip() for s in SENTENCE_SPLIT_RE.split(cleaned) if len(s.strip()) >= 8]
    if not raw_sents:
        raw_sents = [cleaned]

    scored: list[tuple[str, float]] = []
    seen_norm: set[str] = set()
    total_sents = max(1, len(raw_sents))

    def _add_candidate(raw_phrase: str, sent_idx: int, is_full_sent: bool) -> None:
        phrase = _strip_filler_start(raw_phrase).strip(" \"'“”‘’,-:;—–")
        words = re.findall(r"[A-Za-z0-9'\u0900-\u097F]+", phrase)
        if len(words) < 3:
            return
        if all(w.lower() in WEAK_HOOK_WORDS for w in words):
            return
        norm = re.sub(r"[^a-z0-9\u0900-\u097f]+", "", phrase.lower())
        if not norm or norm in seen_norm:
            return
        seen_norm.add(norm)

        char_len = len(phrase)
        score = 0.0
        # Ideal concise hook length: 14..46 chars (fits cleanly in 1-2 short lines on 9:16)
        if 14 <= char_len <= 46:
            score += 2.6
        elif 47 <= char_len <= 64:
            score += 1.4
        elif char_len < 14:
            score += 0.9
        else:
            score += 0.3

        if is_full_sent:
            score += 0.5
        if "?" in phrase:
            score += 2.4
        if "!" in phrase:
            score += 1.2
        score += min(3, len(HOOK_CUES_STRONG.findall(phrase))) * 1.45
        if HOOK_NUM_RE.search(phrase):
            score += 1.15
        # Favor phrases appearing near the opening of the clip
        early_bonus = max(0.0, 1.3 * (1.0 - (sent_idx / total_sents)))
        score += early_bonus
        scored.append((phrase, score))

    for idx, sent in enumerate(raw_sents[:12]):
        _add_candidate(sent, idx, is_full_sent=True)
        for sub in CLAUSE_SPLIT_RE.split(sent):
            sub_c = sub.strip()
            if len(sub_c) >= 12 and sub_c != sent:
                _add_candidate(sub_c, idx, is_full_sent=False)

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def _condense_for_banner(phrase: str, max_chars: int = 46) -> str:
    """Shorten a phrase gracefully at word boundaries without ugly ellipsis when concise."""
    clean = re.sub(r"\s+", " ", phrase).strip(" ,;:-")
    if len(clean) <= max_chars:
        return clean

    # Try taking first clause if split cleanly by punctuation
    for sep in (" — ", " - ", ": ", ", "):
        if sep in clean:
            head = clean.split(sep, 1)[0].strip()
            if 12 <= len(head) <= max_chars:
                return head

    # Trim at word boundary while preserving trailing ? or !
    end_punct = clean[-1] if clean[-1] in ("?", "!") else ""
    words = clean.rstrip("?.!").split()
    out: list[str] = []
    cur_len = 0
    budget = max_chars - (1 if end_punct else 0)
    for w in words:
        add_len = len(w) + (1 if out else 0)
        if cur_len + add_len > budget:
            break
        out.append(w)
        cur_len += add_len
    if len(out) >= 3:
        return (" ".join(out) + end_punct).strip()
    return clean[: max_chars - 1].rsplit(" ", 1)[0].strip()


def _format_banner_case(text: str, caption_language: CaptionLanguage) -> str:
    if caption_language == CaptionLanguage.HINDI or DEVANAGARI_RE.search(text):
        return text.strip()
    return text.strip().upper()


def _normalize_hook_key(text: str) -> str:
    return re.sub(r"[^a-z0-9\u0900-\u097f]+", "", text.lower())


def generate_best_hook_banner(
    clip_text: str,
    video_title: str,
    moment: MomentCandidate,
    platform: str = "YouTube",
    caption_language: CaptionLanguage = CaptionLanguage.AUTO,
    video_tags: list[str] | None = None,
    used_hooks: set[str] | None = None,
    words: list[WordTimestamp] | None = None,
) -> str:
    """
    Generate a punchy, clip-specific <= 48 char on-screen Hook Banner (first ~4s) for any generated clip.
    Guarantees distinct hooks across clips in a batch and supports English, Hindi, and Hinglish (Zero AI).
    """
    used = used_hooks if used_hooks is not None else set()
    video_tags = video_tags or []

    # Prefer opening transcript words if clip_text wasn't already supplied
    source_text = clip_text.strip()
    if not source_text and words:
        source_text = " ".join(w.word.strip() for w in words if w.word.strip())

    candidates = _extract_hook_candidates(source_text, caption_language=caption_language)
    for phrase, _score in candidates:
        banner = _condense_for_banner(phrase, max_chars=46)
        if len(banner) < 10:
            continue
        formatted = _format_banner_case(banner, caption_language)
        key = _normalize_hook_key(formatted)
        if key and key not in used:
            used.add(key)
            return formatted

    # Context-aware fallback hooks using moment signals + title topic keywords (distinct per clip rank)
    topic_words = _extract_keywords(source_text, video_title, video_tags, limit=3)
    topic = topic_words[0].upper() if topic_words else ""
    reasons_low = " ".join(moment.reasons).lower()

    if caption_language == CaptionLanguage.HINDI:
        fallbacks = [
            "सबसे धमाकेदार मोमेंट देखें!",
            "आखिर तक ज़रूर देखें!",
            "ये रिएक्शन मिस मत करना!",
            "यही वो असली पल था!",
            "ध्यान से सुनिए ये बात!",
        ]
    elif caption_language == CaptionLanguage.HINGLISH:
        fallbacks = [
            "SABSE BEST MOMENT DEKHO!",
            "AAKHIR TAK MISS MAT KARNA!",
            "YEH REACTION DEKHO BHAI!",
            "ASLI TWIST YAHAN AAYA!",
            "SUNO KYA BOLA ISNE!",
        ]
    else:
        fallbacks = []
        if "most-replayed" in reasons_low:
            fallbacks.append("MOST REPLAYED MOMENT")
        if "reaction" in reasons_low or "vocal energy" in reasons_low:
            fallbacks.append("WAIT FOR THIS REACTION!")
        if topic and len(topic) <= 14:
            fallbacks.append(f"WAIT FOR THIS {topic} MOMENT")
            fallbacks.append(f"THE BEST {topic} CLIP")
        fallbacks.extend([
            "WAIT FOR WHAT HAPPENS NEXT",
            "WATCH THIS PART CLOSELY!",
            "THE MOMENT EVERYONE MISSED",
            "YOU NEED TO SEE THIS!",
            "THIS CHANGED EVERYTHING!",
            f"TOP {platform.upper()} HIGHLIGHT",
        ])

    for fb in fallbacks:
        formatted = _format_banner_case(_condense_for_banner(fb, max_chars=46), caption_language)
        key = _normalize_hook_key(formatted)
        if key and key not in used:
            used.add(key)
            return formatted

    # Final guaranteed-unique fallback for large batches
    base_fb = f"PART {moment.rank}: TOP HIGHLIGHT"
    formatted = _format_banner_case(base_fb, caption_language)
    used.add(_normalize_hook_key(formatted))
    return formatted


def _sanitize_hashtag(raw: str, caption_language: CaptionLanguage = CaptionLanguage.AUTO) -> str:
    txt = raw.strip()
    if caption_language == CaptionLanguage.HINGLISH and DEVANAGARI_RE.search(txt):
        txt = devanagari_to_hinglish(txt)
    cleaned = re.sub(r"[^A-Za-z0-9\u0900-\u097F]", "", txt)
    cleaned = _DEVA_LEADING_MARK_RE.sub("", cleaned)
    if len(cleaned) < 2 or cleaned.isdigit():
        return ""
    return f"#{cleaned[:24]}"


def _extract_keywords(
    text: str,
    video_title: str,
    video_tags: list[str],
    limit: int = 5,
    caption_language: CaptionLanguage = CaptionLanguage.AUTO,
) -> list[str]:
    """Extract crisp topic keywords deterministically from transcript, title, and tags (supports Hindi/Hinglish)."""
    candidates: list[str] = []

    for tag in video_tags:
        src_tag = devanagari_to_hinglish(tag) if (caption_language == CaptionLanguage.HINGLISH and DEVANAGARI_RE.search(tag)) else tag
        toks = re.findall(r"[A-Za-z0-9\u0900-\u097F]+", src_tag)
        if 1 <= len(toks) <= 2:
            joined = "".join(t.capitalize() if (len(toks) > 1 and t.isascii()) else t for t in toks)
            joined = _DEVA_LEADING_MARK_RE.sub("", joined)
            if joined.lower() not in STOPWORDS and len(joined) >= 2:
                candidates.append(joined)

    cleaned_text = _clean_clip_transcript(text, caption_language=caption_language)
    words = re.findall(r"[A-Za-z]{3,}|[\u0900-\u097F]{2,}", f"{cleaned_text} {video_title}")
    counts = Counter(
        _DEVA_LEADING_MARK_RE.sub("", w.lower())
        for w in words
        if w.lower() not in STOPWORDS and len(_DEVA_LEADING_MARK_RE.sub("", w)) >= 2
    )
    for word, _ in counts.most_common(12):
        if word not in {c.lower() for c in candidates}:
            candidates.append(word)

    return candidates[:limit]


def _pick_hook_title(
    clip_text: str,
    video_title: str,
    moment: MomentCandidate,
    platform: str,
    caption_language: CaptionLanguage = CaptionLanguage.AUTO,
) -> str:
    """Build a punchy <= 70 char title from the clip's best hook sentence or clean source title."""
    ranked = _extract_hook_candidates(clip_text, caption_language=caption_language)
    chosen = ""
    if ranked:
        best_phrase = ranked[0][0]
        if len(best_phrase) <= 68:
            chosen = best_phrase
        else:
            chosen = best_phrase[:65].rsplit(" ", 1)[0] + "..."

    if not chosen:
        clean_flat = re.sub(r"\s+", " ", _clean_clip_transcript(clip_text, caption_language)).strip()
        sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(clean_flat) if len(s.strip()) >= 12]
        if sentences:
            first = _strip_filler_start(sentences[0])
            chosen = first if len(first) <= 68 else (first[:65].rsplit(" ", 1)[0] + "...")

    if not chosen:
        base_title = re.sub(r"\s+", " ", video_title).strip()
        if len(base_title) > 52:
            base_title = base_title[:49].rsplit(" ", 1)[0] + "..."
        reason_hint = moment.reasons[0] if moment.reasons else f"{platform} Highlight"
        chosen = f"{base_title} ({reason_hint})"

    return chosen[0].upper() + chosen[1:] if len(chosen) > 1 else chosen


def build_platform_captions(
    title: str,
    body_block: str,
    video_title: str,
    channel: str,
    platform: str,
    moment: MomentCandidate,
    hashtags: list[str],
    channel_handle: str = "",
    channel_url: str = "",
    source_url: str = "",
) -> tuple[str, str, str, str]:
    """Assemble YouTube and Facebook captions with explicit source video creator credit."""
    hashtag_line = " ".join(hashtags)
    reason_line = " • ".join(moment.reasons[:2]) if moment.reasons else "Top Highlight"

    creator_name = (channel or "").strip() or "Original Creator"
    handle = (channel_handle or "").strip()
    if handle and handle.lower() not in creator_name.lower():
        creator_credit = f"{creator_name} ({handle})"
    else:
        creator_credit = creator_name

    clean_chan_url = (channel_url or "").strip()
    credit_line = (
        f"Credit: {creator_credit} — {clean_chan_url}"
        if clean_chan_url.startswith(("http://", "https://"))
        else f"Credit: {creator_credit}"
    )

    clean_src_url = (source_url or "").strip()
    has_public_src = clean_src_url.startswith(("http://", "https://")) and "example.com/local" not in clean_src_url
    source_line = f"Original Video: {video_title} ({platform})"
    if has_public_src:
        source_line += f"\nWatch Full Video: {clean_src_url}"

    body_part = f"{body_block.strip()}\n\n" if body_block.strip() else ""

    yt_caption = (
        f"{title}\n\n"
        f"{body_part}"
        f"{credit_line}\n"
        f"{source_line}\n"
        f"Moment signal: {reason_line}\n\n"
        f"{hashtag_line}"
    )

    is_hindi_copy = bool(DEVANAGARI_RE.search(title) or DEVANAGARI_RE.search(body_block))
    fb_cta = (
        "कमेंट में अपनी राय ज़रूर लिखें! ऐसे ही और हाइलाइट्स के लिए फॉलो करें।"
        if is_hindi_copy
        else f"Drop your thoughts in the comments! Follow for more {platform} highlights."
    )

    fb_caption = (
        f"{title} 👇🔥\n\n"
        f"{body_part}"
        f"{fb_cta}\n\n"
        f"{credit_line}\n"
        f"{source_line}\n\n"
        f"{hashtag_line}"
    )

    return (
        yt_caption,
        fb_caption,
        f"Title:\n{title}\n\nDescription & Hashtags:\n{yt_caption}",
        f"Title:\n{title}\n\nPost Caption & Hashtags:\n{fb_caption}",
    )


def suggest_social_metadata(
    clip_text: str,
    video_title: str,
    channel: str,
    platform: str,
    moment: MomentCandidate,
    video_tags: list[str] | None = None,
    caption_language: CaptionLanguage = CaptionLanguage.AUTO,
    channel_handle: str = "",
    channel_url: str = "",
    source_url: str = "",
) -> SocialSuggestion:
    """Generate copy-paste ready Title, YouTube/Facebook Captions with creator credit, and Hashtags without AI."""
    video_tags = video_tags or []
    title = _pick_hook_title(
        clip_text, video_title, moment, platform, caption_language=caption_language
    )
    if caption_language == CaptionLanguage.HINDI and not DEVANAGARI_RE.search(title):
        tr = translate_phrases_to_hindi([title])
        if tr and tr[0]:
            title = tr[0][:70]
    keywords = _extract_keywords(
        clip_text, video_title, video_tags, limit=5, caption_language=caption_language
    )

    raw_tags: list[str] = []
    for kw in keywords:
        raw_tags.append(_sanitize_hashtag(kw, caption_language=caption_language))

    channel_tag = _sanitize_hashtag(channel, caption_language=caption_language)
    if channel_tag and len(channel_tag) <= 22:
        raw_tags.append(channel_tag)

    if platform == "Twitch":
        raw_tags.extend(["#Twitch", "#TwitchClips", "#GamingClips"])
    elif platform == "Kick":
        raw_tags.extend(["#Kick", "#KickClips", "#StreamHighlights"])
    else:
        raw_tags.extend(["#Shorts", "#YouTubeShorts", "#Reels"])

    if caption_language == CaptionLanguage.HINDI:
        raw_tags.extend(["#Hindi", "#HindiShorts", "#HindiReels"])
    elif caption_language == CaptionLanguage.HINGLISH:
        raw_tags.extend(["#Hinglish", "#DesiReels", "#HindiShorts"])

    raw_tags.extend(["#FacebookReels", "#Clips", "#ViralMoments"])

    hashtags: list[str] = []
    seen_lower: set[str] = set()
    for t in raw_tags:
        if t and t.lower() not in seen_lower:
            seen_lower.add(t.lower())
            hashtags.append(t)
        if len(hashtags) >= 10:
            break

    snippet = re.sub(r"\s+", " ", _clean_clip_transcript(clip_text, caption_language)).strip()
    if len(snippet) > 140:
        snippet = snippet[:137].rsplit(" ", 1)[0] + "..."
    if caption_language == CaptionLanguage.HINDI and snippet and not DEVANAGARI_RE.search(snippet):
        tr_snip = translate_phrases_to_hindi([snippet])
        if tr_snip and tr_snip[0]:
            snippet = tr_snip[0][:160]
    quote_part = f'"{snippet}"' if snippet else ""

    yt_caption, fb_caption, copy_yt, copy_fb = build_platform_captions(
        title=title,
        body_block=quote_part,
        video_title=video_title,
        channel=channel,
        platform=platform,
        moment=moment,
        hashtags=hashtags,
        channel_handle=channel_handle,
        channel_url=channel_url,
        source_url=source_url,
    )

    return SocialSuggestion(
        title=title,
        youtube_caption=yt_caption,
        facebook_caption=fb_caption,
        hashtags=hashtags,
        copy_all_youtube=copy_yt,
        copy_all_facebook=copy_fb,
    )
