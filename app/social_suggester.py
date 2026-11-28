"""Zero-AI Social Title, Caption, and Hashtag Generator for YouTube Shorts & Facebook Reels."""

from __future__ import annotations

import re
from collections import Counter

from app.models import CaptionLanguage, MomentCandidate, SocialSuggestion

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
}

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _sanitize_hashtag(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw.strip())
    if len(cleaned) < 2 or cleaned.isdigit():
        return ""
    return f"#{cleaned[:24]}"


def _extract_keywords(text: str, video_title: str, video_tags: list[str], limit: int = 5) -> list[str]:
    """Extract crisp topic keywords deterministically from transcript, title, and tags."""
    candidates: list[str] = []

    # 1. Include concise metadata tags first
    for tag in video_tags:
        toks = re.findall(r"[A-Za-z0-9]+", tag)
        if 1 <= len(toks) <= 2:
            joined = "".join(t.capitalize() if len(toks) > 1 else t for t in toks)
            if joined.lower() not in STOPWORDS and len(joined) >= 3:
                candidates.append(joined)

    # 2. Significant words from clip transcript + video title
    words = re.findall(r"[A-Za-z]{3,}", f"{text} {video_title}")
    counts = Counter(w.lower() for w in words if w.lower() not in STOPWORDS)
    for word, _ in counts.most_common(12):
        if word not in {c.lower() for c in candidates}:
            candidates.append(word)

    return candidates[:limit]


def _pick_hook_title(
    clip_text: str,
    video_title: str,
    moment: MomentCandidate,
    platform: str,
) -> str:
    """Build a punchy <= 70 char title from the clip's best sentence or source title."""
    clean_text = re.sub(r"\s+", " ", clip_text).strip()
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(clean_text) if len(s.strip()) >= 14]

    # Prefer a short question or exclamation sentence from the clip itself
    chosen = ""
    for sent in sentences:
        if ("?" in sent or "!" in sent) and len(sent) <= 68:
            chosen = sent
            break

    if not chosen and sentences:
        first = sentences[0]
        if len(first) <= 68:
            chosen = first
        else:
            chosen = first[:65].rsplit(" ", 1)[0] + "..."

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

    fb_caption = (
        f"{title} 👇🔥\n\n"
        f"{body_part}"
        f"Drop your thoughts in the comments! Follow for more {platform} highlights.\n\n"
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
    title = _pick_hook_title(clip_text, video_title, moment, platform)
    keywords = _extract_keywords(clip_text, video_title, video_tags, limit=5)

    # Assemble clean deduplicated hashtags
    raw_tags: list[str] = []
    for kw in keywords:
        raw_tags.append(_sanitize_hashtag(kw))

    channel_tag = _sanitize_hashtag(channel)
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

    snippet = re.sub(r"\s+", " ", clip_text).strip()
    if len(snippet) > 140:
        snippet = snippet[:137].rsplit(" ", 1)[0] + "..."
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
