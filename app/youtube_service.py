"""Multi-platform video & live-stream metadata inspection, capture, downloading, and subtitle parsing."""

from __future__ import annotations

import ipaddress
import json
import math
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

import yt_dlp

from app.models import CaptionLanguage, TranscriptSegment, WordTimestamp

YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be",
    "youtube-nocookie.com",
    "www.youtube-nocookie.com",
}

PLATFORM_MAP = [
    ("twitch.tv", "Twitch"),
    ("kick.com", "Kick"),
    ("facebook.com", "Facebook"),
    ("fb.watch", "Facebook"),
    ("instagram.com", "Instagram"),
    ("tiktok.com", "TikTok"),
    ("twitter.com", "X / Twitter"),
    ("x.com", "X / Twitter"),
    ("vimeo.com", "Vimeo"),
    ("rumble.com", "Rumble"),
    ("dailymotion.com", "Dailymotion"),
    ("bilibili.com", "Bilibili"),
    ("reddit.com", "Reddit"),
    ("chzzk.naver.com", "CHZZK"),
    ("sooplive.co.kr", "SOOP"),
    ("sooplive.com", "SOOP"),
    ("afreecatv.com", "SOOP"),
    ("dlive.tv", "DLive"),
    ("douyu.com", "Douyu"),
    ("huya.com", "Huya"),
    ("live.vkplay.ru", "VK Play Live"),
    ("live.vkvideo.ru", "VK Video Live"),
    ("vkvideo.ru", "VK"),
    ("vk.com", "VK"),
    ("live.nicovideo.jp", "Niconico"),
    ("nicovideo.jp", "Niconico"),
    ("17.live", "17LIVE"),
    ("trovo.live", "Trovo"),
    ("odysee.com", "Odysee"),
    ("streamable.com", "Streamable"),
    ("mixcloud.com", "Mixcloud"),
    ("goodgame.ru", "GoodGame"),
]

DIRECT_STREAM_SCHEMES = ("rtmp://", "rtmps://", "rtsp://", "srt://")
FFMPEG_LIVE_PROTOCOLS = {
    "http",
    "https",
    "ftp",
    "ftps",
    "m3u8",
    "m3u8_native",
    "http_dash_segments",
    "rtmp",
    "rtmp_ffmpeg",
    "rtsp",
    "srt",
    "mms",
    "mmsh",
}

VTT_TIMESTAMP_RE = re.compile(
    r"(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{3})"
)
VTT_INLINE_WORD_RE = re.compile(r"<(\d{2}:\d{2}:\d{2}\.\d{3})><c>(.*?)</c>")
HTML_TAG_RE = re.compile(r"<[^>]+>")
ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff\u2060]")
COOKIES_FILE = Path(__file__).resolve().parent.parent / "cookies.txt"
ALLOWED_COOKIE_BROWSERS = {"firefox", "edge", "chrome", "brave", "opera", "vivaldi"}


def has_cookies_file() -> bool:
    """Return True when a non-empty local Netscape cookies.txt file is loaded."""
    return COOKIES_FILE.exists() and COOKIES_FILE.is_file() and COOKIES_FILE.stat().st_size > 20


def format_ytdlp_error(exc: Exception, platform: str = "YouTube") -> str:
    """Convert raw yt-dlp errors into clear, actionable guidance for the user."""
    msg = str(exc)
    lower = msg.lower()
    if "sign in to confirm" in lower or "not a bot" in lower:
        return (
            f"{platform} asked for sign-in confirmation from your current IP. "
            "Click 'Sign in to YouTube' to log into your account in one step, or use 'Local Video'."
        )
    if "could not copy" in lower and "cookie" in lower:
        return (
            "Could not read browser cookies while that browser is open/locked on Windows. "
            "Export a cookies.txt file (or use Firefox / upload a local video file)."
        )
    return msg


def find_ffmpeg_dir() -> str | None:
    """Locate the directory containing ffmpeg/ffprobe."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return str(Path(ffmpeg_path).parent)
    winget_base = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
    if winget_base.exists():
        for exe in winget_base.rglob("ffmpeg.exe"):
            return str(exe.parent)
    return None


def find_node_exe() -> str | None:
    """Locate Node.js executable for yt-dlp's YouTube JS challenge solver."""
    node_path = shutil.which("node")
    if node_path:
        return node_path
    for cand in (
        Path(r"C:\Program Files\nodejs\node.exe"),
        Path(r"C:\Program Files (x86)\nodejs\node.exe"),
    ):
        if cand.exists():
            return str(cand)
    return None


def _is_private_or_local_host(host: str) -> bool:
    """Block loopback/private network hostnames to prevent local SSRF."""
    lowered = host.lower().strip("[]")
    if lowered in {"localhost", "0.0.0.0"} or lowered.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(lowered)
        return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
    except ValueError:
        return "." not in lowered


def _is_direct_stream_url(url: str) -> bool:
    """Return True for raw stream feeds (RTMP/RTSP/SRT or direct HLS/DASH manifests)."""
    low = url.lower()
    if low.startswith(DIRECT_STREAM_SCHEMES):
        return True
    path_low = urlparse(url).path.lower()
    return path_low.endswith((".m3u8", ".mpd"))


def validate_video_url(raw_url: str) -> tuple[str, str]:
    """Validate and normalize a YouTube, Twitch, Kick, TikTok, or live streaming URL."""
    url = raw_url.strip()
    if not url:
        raise ValueError("Please enter a video or live stream URL.")
    if not url.startswith(("http://", "https://") + DIRECT_STREAM_SCHEMES):
        url = "https://" + url

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if not host or _is_private_or_local_host(host):
        raise ValueError("Please enter a valid public YouTube, Twitch, Kick, or streaming platform URL.")

    if host in YOUTUBE_HOSTS:
        video_id: str | None = None
        if host.endswith("youtu.be"):
            parts = [p for p in parsed.path.split("/") if p]
            if parts:
                video_id = parts[0]
        else:
            qs = parse_qs(parsed.query)
            if "v" in qs and qs["v"]:
                video_id = qs["v"][0]
            else:
                parts = [p for p in parsed.path.split("/") if p]
                if len(parts) >= 2 and parts[0].startswith("@") and parts[1].lower() == "live":
                    return f"https://www.youtube.com/{parts[0]}/live", "YouTube"
                if (
                    len(parts) >= 3
                    and parts[0].lower() in {"channel", "c", "user"}
                    and parts[2].lower() == "live"
                ):
                    return f"https://www.youtube.com/{parts[0].lower()}/{parts[1]}/live", "YouTube"
                if len(parts) >= 2 and parts[0] in {"shorts", "live", "embed", "v"}:
                    video_id = parts[1]

        if not video_id or not re.match(r"^[A-Za-z0-9_-]{6,20}$", video_id):
            raise ValueError("Could not extract a valid YouTube video or live channel ID from that link.")
        return f"https://www.youtube.com/watch?v={video_id}", "YouTube"

    platform = "Direct Stream" if _is_direct_stream_url(url) else "Streaming Video"
    for domain, label in PLATFORM_MAP:
        if host == domain or host.endswith("." + domain):
            platform = label
            break

    if not url.lower().startswith(DIRECT_STREAM_SCHEMES):
        if not parsed.path or parsed.path == "/":
            if not parsed.query:
                raise ValueError(f"Please paste a specific {platform} live channel, video, VOD, or clip URL.")

    return url, platform


def validate_youtube_url(raw_url: str) -> str:
    """Backward-compatible helper returning normalized URL (raises ValueError on unsupported non-YouTube hosts)."""
    clean_url, platform = validate_video_url(raw_url)
    if platform != "YouTube":
        raise ValueError("URL must be a valid YouTube link (youtube.com or youtu.be).")
    return clean_url


def _base_ydl_opts(cookies_browser: str = "") -> dict[str, Any]:
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "noprogress": True,
    }
    ffmpeg_dir = find_ffmpeg_dir()
    if ffmpeg_dir:
        opts["ffmpeg_location"] = ffmpeg_dir
    node_exe = find_node_exe()
    if node_exe:
        # yt-dlp Python API only enables Deno by default; explicitly enable Node.js for yt-dlp-ejs
        opts["js_runtimes"] = {"node": {"path": node_exe}}
    if has_cookies_file():
        opts["cookiefile"] = str(COOKIES_FILE)
    else:
        browser = cookies_browser.strip().lower()
        if browser in ALLOWED_COOKIE_BROWSERS:
            opts["cookiesfrombrowser"] = (browser,)
    return opts


def _select_caption_languages(
    info: dict[str, Any],
    preferred_language: CaptionLanguage = CaptionLanguage.AUTO,
) -> list[str]:
    """Pick preferred caption language codes strictly respecting English/Hindi/Hinglish preference."""
    manual = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}
    orig_lang = (info.get("language") or "").split("-")[0].lower()

    def _has_track(code: str) -> bool:
        return bool(code and (code in manual or code in auto))

    def _all_matching(predicate) -> list[str]:
        found: list[str] = []
        for source in (manual, auto):
            for code in source.keys():
                if predicate(code) and code not in found:
                    found.append(code)
        return found

    if preferred_language == CaptionLanguage.ENGLISH:
        priority = ["en", "en-orig", "en-IN", "en-US", "en-GB"]
        candidates = [c for c in priority if _has_track(c)]
        for c in _all_matching(lambda k: k.lower() == "en" or k.lower().startswith("en-")):
            if c not in candidates:
                candidates.append(c)
        return candidates[:6]

    if preferred_language == CaptionLanguage.HINDI:
        priority = ["hi", "hi-orig", "hi-IN"]
        candidates = [c for c in priority if _has_track(c)]
        for c in _all_matching(
            lambda k: (k.lower() == "hi" or k.lower().startswith("hi-")) and "latn" not in k.lower()
        ):
            if c not in candidates:
                candidates.append(c)
        return candidates[:6]

    if preferred_language == CaptionLanguage.HINGLISH:
        # Prefer Romanized Hindi (hi-Latn) or Devanagari Hindi (transliterated cleanly to Hinglish)
        priority = ["hi-Latn", "hi", "hi-orig", "hi-IN"]
        candidates = [c for c in priority if _has_track(c)]
        for c in _all_matching(lambda k: k.lower() == "hi" or k.lower().startswith("hi-")):
            if c not in candidates:
                candidates.append(c)
        return candidates[:6]

    priority = [orig_lang, "en", "en-orig", "en-IN", "en-US", "en-GB", "hi", "hi-orig", "hi-IN"]
    candidates = [c for c in priority if _has_track(c)]
    for source in (manual, auto):
        for lang in source.keys():
            if "-" in lang and lang not in {"en-US", "en-GB", "en-IN", "en-orig", "hi-IN", "hi-Latn", "pt-BR", "zh-Hans", "zh-Hant"}:
                continue
            if lang not in candidates:
                candidates.append(lang)
    return candidates[:8]


@dataclass
class VideoInspection:
    video_id: str
    platform: str
    title: str
    channel: str
    duration: float
    thumbnail: str | None
    heatmap: list[dict[str, float]]
    caption_languages: list[str]
    tags: list[str] = field(default_factory=list)
    channel_handle: str = ""
    channel_url: str = ""
    source_url: str = ""
    is_live: bool = False
    live_status: str = ""
    supports_dvr_from_start: bool = False


def _refine_platform_label(info: dict[str, Any], platform: str) -> str:
    """Derive a clean platform display name from yt-dlp extractor metadata when domain isn't in PLATFORM_MAP."""
    if platform not in {"Streaming Video", "Direct Stream"}:
        return platform
    ext_key = str(info.get("extractor_key") or info.get("extractor") or "").split(":")[0].strip()
    if ext_key and ext_key.lower() not in {"generic", "html5"}:
        clean = re.sub(r"Live$", "", ext_key)
        return clean or ext_key
    return platform


def _unwrap_playlist_info(info: dict[str, Any] | None) -> dict[str, Any] | None:
    """If yt-dlp returns a wrapper playlist (e.g. live channel tab), unwrap its active entry."""
    if info and info.get("_type") == "playlist" and info.get("entries"):
        for entry in info["entries"]:
            if isinstance(entry, dict):
                return entry
    return info


def _build_inspection_from_info(
    info: dict[str, Any],
    platform: str,
    preferred_language: CaptionLanguage,
) -> VideoInspection:
    info = _unwrap_playlist_info(info) or {}
    if not info:
        raise RuntimeError(f"Failed to retrieve video or stream information from {platform}.")

    live_status = str(info.get("live_status") or "").strip().lower()
    if live_status == "is_upcoming":
        raise ValueError("This scheduled live stream has not started yet.")

    source_candidate = str(info.get("webpage_url") or info.get("original_url") or info.get("url") or "")
    duration = float(info.get("duration") or 0)
    is_live = bool(info.get("is_live")) or live_status == "is_live" or (
        duration <= 0 and bool(source_candidate) and _is_direct_stream_url(source_candidate)
    )
    if not is_live and duration <= 0:
        raise ValueError("Could not determine video duration.")

    platform = _refine_platform_label(info, platform)
    formats = info.get("formats") or []
    supports_dvr = bool(is_live and any(bool(f.get("is_from_start")) for f in formats if isinstance(f, dict)))

    heatmap_raw = info.get("heatmap") or []
    heatmap: list[dict[str, float]] = []
    for pt in heatmap_raw:
        try:
            heatmap.append(
                {
                    "start_time": float(pt["start_time"]),
                    "end_time": float(pt["end_time"]),
                    "value": float(pt["value"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue

    langs = _select_caption_languages(info, preferred_language)
    tags = [str(t) for t in (info.get("tags") or []) if t][:15]
    raw_handle = str(info.get("uploader_id") or "").strip()
    channel_handle = raw_handle if raw_handle.startswith("@") else ""
    channel_url = str(info.get("channel_url") or info.get("uploader_url") or "").strip()
    source_url = str(info.get("webpage_url") or info.get("original_url") or "").strip()
    return VideoInspection(
        video_id=str(info.get("id") or "video"),
        platform=platform,
        title=str(info.get("title") or f"{platform} {'Live Stream' if is_live else 'Highlight Video'}"),
        channel=str(info.get("uploader") or info.get("channel") or info.get("creator") or platform),
        duration=duration,
        thumbnail=info.get("thumbnail"),
        heatmap=heatmap,
        caption_languages=langs,
        tags=tags,
        channel_handle=channel_handle,
        channel_url=channel_url,
        source_url=source_url,
        is_live=is_live,
        live_status=live_status or ("is_live" if is_live else ""),
        supports_dvr_from_start=supports_dvr,
    )


def inspect_video(
    url: str,
    preferred_language: CaptionLanguage = CaptionLanguage.AUTO,
    cookies_browser: str = "",
) -> VideoInspection:
    """Fetch video or live stream metadata, heatmap, tags, and available captions across platforms."""
    clean_url, platform = validate_video_url(url)
    opts = _base_ydl_opts(cookies_browser=cookies_browser)
    opts["skip_download"] = True

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(clean_url, download=False)
    except Exception as exc:
        if _is_direct_stream_url(clean_url):
            return VideoInspection(
                video_id=re.sub(r"[^A-Za-z0-9_-]", "", Path(urlparse(clean_url).path).stem) or "stream",
                platform=platform,
                title=f"{platform} Live Feed",
                channel=urlparse(clean_url).hostname or platform,
                duration=0.0,
                thumbnail=None,
                heatmap=[],
                caption_languages=[],
                source_url=clean_url,
                is_live=True,
                live_status="is_live",
                supports_dvr_from_start=False,
            )
        raise RuntimeError(format_ytdlp_error(exc, platform)) from exc

    return _build_inspection_from_info(info, platform, preferred_language)


def _parse_vtt_time(h: str | None, m: str, s: str, ms: str) -> float:
    hours = int(h) if h else 0
    return hours * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _clean_caption_text(text: str) -> str:
    text = HTML_TAG_RE.sub("", text)
    text = ZERO_WIDTH_RE.sub("", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&nbsp;", " ").replace("&#39;", "'").replace("&quot;", '"')
    return re.sub(r"\s+", " ", text).strip()


def parse_json3_subtitles(path: Path) -> list[TranscriptSegment]:
    """Parse YouTube json3 captions, deduplicating styled pen-layer overlaps and preserving word offsets."""
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_events = data.get("events") or []

    # First pass: clean & deduplicate identical cues emitted by YouTube multi-pen (fill/outline) layers
    cleaned_events: list[dict[str, Any]] = []
    for ev in raw_events:
        segs = ev.get("segs")
        if not segs or "tStartMs" not in ev:
            continue
        ev_start = float(ev["tStartMs"]) / 1000.0
        ev_dur = float(ev.get("dDurationMs") or 2000) / 1000.0

        raw_tokens: list[tuple[list[str], float]] = []
        has_offsets = False
        for seg in segs:
            token = _clean_caption_text(str(seg.get("utf8") or ""))
            if not token:
                continue
            offset_ms = seg.get("tOffsetMs")
            if offset_ms is not None and float(offset_ms) > 0:
                has_offsets = True
            w_start = ev_start + (float(offset_ms or 0) / 1000.0)
            parts = token.split()
            if parts:
                raw_tokens.append((parts, w_start))

        if not raw_tokens:
            continue
        full_text = " ".join(p for parts, _ in raw_tokens for p in parts)
        if not full_text:
            continue

        # Skip duplicate shadow/border events at the same timestamp
        if cleaned_events:
            prev = cleaned_events[-1]
            if prev["full_text"] == full_text and abs(ev_start - prev["ev_start"]) <= 0.25:
                continue

        cleaned_events.append(
            {
                "ev_start": ev_start,
                "ev_dur": ev_dur,
                "raw_tokens": raw_tokens,
                "has_offsets": has_offsets,
                "full_text": full_text,
            }
        )

    segments: list[TranscriptSegment] = []
    for idx, item in enumerate(cleaned_events):
        ev_start = item["ev_start"]
        ev_dur = item["ev_dur"]
        next_start = cleaned_events[idx + 1]["ev_start"] if idx + 1 < len(cleaned_events) else None
        ev_end = min(ev_start + ev_dur, next_start) if next_start and next_start > ev_start else ev_start + ev_dur

        words: list[WordTimestamp] = []
        if item["has_offsets"]:
            raw_tokens = item["raw_tokens"]
            for t_idx, (parts, tok_start) in enumerate(raw_tokens):
                tok_end = raw_tokens[t_idx + 1][1] if t_idx + 1 < len(raw_tokens) else ev_end
                tok_end = max(tok_end, tok_start + 0.08 * len(parts))
                span = tok_end - tok_start
                weights = [max(2, len(p)) for p in parts]
                total_wt = sum(weights)
                cur = tok_start
                for p, wt in zip(parts, weights):
                    p_dur = span * (wt / total_wt)
                    words.append(
                        WordTimestamp(
                            word=p,
                            start=round(cur, 3),
                            end=round(cur + max(0.05, p_dur), 3),
                        )
                    )
                    cur += p_dur

        segments.append(
            TranscriptSegment(
                text=item["full_text"],
                start=round(ev_start, 3),
                end=round(max(ev_end, ev_start + 0.1), 3),
                words=words,
                has_word_timing=item["has_offsets"] and len(words) > 0,
            )
        )

    return segments


def parse_vtt_subtitles(path: Path) -> list[TranscriptSegment]:
    """Parse WebVTT captions, handling YouTube rolling cues and inline <c> word timestamps."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    segments: list[TranscriptSegment] = []
    seen_texts: list[str] = []

    i = 0
    while i < len(lines):
        match = VTT_TIMESTAMP_RE.search(lines[i])
        if not match:
            i += 1
            continue

        start = _parse_vtt_time(*match.groups()[:4])
        end = _parse_vtt_time(*match.groups()[4:])
        i += 1

        cue_lines: list[str] = []
        while i < len(lines) and lines[i].strip():
            cue_lines.append(lines[i])
            i += 1

        raw_cue = " ".join(cue_lines)
        if end - start <= 0.08 and "<c>" not in raw_cue:
            continue

        tagged_lines = [ln for ln in cue_lines if VTT_INLINE_WORD_RE.search(ln)]
        if tagged_lines:
            tagged_raw = " ".join(tagged_lines)
            inline_matches = list(VTT_INLINE_WORD_RE.finditer(tagged_raw))
            words: list[WordTimestamp] = []
            prefix_end = inline_matches[0].start()
            prefix = _clean_caption_text(tagged_raw[:prefix_end])
            first_m = re.match(r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})", inline_matches[0].group(1))
            first_tag_time = _parse_vtt_time(*first_m.groups()) if first_m else start + 0.1
            if prefix:
                for p in prefix.split():
                    words.append(WordTimestamp(word=p, start=round(start, 3), end=round(max(first_tag_time, start + 0.08), 3)))

            for m_idx, m in enumerate(inline_matches):
                t_parts = m.group(1).split(":")
                sec_parts = t_parts[2].split(".")
                w_start = _parse_vtt_time(t_parts[0], t_parts[1], sec_parts[0], sec_parts[1])
                if m_idx + 1 < len(inline_matches):
                    nt = inline_matches[m_idx + 1].group(1).split(":")
                    ns = nt[2].split(".")
                    w_end = _parse_vtt_time(nt[0], nt[1], ns[0], ns[1])
                else:
                    w_end = end
                w_text = _clean_caption_text(m.group(2))
                for p in w_text.split():
                    words.append(WordTimestamp(word=p, start=round(w_start, 3), end=round(max(w_end, w_start + 0.06), 3)))

            if words:
                text = " ".join(w.word for w in words)
                seen_texts.append(text)
                segments.append(
                    TranscriptSegment(
                        text=text,
                        start=round(start, 3),
                        end=round(max(end, start + 0.1), 3),
                        words=words,
                        has_word_timing=True,
                    )
                )
            continue

        cleaned_lines = [_clean_caption_text(cl) for cl in cue_lines]
        cleaned_lines = [cl for cl in cleaned_lines if cl]
        if not cleaned_lines:
            continue

        if len(cleaned_lines) > 1 and seen_texts and cleaned_lines[0] == seen_texts[-1]:
            cleaned_lines = cleaned_lines[1:]
        text = " ".join(cleaned_lines).strip()
        if not text or (seen_texts and text == seen_texts[-1]):
            continue

        seen_texts.append(cleaned_lines[-1])
        segments.append(
            TranscriptSegment(
                text=text,
                start=round(start, 3),
                end=round(max(end, start + 0.1), 3),
                words=[],
                has_word_timing=False,
            )
        )

    return segments


def _parse_downloaded_subtitles(work_dir: Path, langs: list[str]) -> list[TranscriptSegment]:
    """Parse downloaded json3/vtt subtitle files ordered by requested language priority."""
    segments: list[TranscriptSegment] = []
    json3_files: list[Path] = []
    vtt_files: list[Path] = []
    for lang in langs:
        j_cand = work_dir / f"source.{lang}.json3"
        v_cand = work_dir / f"source.{lang}.vtt"
        if j_cand.exists():
            json3_files.append(j_cand)
        if v_cand.exists():
            vtt_files.append(v_cand)
    if not langs:
        return []
    if not json3_files:
        for j_any in sorted(work_dir.glob("source*.json3")):
            if j_any not in json3_files:
                json3_files.append(j_any)
    if not vtt_files:
        for v_any in sorted(work_dir.glob("source*.vtt")):
            if v_any not in vtt_files:
                vtt_files.append(v_any)

    if json3_files:
        segments = parse_json3_subtitles(json3_files[0])
    if (not segments or not any(s.has_word_timing for s in segments)) and vtt_files:
        vtt_segments = parse_vtt_subtitles(vtt_files[0])
        if vtt_segments and (not segments or any(s.has_word_timing for s in vtt_segments)):
            segments = vtt_segments
    return segments


def _download_subtitles_safely(
    ydl: yt_dlp.YoutubeDL,
    info: dict[str, Any],
    work_dir: Path,
    langs: list[str],
    cancel_event: threading.Event | None = None,
) -> list[TranscriptSegment]:
    """
    Download best matching subtitle track (json3/vtt) without ever aborting the video job on HTTP 429/network errors.
    """
    if not langs:
        return []

    manual_subs = info.get("subtitles") or {}
    auto_subs = info.get("automatic_captions") or {}
    video_id = str(info.get("id") or "video")

    saved_hooks = list(getattr(ydl, "_progress_hooks", []))
    saved_writesubs = ydl.params.get("writesubtitles")
    saved_writeauto = ydl.params.get("writeautomaticsub")
    saved_sublangs = ydl.params.get("subtitleslangs")
    saved_subfmt = ydl.params.get("subtitlesformat")

    try:
        ydl._progress_hooks = []
        ydl.params["writesubtitles"] = True
        ydl.params["writeautomaticsub"] = True
        ydl.params["subtitlesformat"] = "json3/vtt/best"

        for lang in langs[:2]:
            if cancel_event and cancel_event.is_set():
                raise RuntimeError("Job cancelled by user.")

            ydl.params["subtitleslangs"] = [lang]
            sub_map = ydl.process_subtitles(video_id, manual_subs, auto_subs) or {}
            sub_info = sub_map.get(lang)
            if not sub_info:
                continue

            sub_ext = str(sub_info.get("ext") or "vtt")
            dest_path = work_dir / f"source.{lang}.{sub_ext}"
            if sub_info.get("data") is not None:
                dest_path.write_text(str(sub_info["data"]), encoding="utf-8")
                segs = _parse_downloaded_subtitles(work_dir, [lang])
                if segs:
                    return segs
                continue

            sub_copy = dict(sub_info)
            if not sub_copy.get("http_headers") and info.get("http_headers"):
                sub_copy["http_headers"] = info.get("http_headers")

            hit_rate_limit = False
            for attempt in range(2):
                if cancel_event and cancel_event.is_set():
                    raise RuntimeError("Job cancelled by user.")
                try:
                    ydl.dl(str(dest_path), sub_copy, subtitle=True)
                    if dest_path.exists() and dest_path.stat().st_size > 10:
                        segs = _parse_downloaded_subtitles(work_dir, [lang])
                        if segs:
                            return segs
                    break
                except Exception as sub_exc:
                    dest_path.unlink(missing_ok=True)
                    err_low = str(sub_exc).lower()
                    if "429" in err_low or "too many requests" in err_low:
                        hit_rate_limit = True
                        if attempt == 0:
                            time.sleep(1.2)
                            continue
                    break

            if hit_rate_limit:
                # Stop hammering timedtext once YouTube returns 429; offline Whisper-tiny will take over cleanly
                break
    finally:
        ydl._progress_hooks = saved_hooks
        ydl.params["writesubtitles"] = saved_writesubs
        ydl.params["writeautomaticsub"] = saved_writeauto
        ydl.params["subtitleslangs"] = saved_sublangs
        ydl.params["subtitlesformat"] = saved_subfmt

    return _parse_downloaded_subtitles(work_dir, langs)


def fetch_captions_only(
    url: str,
    work_dir: Path,
    preferred_language: CaptionLanguage = CaptionLanguage.AUTO,
    cookies_browser: str = "",
) -> list[TranscriptSegment]:
    """Fetch only subtitles for a specific caption language (used when reusing cached source video)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    for old_sub in work_dir.glob("source*.json3"):
        old_sub.unlink(missing_ok=True)
    for old_sub in work_dir.glob("source*.vtt"):
        old_sub.unlink(missing_ok=True)

    clean_url, platform = validate_video_url(url)
    opts = _base_ydl_opts(cookies_browser=cookies_browser)
    opts.update(
        {
            "skip_download": True,
            "outtmpl": str(work_dir / "source.%(ext)s"),
            "writesubtitles": False,
            "writeautomaticsub": False,
            "subtitlesformat": "json3/vtt/best",
        }
    )
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(clean_url, download=False)
            inspection = _build_inspection_from_info(info, platform, preferred_language)
            langs = list(inspection.caption_languages)
            if not langs and preferred_language == CaptionLanguage.AUTO:
                langs = ["en", "en-US", "en-GB"]
            segs = _download_subtitles_safely(ydl, info, work_dir, langs) if langs else []
            if not segs and preferred_language != CaptionLanguage.AUTO:
                fallback_langs = [c for c in _select_caption_languages(info, CaptionLanguage.AUTO) if c not in langs]
                if fallback_langs:
                    segs = _download_subtitles_safely(ydl, info, work_dir, fallback_langs)
            return segs
    except Exception:
        return []


def _exe_in_ffmpeg_dir(name: str) -> str:
    ffdir = find_ffmpeg_dir()
    if ffdir:
        cand = Path(ffdir) / f"{name}.exe"
        if cand.exists():
            return str(cand)
    return name


def _probe_stream_starts(path: Path) -> dict[str, float]:
    """Return initial PTS start_time per stream type ('video', 'audio') via ffprobe."""
    cmd = [
        _exe_in_ffmpeg_dir("ffprobe"),
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,start_time",
        "-of",
        "csv=p=0",
        str(path),
    ]
    starts: dict[str, float] = {}
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15, check=False)
        for line in (res.stdout or "").splitlines():
            parts = line.strip().split(",")
            if len(parts) == 2 and parts[1] not in {"", "N/A"}:
                try:
                    val = float(parts[1])
                    if math.isfinite(val) and parts[0] not in starts:
                        starts[parts[0]] = val
                except ValueError:
                    pass
    except Exception:
        pass
    return starts


def _find_first_keyframe_after(path: Path, min_pts: float) -> float | None:
    """Find the first video keyframe PTS >= min_pts within the opening 12s for keyframe-aligned copy trimming."""
    cmd = [
        _exe_in_ffmpeg_dir("ffprobe"),
        "-v",
        "error",
        "-skip_frame",
        "nokey",
        "-select_streams",
        "v:0",
        "-read_intervals",
        f"%+{max(10.0, min_pts + 8.0):.1f}",
        "-show_entries",
        "frame=pts_time",
        "-of",
        "csv=p=0",
        str(path),
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15, check=False)
        for line in (res.stdout or "").splitlines():
            txt = line.strip()
            if not txt or txt == "N/A":
                continue
            try:
                pts = float(txt)
                if pts >= min_pts - 0.02:
                    return max(0.0, pts)
            except ValueError:
                continue
    except Exception:
        pass
    return None


def _probe_media_duration(path: Path) -> float:
    """Return media duration in seconds via ffprobe, or 0.0 if unreadable."""
    if not path.exists() or path.stat().st_size < 1024:
        return 0.0
    cmd = [
        _exe_in_ffmpeg_dir("ffprobe"),
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nw=1:nk=1",
        str(path),
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15, check=False)
        val = float((res.stdout or "").strip() or 0.0)
        return val if math.isfinite(val) and val > 0.0 else 0.0
    except Exception:
        return 0.0


def _parse_ffmpeg_progress_time(key: str, val: str) -> float | None:
    """Parse out_time_us / out_time_ms / out_time from ffmpeg -progress output into seconds."""
    if key in {"out_time_us", "out_time_ms"}:
        try:
            num = float(val)
            return max(0.0, num / 1_000_000.0)
        except ValueError:
            return None
    if key == "out_time":
        m = re.match(r"(\d+):(\d{2}):(\d{2}(?:\.\d+)?)", val.strip())
        if m:
            return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    return None


def _stop_ffmpeg_gracefully(proc: subprocess.Popen) -> None:
    """Ask FFmpeg to finalize output cleanly with 'q', falling back to terminate/kill."""
    if proc.poll() is not None:
        return
    try:
        if proc.stdin:
            proc.stdin.write("q\n")
            proc.stdin.flush()
        proc.wait(timeout=3.0)
        return
    except Exception:
        pass
    if proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=2.5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def _capture_live_stream_ffmpeg(
    selected_formats: list[dict[str, Any]],
    info: dict[str, Any],
    work_dir: Path,
    platform: str,
    capture_seconds: int,
    cookiejar: Any = None,
    progress_cb: Callable[[int, str], None] | None = None,
    cancel_event: threading.Event | None = None,
    proc_cb: Callable[[subprocess.Popen | None], None] | None = None,
) -> Path:
    """Capture a bounded live window via FFmpeg directly from resolved stream URLs across platforms."""
    ffmpeg_exe = _exe_in_ffmpeg_dir("ffmpeg")
    raw_mkv = work_dir / "live_capture.mkv"
    cmd: list[str] = [ffmpeg_exe, "-y", "-hide_banner", "-loglevel", "warning"]

    for fmt in selected_formats:
        f_url = str(fmt.get("url") or "")
        if not f_url:
            raise RuntimeError(f"Missing live stream URL for {platform}.")
        proto = str(fmt.get("protocol") or "").lower()
        is_http = f_url.startswith(("http://", "https://"))

        if is_http and cookiejar is not None:
            try:
                cookies = cookiejar.get_cookies_for_url(f_url)
            except Exception:
                cookies = []
            if cookies:
                cookie_str = "".join(
                    f"{c.name}={c.value}; path={c.path}; domain={c.domain};\r\n" for c in cookies
                )
                cmd.extend(["-cookies", cookie_str])

        http_headers = fmt.get("http_headers") or info.get("http_headers")
        if is_http and isinstance(http_headers, Mapping) and http_headers:
            hdr_str = "".join(f"{k}: {v}\r\n" for k, v in http_headers.items())
            cmd.extend(["-headers", hdr_str])

        if proto in {"http", "https", "m3u8", "m3u8_native", "http_dash_segments"}:
            cmd.extend(["-rw_timeout", "20000000"])
        if proto == "http_dash_segments":
            cmd.append("-re")
        elif proto in {"rtmp", "rtmp_ffmpeg"}:
            rtmp_map = (
                ("player_url", "-rtmp_swfverify"),
                ("page_url", "-rtmp_pageurl"),
                ("app", "-rtmp_app"),
                ("play_path", "-rtmp_playpath"),
                ("tc_url", "-rtmp_tcurl"),
                ("flash_version", "-rtmp_flashver"),
            )
            for key, flag in rtmp_map:
                if fmt.get(key) is not None:
                    cmd.extend([flag, str(fmt[key])])
            if fmt.get("rtmp_live"):
                cmd.extend(["-rtmp_live", "live"])
            conn = fmt.get("rtmp_conn")
            if isinstance(conn, list):
                for entry in conn:
                    cmd.extend(["-rtmp_conn", str(entry)])
            elif isinstance(conn, str):
                cmd.extend(["-rtmp_conn", conn])
        elif proto == "rtsp" or f_url.lower().startswith("rtsp://"):
            cmd.extend(["-rtsp_transport", "tcp"])

        cmd.extend(["-i", f_url])

    if len(selected_formats) == 1:
        cmd.extend(["-map", "0:v:0?", "-map", "0:a:0?"])
    else:
        for i, fmt in enumerate(selected_formats):
            stream_num = fmt.get("manifest_stream_number", 0)
            cmd.extend(["-map", f"{i}:{stream_num}"])

    cmd.extend(
        [
            "-t",
            str(capture_seconds),
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            "-progress",
            "pipe:1",
            "-nostats",
            str(raw_mkv),
        ]
    )

    if progress_cb:
        progress_cb(0, f"Connecting to {platform} live stream ({capture_seconds}s window)...")

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc_cb:
        proc_cb(proc)

    state_lock = threading.Lock()
    prog_state: dict[str, float] = {"sec": 0.0, "last_wall": time.monotonic()}
    stderr_lines: list[str] = []

    def _read_stdout() -> None:
        if not proc.stdout:
            return
        for raw_line in proc.stdout:
            line = raw_line.strip()
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            sec = _parse_ffmpeg_progress_time(k.strip(), v.strip())
            if sec is not None:
                with state_lock:
                    if sec > prog_state["sec"]:
                        prog_state["sec"] = sec
                        prog_state["last_wall"] = time.monotonic()

    def _read_stderr() -> None:
        if not proc.stderr:
            return
        for raw_line in proc.stderr:
            txt = raw_line.strip()
            if txt:
                stderr_lines.append(txt)
                if len(stderr_lines) > 20:
                    del stderr_lines[0]

    t_out = threading.Thread(target=_read_stdout, daemon=True)
    t_err = threading.Thread(target=_read_stderr, daemon=True)
    t_out.start()
    t_err.start()

    start_wall = time.monotonic()
    last_reported = -1
    max_wall = float(capture_seconds) * 2.5 + 120.0
    try:
        while proc.poll() is None:
            if cancel_event and cancel_event.is_set():
                _stop_ffmpeg_gracefully(proc)
                raise RuntimeError("Job cancelled by user.")

            with state_lock:
                cur_sec = prog_state["sec"]
                last_wall = prog_state["last_wall"]

            elapsed_disp = min(capture_seconds, int(cur_sec))
            pct = min(98, int((cur_sec / max(1.0, float(capture_seconds))) * 100))
            if progress_cb and (elapsed_disp != last_reported or pct >= 98):
                last_reported = elapsed_disp
                progress_cb(pct, f"Capturing {platform} live stream ({elapsed_disp}s / {capture_seconds}s)...")

            now = time.monotonic()
            if cur_sec <= 0.1 and now - start_wall > 45.0:
                _stop_ffmpeg_gracefully(proc)
                raise RuntimeError(f"{platform} live stream timed out before delivering video frames.")
            if cur_sec >= 3.0 and now - last_wall > 30.0:
                _stop_ffmpeg_gracefully(proc)
                break
            if now - start_wall > max_wall:
                _stop_ffmpeg_gracefully(proc)
                break

            time.sleep(0.25)
    finally:
        if proc_cb:
            proc_cb(None)
        if proc.poll() is None:
            _stop_ffmpeg_gracefully(proc)
        for pipe in (proc.stdin, proc.stdout, proc.stderr):
            if pipe is not None:
                try:
                    pipe.close()
                except Exception:
                    pass
        t_out.join(timeout=1.5)
        t_err.join(timeout=1.5)

    captured_dur = _probe_media_duration(raw_mkv)
    if captured_dur < 3.0:
        err_hint = stderr_lines[-1] if stderr_lines else "stream may be offline or unavailable"
        raise RuntimeError(f"Could not capture playable live stream from {platform} ({err_hint}).")

    if progress_cb:
        progress_cb(99, f"Finalizing {int(round(captured_dur))}s live capture...")

    # Align audio & video start_time to 0.000 so downstream Whisper captions & hook cuts stay frame-locked
    align_ss: float | None = None
    starts = _probe_stream_starts(raw_mkv)
    if "video" in starts and "audio" in starts:
        v_st, a_st = starts["video"], starts["audio"]
        if abs(v_st - a_st) > 0.04 or max(v_st, a_st) > 0.05:
            target = max(v_st, a_st)
            if a_st > v_st + 0.04 and captured_dur > target + 2.5:
                kf = _find_first_keyframe_after(raw_mkv, target)
                if kf is not None and kf + 2.5 < captured_dur:
                    target = kf
            if target > 0.04 and captured_dur - target >= 2.5:
                align_ss = target

    out_mp4 = work_dir / "source.mp4"
    remux_cmd = [ffmpeg_exe, "-y", "-v", "error"]
    if align_ss is not None:
        remux_cmd.extend(["-ss", f"{align_ss:.3f}"])
    remux_cmd.extend(
        [
            "-i",
            str(raw_mkv),
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            "-movflags",
            "+faststart",
            str(out_mp4),
        ]
    )
    remux = subprocess.run(remux_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if remux.returncode == 0 and _probe_media_duration(out_mp4) >= 3.0:
        raw_mkv.unlink(missing_ok=True)
        return out_mp4

    out_mkv = work_dir / "source.mkv"
    if align_ss is not None:
        remux_mkv = subprocess.run(
            [
                ffmpeg_exe,
                "-y",
                "-v",
                "error",
                "-ss",
                f"{align_ss:.3f}",
                "-i",
                str(raw_mkv),
                "-c",
                "copy",
                "-avoid_negative_ts",
                "make_zero",
                str(out_mkv),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if remux_mkv.returncode == 0 and _probe_media_duration(out_mkv) >= 2.5:
            raw_mkv.unlink(missing_ok=True)
            return out_mkv
    raw_mkv.replace(out_mkv)
    return out_mkv


def _wrap_dvr_fragments_for_capture(
    info: dict[str, Any],
    capture_seconds: int,
    cancel_event: threading.Event | None = None,
) -> bool:
    """Bound callable YouTube live-from-start fragment generators to capture_seconds. Return True if wrapped."""
    wrapped = False
    candidates = [
        *(info.get("formats") or []),
        *(info.get("requested_formats") or []),
        info,
    ]
    seen_funcs: dict[int, Any] = {}
    for fmt in candidates:
        if not isinstance(fmt, dict):
            continue
        frags = fmt.get("fragments")
        if not callable(frags):
            continue
        fid = id(frags)
        if fid in seen_funcs:
            fmt["fragments"] = seen_funcs[fid]
            wrapped = True
            continue
        frag_dur = max(1.0, float(fmt.get("target_duration") or 5.0))
        max_frags = max(2, int(math.ceil(float(capture_seconds) / frag_dur)) + 1)

        def _bounded(ctx: dict[str, Any], orig=frags, limit=max_frags):
            count = 0
            for item in orig(ctx):
                if cancel_event and cancel_event.is_set():
                    raise RuntimeError("Job cancelled by user.")
                yield item
                count += 1
                if count >= limit:
                    break

        seen_funcs[fid] = _bounded
        fmt["fragments"] = _bounded
        wrapped = True
    return wrapped


def download_video_and_captions(
    url: str,
    work_dir: Path,
    preferred_langs: list[str] | None = None,
    max_height: int = 1080,
    progress_cb: Callable[[int, str], None] | None = None,
    preferred_language: CaptionLanguage = CaptionLanguage.AUTO,
    cookies_browser: str = "",
    inspection_cb: Callable[[VideoInspection], None] | None = None,
    cancel_event: threading.Event | None = None,
    live_capture_seconds: int = 120,
    live_from_start: bool = False,
    proc_cb: Callable[[subprocess.Popen | None], None] | None = None,
) -> tuple[Path, list[TranscriptSegment], dict[str, Any], VideoInspection]:
    """Single-fetch download (VODs or live-stream capture) + rate-limit-resilient captions."""
    work_dir.mkdir(parents=True, exist_ok=True)
    clean_url, platform = validate_video_url(url)
    capture_secs = max(20, min(3600, int(live_capture_seconds or 120)))

    def _hook(d: dict[str, Any]) -> None:
        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Job cancelled by user.")
        if not progress_cb:
            return
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes") or 0
            frag_idx = d.get("fragment_index") or 0
            if total > 0:
                pct = min(100, int(downloaded * 100 / total))
                progress_cb(pct, f"Downloading {platform} video ({pct}%)...")
            elif frag_idx > 0:
                progress_cb(min(95, frag_idx * 3), f"Capturing {platform} live DVR segment #{frag_idx}...")
        elif d.get("status") == "finished":
            progress_cb(100, "Merging video & audio streams...")

    def _build_opts(use_live_from_start: bool) -> dict[str, Any]:
        opts = _base_ydl_opts(cookies_browser=cookies_browser)
        opts.update(
            {
                "format": (
                    f"bestvideo[height<={max_height}][vcodec^=avc][ext=mp4]+bestaudio[ext=m4a]/"
                    f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/"
                    f"bestvideo[height<={max_height}]+bestaudio/"
                    f"best[height<={max_height}][ext=mp4]/"
                    f"best[height<={max_height}]/best"
                ),
                "merge_output_format": "mp4",
                "outtmpl": str(work_dir / "source.%(ext)s"),
                "writesubtitles": False,
                "writeautomaticsub": False,
                "subtitleslangs": [],
                "subtitlesformat": "json3/vtt/best",
                "progress_hooks": [_hook],
            }
        )
        if use_live_from_start:
            opts["live_from_start"] = True
        return opts

    langs: list[str] = []
    segments: list[TranscriptSegment] = []
    info: dict[str, Any] | None = None
    inspection: VideoInspection | None = None
    effective_from_start = bool(live_from_start)

    try:
        opts = _build_opts(effective_from_start)
        with yt_dlp.YoutubeDL(opts) as ydl:
            try:
                info = ydl.extract_info(clean_url, download=False)
            except Exception as extract_exc:
                err_low = str(extract_exc).lower()
                if effective_from_start and ("from the start" in err_low or "live_from_start" in err_low):
                    effective_from_start = False
                    if progress_cb:
                        progress_cb(0, f"DVR from start unavailable on {platform}; switching to live capture...")
                    with yt_dlp.YoutubeDL(_build_opts(False)) as ydl_live:
                        return _download_with_extracted(
                            ydl_live,
                            ydl_live.extract_info(clean_url, download=False),
                            clean_url=clean_url,
                            platform=platform,
                            work_dir=work_dir,
                            preferred_langs=preferred_langs,
                            preferred_language=preferred_language,
                            inspection_cb=inspection_cb,
                            progress_cb=progress_cb,
                            cancel_event=cancel_event,
                            proc_cb=proc_cb,
                            capture_secs=capture_secs,
                            effective_from_start=False,
                        )
                if _is_direct_stream_url(clean_url):
                    direct_insp = VideoInspection(
                        video_id=re.sub(r"[^A-Za-z0-9_-]", "", Path(urlparse(clean_url).path).stem) or "stream",
                        platform=platform,
                        title=f"{platform} Live Feed",
                        channel=urlparse(clean_url).hostname or platform,
                        duration=0.0,
                        thumbnail=None,
                        heatmap=[],
                        caption_languages=[],
                        source_url=clean_url,
                        is_live=True,
                        live_status="is_live",
                        supports_dvr_from_start=False,
                    )
                    if inspection_cb:
                        inspection_cb(direct_insp)
                    scheme = urlparse(clean_url).scheme.lower()
                    direct_fmt = {"url": clean_url, "protocol": scheme}
                    captured = _capture_live_stream_ffmpeg(
                        selected_formats=[direct_fmt],
                        info={"url": clean_url},
                        work_dir=work_dir,
                        platform=platform,
                        capture_seconds=capture_secs,
                        cookiejar=None,
                        progress_cb=progress_cb,
                        cancel_event=cancel_event,
                        proc_cb=proc_cb,
                    )
                    return captured, [], {"url": clean_url, "is_live": True}, direct_insp
                raise extract_exc

            return _download_with_extracted(
                ydl,
                info,
                clean_url=clean_url,
                platform=platform,
                work_dir=work_dir,
                preferred_langs=preferred_langs,
                preferred_language=preferred_language,
                inspection_cb=inspection_cb,
                progress_cb=progress_cb,
                cancel_event=cancel_event,
                proc_cb=proc_cb,
                capture_secs=capture_secs,
                effective_from_start=effective_from_start,
            )
    except Exception as exc:
        raise RuntimeError(format_ytdlp_error(exc, platform)) from exc


def _download_with_extracted(
    ydl: yt_dlp.YoutubeDL,
    info: dict[str, Any] | None,
    *,
    clean_url: str,
    platform: str,
    work_dir: Path,
    preferred_langs: list[str] | None,
    preferred_language: CaptionLanguage,
    inspection_cb: Callable[[VideoInspection], None] | None,
    progress_cb: Callable[[int, str], None] | None,
    cancel_event: threading.Event | None,
    proc_cb: Callable[[subprocess.Popen | None], None] | None,
    capture_secs: int,
    effective_from_start: bool,
) -> tuple[Path, list[TranscriptSegment], dict[str, Any], VideoInspection]:
    """Execute download or live capture from an already-extracted yt-dlp info dict."""
    info = _unwrap_playlist_info(info) or {}
    inspection = _build_inspection_from_info(info, platform, preferred_language)
    if inspection_cb:
        inspection_cb(inspection)

    langs: list[str] = []
    if preferred_langs:
        langs = [ln for ln in preferred_langs if ln]
    elif inspection.caption_languages:
        langs = list(inspection.caption_languages)
    elif preferred_language == CaptionLanguage.AUTO:
        langs = ["en", "en-US", "en-GB"]

    segments: list[TranscriptSegment] = []
    if inspection.is_live:
        selected = info.get("requested_formats") or ([info] if info.get("url") else [])
        protos = {str(fmt.get("protocol") or "").lower() for fmt in selected}
        can_ffmpeg = bool(selected) and all(fmt.get("url") for fmt in selected) and protos.issubset(FFMPEG_LIVE_PROTOCOLS)

        if can_ffmpeg:
            video_path = _capture_live_stream_ffmpeg(
                selected_formats=selected,
                info=info,
                work_dir=work_dir,
                platform=inspection.platform,
                capture_seconds=capture_secs,
                cookiejar=getattr(ydl, "cookiejar", None),
                progress_cb=progress_cb,
                cancel_event=cancel_event,
                proc_cb=proc_cb,
            )
        else:
            wrapped = _wrap_dvr_fragments_for_capture(info, capture_secs, cancel_event=cancel_event)
            if not wrapped:
                ydl.params["download_ranges"] = lambda *_a: [{"start_time": 0, "end_time": capture_secs}]
            ydl.process_ie_result(info, download=True)
            video_path = _locate_downloaded_video(work_dir)

        if effective_from_start and langs:
            segments = _download_subtitles_safely(ydl, info, work_dir, langs, cancel_event=cancel_event)
        return video_path, segments, info, inspection

    # Standard VOD / clip download: fetch video first so HTTP 429 on timedtext never aborts the video
    ydl.process_ie_result(info, download=True)
    if langs:
        segments = _download_subtitles_safely(ydl, info, work_dir, langs, cancel_event=cancel_event)
    if not segments and preferred_language != CaptionLanguage.AUTO and not preferred_langs:
        fallback_langs = [c for c in _select_caption_languages(info, CaptionLanguage.AUTO) if c not in langs]
        if fallback_langs:
            segments = _download_subtitles_safely(ydl, info, work_dir, fallback_langs, cancel_event=cancel_event)

    video_path = _locate_downloaded_video(work_dir)
    return video_path, segments, info, inspection


def _locate_downloaded_video(work_dir: Path) -> Path:
    for candidate in sorted(work_dir.glob("source.*")):
        if candidate.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}:
            if candidate.exists():
                return candidate
    raise FileNotFoundError("Downloaded video file could not be located.")
