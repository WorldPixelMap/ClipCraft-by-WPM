"""Multi-platform video metadata inspection, downloading (YouTube, Twitch, Kick, etc.), and subtitle parsing."""

from __future__ import annotations

import ipaddress
import json
import re
import shutil
import threading
import time
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
]

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


def validate_video_url(raw_url: str) -> tuple[str, str]:
    """Validate and normalize a YouTube, Twitch, Kick, or other supported streaming URL."""
    url = raw_url.strip()
    if not url:
        raise ValueError("Please enter a video or stream URL.")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if not host or _is_private_or_local_host(host):
        raise ValueError("Please enter a valid public YouTube, Twitch, or streaming platform URL.")

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
                if len(parts) >= 2 and parts[0] in {"shorts", "live", "embed", "v"}:
                    video_id = parts[1]

        if not video_id or not re.match(r"^[A-Za-z0-9_-]{6,20}$", video_id):
            raise ValueError("Could not extract a valid YouTube video ID from that link.")
        return f"https://www.youtube.com/watch?v={video_id}", "YouTube"

    platform = "Streaming Video"
    for domain, label in PLATFORM_MAP:
        if host == domain or host.endswith("." + domain):
            platform = label
            break

    if not parsed.path or parsed.path == "/":
        if not parsed.query:
            raise ValueError(f"Please paste a specific {platform} video, VOD, or clip URL.")

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


def _build_inspection_from_info(
    info: dict[str, Any],
    platform: str,
    preferred_language: CaptionLanguage,
) -> VideoInspection:
    if not info:
        raise RuntimeError(f"Failed to retrieve video information from {platform}.")
    if info.get("is_live"):
        raise ValueError("Ongoing live streams are not supported yet. Please use a VOD, clip, or recorded video.")

    duration = float(info.get("duration") or 0)
    if duration <= 0:
        raise ValueError("Could not determine video duration.")

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
        title=str(info.get("title") or f"{platform} Highlight Video"),
        channel=str(info.get("uploader") or info.get("channel") or info.get("creator") or platform),
        duration=duration,
        thumbnail=info.get("thumbnail"),
        heatmap=heatmap,
        caption_languages=langs,
        tags=tags,
        channel_handle=channel_handle,
        channel_url=channel_url,
        source_url=source_url,
    )


def inspect_video(
    url: str,
    preferred_language: CaptionLanguage = CaptionLanguage.AUTO,
    cookies_browser: str = "",
) -> VideoInspection:
    """Fetch video metadata, heatmap, tags, and available captions from YouTube/Twitch/etc."""
    clean_url, platform = validate_video_url(url)
    opts = _base_ydl_opts(cookies_browser=cookies_browser)
    opts["skip_download"] = True

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(clean_url, download=False)
    except Exception as exc:
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
) -> tuple[Path, list[TranscriptSegment], dict[str, Any], VideoInspection]:
    """Single-fetch download of video + rate-limit-resilient captions (never fails the video on subtitle 429)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    clean_url, platform = validate_video_url(url)

    def _hook(d: dict[str, Any]) -> None:
        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Job cancelled by user.")
        if not progress_cb:
            return
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes") or 0
            if total > 0:
                pct = min(100, int(downloaded * 100 / total))
                progress_cb(pct, f"Downloading {platform} video ({pct}%)...")
        elif d.get("status") == "finished":
            progress_cb(100, "Merging video & audio streams...")

    out_template = str(work_dir / "source.%(ext)s")
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
            "outtmpl": out_template,
            "writesubtitles": False,
            "writeautomaticsub": False,
            "subtitleslangs": [],
            "subtitlesformat": "json3/vtt/best",
            "progress_hooks": [_hook],
        }
    )

    langs: list[str] = []
    segments: list[TranscriptSegment] = []
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(clean_url, download=False)
            inspection = _build_inspection_from_info(info, platform, preferred_language)
            if inspection_cb:
                inspection_cb(inspection)
            if preferred_langs:
                langs = [ln for ln in preferred_langs if ln]
            elif inspection.caption_languages:
                langs = list(inspection.caption_languages)
            elif preferred_language == CaptionLanguage.AUTO:
                langs = ["en", "en-US", "en-GB"]
            else:
                langs = []
            # Download video stream first without subtitle flags so HTTP 429 on timedtext never aborts the video
            ydl.process_ie_result(info, download=True)
            if langs:
                segments = _download_subtitles_safely(
                    ydl, info, work_dir, langs, cancel_event=cancel_event
                )
            if not segments and preferred_language != CaptionLanguage.AUTO and not preferred_langs:
                # If translated track (e.g. tlang=hi) got 429 or was missing, fetch source video's native track
                fallback_langs = [
                    c for c in _select_caption_languages(info, CaptionLanguage.AUTO) if c not in langs
                ]
                if fallback_langs:
                    segments = _download_subtitles_safely(
                        ydl, info, work_dir, fallback_langs, cancel_event=cancel_event
                    )
    except Exception as exc:
        raise RuntimeError(format_ytdlp_error(exc, platform)) from exc

    video_path: Path | None = None
    for candidate in sorted(work_dir.glob("source.*")):
        if candidate.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}:
            video_path = candidate
            break
    if not video_path or not video_path.exists():
        raise FileNotFoundError("Downloaded video file could not be located.")

    return video_path, segments, info or {}, inspection
