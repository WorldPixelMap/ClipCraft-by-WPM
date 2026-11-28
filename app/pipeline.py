"""Job orchestration pipeline: multi-platform fetch / local file, source caching, zero-AI & optional Ollama picking, captions, FFmpeg & single-clip edits."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.caption_engine import (
    _apply_language_format,
    _words_match_requested_language,
    align_edited_text_to_words,
    get_clip_words,
    parse_custom_script_or_srt,
    remap_words_after_jump_cuts,
    transcribe_clip_offline_tiny,
    translate_words_to_hindi,
    write_highlighted_ass_and_srt,
)
from app.models import (
    AspectRatio,
    CaptionEngineMode,
    CaptionLanguage,
    CaptionPosition,
    CaptionStyle,
    ClipRequest,
    ClipUpdateRequest,
    CornerPosition,
    EncoderMode,
    GeneratedClip,
    HighlightColor,
    JobStatus,
    MomentCandidate,
    QualityPreset,
    SplitCamPosition,
    TranscriptSegment,
    WordTimestamp,
)
from app.moment_detector import (
    analyze_audio_signal,
    compute_speech_keep_intervals,
    detect_best_moments,
    find_clip_hook_interval,
)
from app.ollama_service import enhance_social_hook_with_ollama, format_ai_label
from app.social_suggester import (
    _normalize_hook_key,
    build_platform_captions,
    generate_best_hook_banner,
    suggest_social_metadata,
)
from app.video_renderer import probe_video, render_clip, target_dimensions
from app.youtube_service import (
    VideoInspection,
    download_video_and_captions,
    fetch_captions_only,
    validate_video_url,
)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
SOURCE_CACHE_DIR = OUTPUT_DIR / "_source_cache"
MUSIC_CACHE_DIR = OUTPUT_DIR / "_music_cache"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
MUSIC_CACHE_DIR.mkdir(parents=True, exist_ok=True)

SAFE_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{4,24}$")
MAX_CACHED_SOURCES = 3


class _JobCancelled(Exception):
    """Internal signal raised when a user cancels an active job."""


def resolve_bg_music(bg_music_id: str) -> Path | None:
    """Safely resolve an uploaded background music file inside MUSIC_CACHE_DIR."""
    clean = (bg_music_id or "").strip()
    if not clean or Path(clean).name != clean or not re.fullmatch(r"[A-Za-z0-9_.-]{4,64}", clean):
        return None
    cand = MUSIC_CACHE_DIR / clean
    return cand if cand.exists() and cand.is_file() else None


def count_cached_sources() -> int:
    """Return number of cached source video folders currently stored locally."""
    if not SOURCE_CACHE_DIR.exists():
        return 0
    return sum(1 for d in SOURCE_CACHE_DIR.iterdir() if d.is_dir() and any(d.glob("source.*")))


def clear_source_cache() -> dict[str, int]:
    """Delete all cached source videos in _source_cache."""
    removed = 0
    if SOURCE_CACHE_DIR.exists():
        for d in list(SOURCE_CACHE_DIR.iterdir()):
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
    return {"cleared": removed}


def _source_cache_key(clean_url: str, max_height: int) -> str:
    return hashlib.sha1(f"{clean_url}|{max_height}".encode("utf-8")).hexdigest()[:14]


def _prune_source_cache(keep_dir: Path | None = None) -> None:
    """Keep at most MAX_CACHED_SOURCES directories in _source_cache (LRU by mtime)."""
    if not SOURCE_CACHE_DIR.exists():
        return
    dirs = [d for d in SOURCE_CACHE_DIR.iterdir() if d.is_dir()]
    if len(dirs) <= MAX_CACHED_SOURCES:
        return
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for old_dir in dirs[MAX_CACHED_SOURCES:]:
        if keep_dir and old_dir.resolve() == keep_dir.resolve():
            continue
        shutil.rmtree(old_dir, ignore_errors=True)


def _find_source_in_dir(folder: Path) -> Path | None:
    for cand in sorted(folder.glob("source.*")):
        if cand.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi"} and cand.exists():
            return cand
    return None


def _load_cached_source(
    clean_url: str,
    max_height: int,
    caption_language: CaptionLanguage,
    raw_url: str = "",
    cookies_browser: str = "",
    work_dir: Path | None = None,
) -> tuple[Path, list[TranscriptSegment], VideoInspection] | None:
    key = _source_cache_key(clean_url, max_height)
    cdir = SOURCE_CACHE_DIR / key
    meta_path = cdir / "meta.json"
    vid = _find_source_in_dir(cdir) if cdir.exists() else None
    if not vid or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        insp = VideoInspection(**meta["inspection"])
        lang_key = caption_language.value
        subs_file = cdir / f"subs_{lang_key}.json"
        segs: list[TranscriptSegment] = []
        if subs_file.exists():
            segs = [TranscriptSegment.model_validate(s) for s in json.loads(subs_file.read_text(encoding="utf-8"))]
        if not segs and raw_url and work_dir is not None and (insp.caption_languages or not subs_file.exists()):
            # Reuse cached video while fetching subtitles (retrying if an earlier transient 429 cached empty)
            segs = fetch_captions_only(
                raw_url,
                work_dir=work_dir,
                preferred_language=caption_language,
                cookies_browser=cookies_browser,
            )
            if segs:
                subs_file.write_text(json.dumps([s.model_dump() for s in segs]), encoding="utf-8")
        elif not segs and meta.get("caption_language") == lang_key:
            segs = [TranscriptSegment.model_validate(s) for s in meta.get("segments", [])]
        cdir.touch(exist_ok=True)
        return vid, segs, insp
    except Exception:
        return None


def _save_cached_source(
    clean_url: str,
    max_height: int,
    caption_language: CaptionLanguage,
    video_path: Path,
    segments: list[TranscriptSegment],
    inspection: VideoInspection,
) -> Path:
    key = _source_cache_key(clean_url, max_height)
    cdir = SOURCE_CACHE_DIR / key
    cdir.mkdir(parents=True, exist_ok=True)
    for old in cdir.glob("source.*"):
        old.unlink(missing_ok=True)
    dest = cdir / f"source{video_path.suffix}"
    shutil.copy2(video_path, dest)
    payload = {
        "caption_language": caption_language.value,
        "inspection": {
            "video_id": inspection.video_id,
            "platform": inspection.platform,
            "title": inspection.title,
            "channel": inspection.channel,
            "duration": inspection.duration,
            "thumbnail": inspection.thumbnail,
            "heatmap": inspection.heatmap,
            "caption_languages": inspection.caption_languages,
            "tags": inspection.tags,
            "channel_handle": inspection.channel_handle,
            "channel_url": inspection.channel_url,
            "source_url": inspection.source_url,
        },
        "segments": [s.model_dump() for s in segments],
    }
    (cdir / "meta.json").write_text(json.dumps(payload), encoding="utf-8")
    if segments or not inspection.caption_languages:
        (cdir / f"subs_{caption_language.value}.json").write_text(
            json.dumps([s.model_dump() for s in segments]), encoding="utf-8"
        )
    _prune_source_cache(keep_dir=cdir)
    return dest


def _apply_ollama_social_polish(
    social,
    clip_text: str,
    video_title: str,
    channel: str,
    platform: str,
    moment: MomentCandidate,
    caption_language: CaptionLanguage,
    ollama_model: str,
    ai_provider: str = "auto",
    channel_handle: str = "",
    channel_url: str = "",
    source_url: str = "",
):
    """Optionally refine on-screen hook, social title & caption via local Ollama/LM Studio while preserving creator credit."""
    polished = enhance_social_hook_with_ollama(
        model=ollama_model,
        clip_text=clip_text,
        video_title=video_title,
        platform=platform,
        language_hint=caption_language.value,
        provider=ai_provider,
    )
    if not polished:
        return social, ""
    title = polished["title"] or social.title
    hook_cap = polished.get("hook_caption") or ""
    ai_hook_banner = polished.get("hook_banner") or ""
    yt_caption, fb_caption, copy_yt, copy_fb = build_platform_captions(
        title=title,
        body_block=hook_cap,
        video_title=video_title,
        channel=channel,
        platform=platform,
        moment=moment,
        hashtags=social.hashtags,
        channel_handle=channel_handle,
        channel_url=channel_url,
        source_url=source_url,
    )
    social.title = title
    social.youtube_caption = yt_caption
    social.facebook_caption = fb_caption
    social.copy_all_youtube = copy_yt
    social.copy_all_facebook = copy_fb
    return social, ai_hook_banner


class JobManager:
    """Manages background clipping jobs, live queue, cancellation, persisted history, and single-clip re-edits."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobStatus] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._active_procs: dict[str, subprocess.Popen | None] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)

    def _register_job(self, job: JobStatus) -> threading.Event:
        ev = threading.Event()
        with self._lock:
            self._jobs[job.job_id] = job
            self._cancel_events[job.job_id] = ev
        return ev

    def _set_active_proc(self, job_id: str, proc: subprocess.Popen | None) -> None:
        with self._lock:
            self._active_procs[job_id] = proc

    def _check_cancelled(self, job_id: str) -> None:
        with self._lock:
            ev = self._cancel_events.get(job_id)
        if ev and ev.is_set():
            raise _JobCancelled("Job cancelled by user.")

    def get_job(self, job_id: str) -> JobStatus | None:
        if not SAFE_JOB_ID_RE.fullmatch(job_id):
            return None
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                return job.model_copy(deep=True)
        saved = OUTPUT_DIR / job_id / "job.json"
        if saved.exists():
            try:
                return JobStatus.model_validate_json(saved.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def list_active_jobs(self) -> list[JobStatus]:
        """Return currently queued or running jobs ordered oldest-first (execution order)."""
        with self._lock:
            active = [
                j.model_copy(deep=True)
                for j in self._jobs.values()
                if j.status in {"queued", "running"}
            ]
        return active

    def list_recent_jobs(self, limit: int = 12) -> list[JobStatus]:
        """Return recent completed jobs from output/*/job.json sorted newest-first."""
        found: dict[str, tuple[float, JobStatus]] = {}
        for job_file in OUTPUT_DIR.glob("*/job.json"):
            if job_file.parent.name.startswith("_"):
                continue
            try:
                status_obj = JobStatus.model_validate_json(job_file.read_text(encoding="utf-8"))
                found[status_obj.job_id] = (job_file.stat().st_mtime, status_obj)
            except Exception:
                continue

        ranked = sorted(found.values(), key=lambda pair: pair[0], reverse=True)
        return [item[1] for item in ranked[:limit]]

    def update_job(self, job_id: str, **kwargs) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            for k, v in kwargs.items():
                setattr(job, k, v)

    def cancel_job(self, job_id: str) -> JobStatus | None:
        if not SAFE_JOB_ID_RE.fullmatch(job_id):
            return None
        with self._lock:
            ev = self._cancel_events.get(job_id)
            if ev:
                ev.set()
            proc = self._active_procs.get(job_id)
            job = self._jobs.get(job_id)
            if job and job.status == "queued":
                job.status = "cancelled"
                job.stage = "Cancelled"
                job.message = "Job cancelled before start."
            snapshot = job.model_copy(deep=True) if job else None
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
        return snapshot

    def delete_job(self, job_id: str) -> bool:
        if not SAFE_JOB_ID_RE.fullmatch(job_id):
            return False
        self.cancel_job(job_id)
        with self._lock:
            self._jobs.pop(job_id, None)
            self._cancel_events.pop(job_id, None)
            self._active_procs.pop(job_id, None)
        job_dir = OUTPUT_DIR / job_id
        if job_dir.exists() and job_dir.is_dir():
            shutil.rmtree(job_dir, ignore_errors=True)
            return True
        return False

    def submit_job(self, req: ClipRequest, platform: str = "YouTube") -> JobStatus:
        job_id = uuid.uuid4().hex[:10]
        job = JobStatus(
            job_id=job_id,
            status="queued",
            stage="Queued",
            progress=2,
            platform=platform,
            message="Queued for clipping...",
            request_options=req.model_dump(mode="json"),
        )
        self._register_job(job)
        self._executor.submit(self._run_url_job, job_id, req)
        return job.model_copy(deep=True)

    def submit_local_file_job(self, job_id: str, video_path: Path, title: str, req: ClipRequest) -> JobStatus:
        job = JobStatus(
            job_id=job_id,
            status="queued",
            stage="Queued",
            progress=15,
            platform="Local File",
            video_title=title,
            video_id=job_id,
            message=f"Uploaded '{title}', queuing for moment selection...",
            request_options=req.model_dump(mode="json"),
        )
        self._register_job(job)
        self._executor.submit(self._run_local_job, job_id, video_path, title, req)
        return job.model_copy(deep=True)

    def submit_reel_caption_job(
        self,
        job_id: str,
        video_path: Path,
        title: str,
        caption_language: CaptionLanguage,
        highlight_color: HighlightColor,
        aspect_ratio: AspectRatio,
        quality: QualityPreset,
        custom_script: str = "",
        watermark_text: str = "",
        audio_boost: bool = True,
        caption_style: CaptionStyle = CaptionStyle.CLASSIC,
        caption_position: CaptionPosition = CaptionPosition.BOTTOM,
        caption_scale: float = 1.0,
        hook_banner: str = "",
        auto_hook: bool = False,
        trim_silence: bool = False,
        keep_source: bool = False,
        encoder_mode: EncoderMode = EncoderMode.AUTO,
        bg_music_id: str = "",
        music_volume: float = 0.14,
    ) -> JobStatus:
        opts = {
            "aspect_ratio": aspect_ratio.value,
            "quality": quality.value,
            "caption_language": caption_language.value,
            "highlight_color": highlight_color.value,
            "caption_style": caption_style.value,
            "caption_position": caption_position.value,
            "caption_scale": caption_scale,
            "hook_banner": hook_banner,
            "auto_hook": auto_hook,
            "trim_silence": trim_silence,
            "keep_source": keep_source,
            "encoder_mode": encoder_mode.value,
            "bg_music_id": bg_music_id,
            "music_volume": music_volume,
            "watermark_text": watermark_text,
            "audio_boost": audio_boost,
            "streamer_cam_source": CornerPosition.OFF.value,
            "streamer_cam_target": CornerPosition.TOP_RIGHT.value,
        }
        job = JobStatus(
            job_id=job_id,
            status="queued",
            stage="Queued",
            progress=15,
            platform="Pre-Edited Reel",
            video_title=title,
            video_id=job_id,
            message=f"Uploaded reel '{title}', preparing highlighted captions...",
            request_options=opts,
        )
        self._register_job(job)
        self._executor.submit(
            self._run_reel_caption_job,
            job_id,
            video_path,
            title,
            caption_language,
            highlight_color,
            aspect_ratio,
            quality,
            custom_script,
            watermark_text,
            audio_boost,
            caption_style,
            caption_position,
            caption_scale,
            hook_banner,
            auto_hook,
            trim_silence,
            keep_source,
            encoder_mode,
            bg_music_id,
            music_volume,
        )
        return job.model_copy(deep=True)

    def _run_url_job(self, job_id: str, req: ClipRequest) -> None:
        job_dir = OUTPUT_DIR / job_id
        work_dir = job_dir / "_work"
        work_dir.mkdir(parents=True, exist_ok=True)
        cancel_ev = self._cancel_events.get(job_id)

        try:
            self._check_cancelled(job_id)
            clean_url, _plat = validate_video_url(req.url)
            max_h = 1080 if req.quality.value == "1080p" else 720
            used_cache = False
            cached = (
                _load_cached_source(
                    clean_url,
                    max_h,
                    caption_language=req.caption_language,
                    raw_url=req.url,
                    cookies_browser=req.cookies_browser,
                    work_dir=work_dir,
                )
                if req.use_source_cache
                else None
            )

            if cached is not None:
                video_path, segments, inspection = cached
                used_cache = True
                self.update_job(
                    job_id,
                    status="running",
                    platform=inspection.platform,
                    video_title=inspection.title,
                    video_id=inspection.video_id,
                    duration=inspection.duration,
                    thumbnail=inspection.thumbnail,
                    used_cache=True,
                    stage="Using Cached Source",
                    progress=38,
                    message=f"Reusing cached source for '{inspection.title}'...",
                )
            else:
                self.update_job(
                    job_id,
                    status="running",
                    stage="Inspecting & Downloading",
                    progress=6,
                    message="Fetching metadata, heatmap & video stream...",
                )

                def _on_inspect(insp: VideoInspection) -> None:
                    self.update_job(
                        job_id,
                        platform=insp.platform,
                        video_title=insp.title,
                        video_id=insp.video_id,
                        duration=insp.duration,
                        thumbnail=insp.thumbnail,
                        stage="Downloading",
                        progress=12,
                        message=f"Downloading from {insp.platform}: '{insp.title}'...",
                    )

                def _dl_progress(pct: int, msg: str) -> None:
                    mapped = 12 + int(pct * 0.28)
                    self.update_job(job_id, progress=mapped, message=msg)

                video_path, segments, _info, inspection = download_video_and_captions(
                    req.url,
                    work_dir=work_dir,
                    max_height=max_h,
                    progress_cb=_dl_progress,
                    preferred_language=req.caption_language,
                    cookies_browser=req.cookies_browser,
                    inspection_cb=_on_inspect,
                    cancel_event=cancel_ev,
                )
                if req.use_source_cache:
                    video_path = _save_cached_source(
                        clean_url, max_h, req.caption_language, video_path, segments, inspection
                    )

            self._check_cancelled(job_id)
            preserved_path = self._preserve_job_source(job_id, job_dir, video_path, req.keep_source)
            active_video = preserved_path or video_path

            self._process_video_to_clips(
                job_id=job_id,
                job_dir=job_dir,
                work_dir=work_dir,
                video_path=active_video,
                segments=segments,
                inspection=inspection,
                req=req,
                used_cache=used_cache,
                cached_source_ref=str(active_video) if (req.keep_source or req.use_source_cache) else "",
            )
        except _JobCancelled:
            self.update_job(job_id, status="cancelled", stage="Cancelled", message="Job cancelled by user.")
        except Exception as exc:
            if cancel_ev and cancel_ev.is_set() or "cancelled by user" in str(exc).lower():
                self.update_job(job_id, status="cancelled", stage="Cancelled", message="Job cancelled by user.")
            else:
                traceback.print_exc()
                self.update_job(
                    job_id,
                    status="failed",
                    stage="Failed",
                    error=str(exc) or "Unexpected clipping error.",
                    message=f"Error: {exc}",
                )
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def _preserve_job_source(
        self,
        job_id: str,
        job_dir: Path,
        video_path: Path,
        keep_source: bool,
    ) -> Path | None:
        if not keep_source or not video_path.exists():
            return None
        dest = job_dir / f"source{video_path.suffix}"
        if video_path.resolve() != dest.resolve():
            shutil.copy2(video_path, dest)
        self.update_job(
            job_id,
            source_kept=True,
            source_video_name=dest.name,
            source_video_url=f"/output/{job_id}/{dest.name}",
        )
        return dest

    def _run_local_job(self, job_id: str, video_path: Path, title: str, req: ClipRequest) -> None:
        job_dir = OUTPUT_DIR / job_id
        work_dir = job_dir / "_work"
        work_dir.mkdir(parents=True, exist_ok=True)
        cancel_ev = self._cancel_events.get(job_id)
        try:
            self._check_cancelled(job_id)
            self.update_job(job_id, status="running", stage="Analyzing Local Video", progress=35)
            src_info = probe_video(video_path)
            inspection = VideoInspection(
                video_id=job_id,
                platform="Local File",
                title=title,
                channel="Local Upload",
                duration=src_info.duration,
                thumbnail=None,
                heatmap=[],
                caption_languages=[],
                tags=[],
            )
            preserved = self._preserve_job_source(job_id, job_dir, video_path, req.keep_source)
            active_video = preserved or video_path
            if not preserved and req.use_source_cache:
                cdir = SOURCE_CACHE_DIR / f"local_{job_id}"
                cdir.mkdir(parents=True, exist_ok=True)
                cached_local = cdir / f"source{video_path.suffix}"
                shutil.copy2(video_path, cached_local)
                active_video = cached_local
                _prune_source_cache(keep_dir=cdir)

            self._process_video_to_clips(
                job_id=job_id,
                job_dir=job_dir,
                work_dir=work_dir,
                video_path=active_video,
                segments=[],
                inspection=inspection,
                req=req,
                used_cache=False,
                cached_source_ref=str(active_video) if (req.keep_source or req.use_source_cache) else "",
            )
        except _JobCancelled:
            self.update_job(job_id, status="cancelled", stage="Cancelled", message="Job cancelled by user.")
        except Exception as exc:
            if cancel_ev and cancel_ev.is_set() or "cancelled by user" in str(exc).lower():
                self.update_job(job_id, status="cancelled", stage="Cancelled", message="Job cancelled by user.")
            else:
                traceback.print_exc()
                self.update_job(
                    job_id,
                    status="failed",
                    stage="Failed",
                    error=str(exc) or "Unexpected clipping error.",
                    message=f"Error: {exc}",
                )
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def _run_reel_caption_job(
        self,
        job_id: str,
        video_path: Path,
        title: str,
        caption_language: CaptionLanguage,
        highlight_color: HighlightColor,
        aspect_ratio: AspectRatio,
        quality: QualityPreset,
        custom_script: str,
        watermark_text: str,
        audio_boost: bool,
        caption_style: CaptionStyle,
        caption_position: CaptionPosition,
        caption_scale: float,
        hook_banner: str,
        auto_hook: bool,
        trim_silence: bool,
        keep_source: bool,
        encoder_mode: EncoderMode,
        bg_music_id: str,
        music_volume: float,
    ) -> None:
        """Add word-highlighted captions & upload copy kit to a full pre-edited reel without slicing."""
        job_dir = OUTPUT_DIR / job_id
        work_dir = job_dir / "_work"
        work_dir.mkdir(parents=True, exist_ok=True)
        cancel_ev = self._cancel_events.get(job_id)
        try:
            self._check_cancelled(job_id)
            self.update_job(
                job_id,
                status="running",
                stage="Inspecting Pre-Edited Reel",
                progress=25,
                message="Reading reel dimensions and audio track...",
            )
            preserved = self._preserve_job_source(job_id, job_dir, video_path, keep_source)
            active_video = preserved or video_path
            if not preserved:
                cdir = SOURCE_CACHE_DIR / f"local_{job_id}"
                cdir.mkdir(parents=True, exist_ok=True)
                cached_local = cdir / f"source{video_path.suffix}"
                shutil.copy2(video_path, cached_local)
                active_video = cached_local
                _prune_source_cache(keep_dir=cdir)

            src_info = probe_video(active_video)
            dur = src_info.duration
            if dur <= 0.2:
                raise ValueError("Could not determine duration of the uploaded reel.")
            self.update_job(job_id, duration=dur)

            audio = analyze_audio_signal(active_video, dur)
            keep_intervals: list[tuple[float, float]] | None = None
            trimmed_pct = 0.0
            effective_dur = dur
            if trim_silence:
                keep_intervals, trimmed_pct = compute_speech_keep_intervals(audio, 0.0, dur)
                effective_dur = max(0.5, sum(b - a for a, b in keep_intervals))

            ai_used = False
            caption_source = "none"
            if custom_script.strip():
                self.update_job(
                    job_id,
                    stage="Syncing Custom Script (Zero AI)",
                    progress=45,
                    message="Aligning your script/SRT words to speech energy...",
                )
                words = parse_custom_script_or_srt(
                    custom_script, dur, audio=audio, caption_language=caption_language
                )
                caption_source = "custom_script_paced" if words else "none"
                ai_note = "100% Zero AI (custom script/SRT synced via audio energy pacing)."
            else:
                self.update_job(
                    job_id,
                    stage="Transcribing Offline (Whisper-Tiny)",
                    progress=45,
                    message="Generating word timestamps using inbuilt offline Whisper-tiny...",
                )
                raw_words = transcribe_clip_offline_tiny(active_video, 0.0, dur, caption_language)
                if (
                    raw_words
                    and caption_language in (CaptionLanguage.HINDI, CaptionLanguage.HINGLISH)
                    and not _words_match_requested_language(raw_words, CaptionLanguage.HINDI)
                ):
                    translated_reel = translate_words_to_hindi(raw_words, audio=audio)
                    if translated_reel:
                        raw_words = translated_reel
                words = _apply_language_format(raw_words, caption_language)
                ai_used = True
                caption_source = "offline_whisper_tiny" if words else "none"
                ai_note = "Inbuilt offline Whisper-tiny (int8 CPU) synced word-by-word captions on your reel."

            self._check_cancelled(job_id)
            if trim_silence and keep_intervals and trimmed_pct > 0.0 and words:
                words = remap_words_after_jump_cuts(words, keep_intervals)

            w_out, h_out = target_dimensions(aspect_ratio, quality, src_info=src_info)
            ass_path: Path | None = None
            srt_path: Path | None = None
            full_text = " ".join(w.word.strip() for w in words) if words else ""
            preview = ""

            effective_hook_banner = hook_banner.strip()
            if not effective_hook_banner and auto_hook:
                reel_moment = MomentCandidate(
                    rank=1,
                    start=0.0,
                    end=round(dur, 2),
                    duration=round(effective_dur, 2),
                    score=100.0,
                    reasons=["Full Pre-Edited Reel", "Highlighted Captions"],
                )
                effective_hook_banner = generate_best_hook_banner(
                    clip_text=full_text,
                    video_title=title,
                    moment=reel_moment,
                    platform="Reels / Shorts",
                    caption_language=caption_language,
                    words=words,
                )

            if words or watermark_text.strip() or effective_hook_banner:
                ass_path = work_dir / "reel_1.ass"
                srt_path = (job_dir / "reel_captioned.srt") if words else None
                preview = write_highlighted_ass_and_srt(
                    words=words,
                    ass_path=ass_path,
                    srt_path=srt_path,
                    video_w=w_out,
                    video_h=h_out,
                    clip_duration=effective_dur,
                    aspect_ratio=aspect_ratio,
                    highlight_color=highlight_color,
                    caption_language=caption_language,
                    watermark_text=watermark_text,
                    caption_style=caption_style,
                    caption_position=caption_position,
                    caption_scale=caption_scale,
                    hook_banner=effective_hook_banner,
                )

            self.update_job(
                job_id,
                stage="Burning Highlighted Captions",
                progress=70,
                message="Rendering captioned reel & generating upload copy kit...",
            )

            out_mp4 = job_dir / "reel_captioned.mp4"
            out_thumb = job_dir / "reel_captioned.jpg"
            bg_music_path = resolve_bg_music(bg_music_id)
            enc_used = render_clip(
                source_video=active_video,
                src_info=src_info,
                output_mp4=out_mp4,
                output_thumb=out_thumb,
                start=0.0,
                end=dur,
                aspect_ratio=aspect_ratio,
                quality=quality,
                ass_path=ass_path,
                work_dir=work_dir,
                streamer_cam_source=CornerPosition.OFF,
                audio_boost=audio_boost,
                keep_intervals=keep_intervals if (trim_silence and trimmed_pct > 0.0) else None,
                encoder_mode=encoder_mode,
                bg_music_path=bg_music_path,
                music_volume=music_volume,
                cancel_event=cancel_ev,
                proc_cb=lambda p: self._set_active_proc(job_id, p),
            )

            reasons = ["Full Pre-Edited Reel", "Highlighted Captions"]
            if trimmed_pct > 0.0:
                reasons.append(f"Jump-Cut -{int(round(trimmed_pct * 100))}% Silence")

            moment = MomentCandidate(
                rank=1,
                start=0.0,
                end=round(dur, 2),
                duration=round(effective_dur, 2),
                score=100.0,
                reasons=reasons,
                title_snippet=preview,
            )
            social = suggest_social_metadata(
                clip_text=full_text,
                video_title=title,
                channel=watermark_text.strip() or "Creator",
                platform="Reels / Shorts",
                moment=moment,
                video_tags=[],
                caption_language=caption_language,
            )

            (job_dir / "social_posts.txt").write_text(
                f"Local ClipCraft by WPM — Pre-Edited Reel Upload Kit\n"
                f"Source: {title}\n"
                + "=" * 64
                + f"\n\nYOUTUBE SHORTS:\n{social.copy_all_youtube}\n\nFACEBOOK REELS:\n{social.copy_all_facebook}\n",
                encoding="utf-8",
            )

            clip_obj = GeneratedClip(
                rank=1,
                filename=out_mp4.name,
                srt_filename=srt_path.name if srt_path and srt_path.exists() else None,
                thumb_filename=out_thumb.name if out_thumb.exists() else None,
                video_url=f"/output/{job_id}/{out_mp4.name}",
                srt_url=f"/output/{job_id}/{srt_path.name}" if srt_path and srt_path.exists() else None,
                thumb_url=f"/output/{job_id}/{out_thumb.name}" if out_thumb.exists() else None,
                start=0.0,
                end=round(dur, 2),
                duration=round(effective_dur, 2),
                score=100.0,
                reasons=reasons,
                caption_source=caption_source,
                transcript_preview=preview,
                hook_banner=effective_hook_banner,
                words=words,
                trimmed_silence_pct=round(trimmed_pct * 100, 1),
                social=social,
            )

            current_job = self.get_job(job_id)
            req_opts = dict(current_job.request_options or {}) if current_job else {}
            req_opts["cached_source_path"] = str(active_video)

            self.update_job(
                job_id,
                status="completed",
                stage="Completed",
                progress=100,
                message="Your captioned reel & upload copy kit are ready!",
                ai_used=ai_used,
                ai_note=f"{ai_note} (Encoder: {enc_used})",
                encoder_used=enc_used,
                request_options=req_opts,
                clips=[clip_obj],
            )
            final_job = self.get_job(job_id)
            if final_job:
                (job_dir / "job.json").write_text(final_job.model_dump_json(indent=2), encoding="utf-8")
        except _JobCancelled:
            self.update_job(job_id, status="cancelled", stage="Cancelled", message="Job cancelled by user.")
        except Exception as exc:
            if cancel_ev and cancel_ev.is_set() or "cancelled by user" in str(exc).lower():
                self.update_job(job_id, status="cancelled", stage="Cancelled", message="Job cancelled by user.")
            else:
                traceback.print_exc()
                self.update_job(
                    job_id,
                    status="failed",
                    stage="Failed",
                    error=str(exc) or "Unexpected reel captioning error.",
                    message=f"Error: {exc}",
                )
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def _process_video_to_clips(
        self,
        job_id: str,
        job_dir: Path,
        work_dir: Path,
        video_path: Path,
        segments: list[TranscriptSegment],
        inspection: VideoInspection,
        req: ClipRequest,
        used_cache: bool = False,
        cached_source_ref: str = "",
    ) -> None:
        """Shared moment selection (or manual trim), caption generation, face-tracked rendering & upload kit builder."""
        self._check_cancelled(job_id)
        cancel_ev = self._cancel_events.get(job_id)

        self.update_job(
            job_id,
            stage="Picking Best Moments",
            progress=44,
            message="Analyzing audio energy, heatmap peaks & transcript hooks...",
        )
        src_info = probe_video(video_path)
        true_duration = src_info.duration if src_info.duration > 0 else inspection.duration
        if true_duration <= 0:
            raise ValueError("Could not read video duration from file.")
        self.update_job(job_id, duration=true_duration)

        ollama_model = req.ollama_model.strip() if req.ollama_enabled else ""
        if req.manual_start is not None or req.manual_end is not None:
            m_start = max(0.0, min(true_duration - 1.5, float(req.manual_start or 0.0)))
            m_end = min(true_duration, float(req.manual_end or (m_start + req.clip_duration)))
            if m_end <= m_start + 1.0:
                m_end = min(true_duration, m_start + min(15.0, float(req.clip_duration)))
            audio_analysis = analyze_audio_signal(video_path, true_duration)
            moments = [
                MomentCandidate(
                    rank=1,
                    start=round(m_start, 2),
                    end=round(m_end, 2),
                    duration=round(m_end - m_start, 2),
                    score=100.0,
                    reasons=["Custom Manual Range"],
                    title_snippet=" ".join(
                        s.text for s in segments if s.end > m_start and s.start < m_end
                    )[:120],
                )
            ]
        else:
            moments, audio_analysis = detect_best_moments(
                video_path=video_path,
                duration=true_duration,
                num_clips=req.num_clips,
                target_duration=req.clip_duration,
                heatmap=inspection.heatmap,
                segments=segments,
                ollama_model=ollama_model,
                ollama_topic=req.ollama_topic if req.ollama_enabled else "",
                ai_provider=req.ai_provider.value,
            )

        total = len(moments)
        generated: list[GeneratedClip] = []
        any_ai_used = False
        encoder_used = "libx264"
        w_out, h_out = target_dimensions(req.aspect_ratio, req.quality, src_info=src_info)
        bg_music_path = resolve_bg_music(req.bg_music_id)
        used_hooks: set[str] = set()

        social_dump_lines: list[str] = [
            "Local ClipCraft by WPM — Social Upload Kit",
            f"Source ({inspection.platform}): {inspection.title} | Channel: {inspection.channel}",
            "=" * 64,
            "",
        ]

        for idx, m in enumerate(moments, start=1):
            self._check_cancelled(job_id)
            base_pct = 50 + int(((idx - 1) / max(1, total)) * 46)
            self.update_job(
                job_id,
                stage=f"Generating Clip {idx}/{total}",
                progress=base_pct,
                message=f"Tracking faces, crafting best hook, syncing captions & rendering clip {idx} of {total}...",
            )

            keep_intervals: list[tuple[float, float]] | None = None
            trimmed_pct = 0.0
            effective_dur = m.duration
            if req.trim_silence:
                keep_intervals, trimmed_pct = compute_speech_keep_intervals(audio_analysis, m.start, m.end)
                effective_dur = max(0.5, sum(b - a for a, b in keep_intervals))

            prepend_hook: tuple[float, float] | None = None
            if req.auto_hook:
                prepend_hook = find_clip_hook_interval(
                    audio=audio_analysis,
                    clip_start=m.start,
                    clip_end=m.end,
                    duration=true_duration,
                    heatmap=inspection.heatmap,
                    segments=segments,
                )
            hook_dur = round(prepend_hook[1] - prepend_hook[0], 3) if prepend_hook else 0.0
            total_effective_dur = round(hook_dur + effective_dur, 2)

            ass_path: Path | None = None
            srt_path: Path | None = None
            caption_source = "none"
            full_clip_text = m.title_snippet
            preview = m.title_snippet
            words: list[WordTimestamp] = []

            if req.highlighted_captions:
                words, caption_source, ai_flag = get_clip_words(
                    video_path=video_path,
                    segments=segments,
                    clip_start=m.start,
                    clip_end=m.end,
                    audio=audio_analysis,
                    mode=req.caption_mode,
                    work_dir=work_dir,
                    rank=idx,
                    caption_language=req.caption_language,
                    ollama_model=ollama_model,
                    ai_provider=req.ai_provider.value,
                )
                if ai_flag:
                    any_ai_used = True
                if words:
                    full_clip_text = " ".join(w.word.strip() for w in words)
                    hook_words: list[WordTimestamp] = []
                    if prepend_hook is not None and hook_dur > 0.0:
                        h_rel_s = prepend_hook[0] - m.start
                        h_rel_e = prepend_hook[1] - m.start
                        for w in words:
                            if w.end <= h_rel_s or w.start >= h_rel_e:
                                continue
                            ws = max(0.0, round(w.start - h_rel_s, 3))
                            we = min(hook_dur, max(ws + 0.06, round(w.end - h_rel_s, 3)))
                            if we > ws + 0.02:
                                hook_words.append(WordTimestamp(word=w.word, start=ws, end=we))
                    main_words = (
                        remap_words_after_jump_cuts(words, keep_intervals)
                        if (req.trim_silence and keep_intervals and trimmed_pct > 0.0)
                        else words
                    )
                    if prepend_hook is not None and hook_dur > 0.0:
                        shifted_main = [
                            WordTimestamp(
                                word=w.word,
                                start=round(w.start + hook_dur, 3),
                                end=round(w.end + hook_dur, 3),
                            )
                            for w in main_words
                        ]
                        words = hook_words + shifted_main
                    else:
                        words = main_words
            elif segments:
                seg_slice = [s.text for s in segments if s.end > m.start and s.start < m.end]
                if seg_slice:
                    full_clip_text = " ".join(seg_slice)

            reasons = list(m.reasons)
            if prepend_hook is not None and hook_dur > 0.0:
                reasons.insert(0, f"Hook Teaser +{hook_dur:.1f}s")
            if trimmed_pct > 0.0:
                reasons.append(f"Jump-Cut -{int(round(trimmed_pct * 100))}% Silence")

            manual_hook = req.hook_banner.strip()
            if manual_hook:
                effective_hook_banner = manual_hook
            elif req.auto_hook:
                effective_hook_banner = generate_best_hook_banner(
                    clip_text=full_clip_text,
                    video_title=inspection.title,
                    moment=m,
                    platform=inspection.platform,
                    caption_language=req.caption_language,
                    video_tags=inspection.tags,
                    used_hooks=used_hooks,
                    words=words,
                )
            else:
                effective_hook_banner = ""

            social = suggest_social_metadata(
                clip_text=full_clip_text,
                video_title=inspection.title,
                channel=inspection.channel,
                platform=inspection.platform,
                moment=m,
                video_tags=inspection.tags,
                caption_language=req.caption_language,
                channel_handle=inspection.channel_handle,
                channel_url=inspection.channel_url,
                source_url=inspection.source_url or (req.url if "example.com/local" not in req.url else ""),
            )
            if ollama_model:
                social, ai_hook_banner = _apply_ollama_social_polish(
                    social=social,
                    clip_text=full_clip_text,
                    video_title=inspection.title,
                    channel=inspection.channel,
                    platform=inspection.platform,
                    moment=m,
                    caption_language=req.caption_language,
                    ollama_model=ollama_model,
                    ai_provider=req.ai_provider.value,
                    channel_handle=inspection.channel_handle,
                    channel_url=inspection.channel_url,
                    source_url=inspection.source_url or (req.url if "example.com/local" not in req.url else ""),
                )
                if not manual_hook and req.auto_hook and ai_hook_banner:
                    ai_key = _normalize_hook_key(ai_hook_banner)
                    if ai_key and ai_key not in used_hooks:
                        used_hooks.add(ai_key)
                        effective_hook_banner = ai_hook_banner

            if words or req.watermark_text.strip() or effective_hook_banner:
                ass_path = work_dir / f"clip_{idx}.ass"
                srt_path = (job_dir / f"clip_{idx}.srt") if words else None
                rendered_preview = write_highlighted_ass_and_srt(
                    words=words,
                    ass_path=ass_path,
                    srt_path=srt_path,
                    video_w=w_out,
                    video_h=h_out,
                    clip_duration=total_effective_dur,
                    aspect_ratio=req.aspect_ratio,
                    highlight_color=req.highlight_color,
                    caption_language=req.caption_language,
                    watermark_text=req.watermark_text,
                    caption_style=req.caption_style,
                    caption_position=req.caption_position,
                    caption_scale=req.caption_scale,
                    hook_banner=effective_hook_banner,
                )
                if words:
                    preview = rendered_preview

            hook_dump = f" | Hook: {effective_hook_banner}" if effective_hook_banner else ""
            if prepend_hook is not None:
                hook_dump += f" (Prepended {prepend_hook[0]}s-{prepend_hook[1]}s)"
            social_dump_lines.extend(
                [
                    f"--- CLIP #{idx} ({m.start}s - {m.end}s) | Score {m.score}/100{hook_dump} ---",
                    f"YOUTUBE SHORTS:\n{social.copy_all_youtube}",
                    "",
                    f"FACEBOOK REELS:\n{social.copy_all_facebook}",
                    "",
                ]
            )

            out_mp4 = job_dir / f"clip_{idx}.mp4"
            out_thumb = job_dir / f"clip_{idx}.jpg"
            encoder_used = render_clip(
                source_video=video_path,
                src_info=src_info,
                output_mp4=out_mp4,
                output_thumb=out_thumb,
                start=m.start,
                end=m.end,
                aspect_ratio=req.aspect_ratio,
                quality=req.quality,
                ass_path=ass_path,
                work_dir=work_dir,
                streamer_cam_source=req.streamer_cam_source,
                streamer_cam_target=req.streamer_cam_target,
                split_cam_position=req.split_cam_position,
                audio_boost=req.audio_boost,
                keep_intervals=keep_intervals if (req.trim_silence and trimmed_pct > 0.0) else None,
                prepend_hook=prepend_hook,
                encoder_mode=req.encoder_mode,
                bg_music_path=bg_music_path,
                music_volume=req.music_volume,
                cancel_event=cancel_ev,
                proc_cb=lambda p: self._set_active_proc(job_id, p),
            )

            generated.append(
                GeneratedClip(
                    rank=idx,
                    filename=out_mp4.name,
                    srt_filename=srt_path.name if srt_path and srt_path.exists() else None,
                    thumb_filename=out_thumb.name if out_thumb.exists() else None,
                    video_url=f"/output/{job_id}/{out_mp4.name}",
                    srt_url=f"/output/{job_id}/{srt_path.name}" if srt_path and srt_path.exists() else None,
                    thumb_url=f"/output/{job_id}/{out_thumb.name}" if out_thumb.exists() else None,
                    start=m.start,
                    end=m.end,
                    duration=total_effective_dur,
                    score=m.score,
                    reasons=reasons[:4],
                    caption_source=caption_source,
                    transcript_preview=preview,
                    hook_banner=effective_hook_banner,
                    hook_start=prepend_hook[0] if prepend_hook else None,
                    hook_end=prepend_hook[1] if prepend_hook else None,
                    words=words,
                    trimmed_silence_pct=round(trimmed_pct * 100, 1),
                    social=social,
                )
            )

        notes: list[str] = []
        if not req.highlighted_captions:
            notes.append("Zero AI used (captions disabled; moments picked via deterministic signals).")
        elif any_ai_used:
            notes.append("Moments picked deterministically; offline Whisper-tiny generated word-level captions.")
        else:
            notes.append(f"100% Zero AI ({inspection.platform} captions + audio/heatmap signals).")
        if ollama_model:
            notes.append(f"Local AI ({format_ai_label(ollama_model, req.ai_provider.value)}) enabled.")
        if used_cache:
            notes.append("Reused cached source.")
        notes.append(f"Encoder: {encoder_used}.")

        (job_dir / "social_posts.txt").write_text("\n".join(social_dump_lines), encoding="utf-8")

        req_opts = req.model_dump(mode="json")
        if cached_source_ref:
            req_opts["cached_source_path"] = cached_source_ref

        self.update_job(
            job_id,
            status="completed",
            stage="Completed",
            progress=100,
            message=f"Generated {len(generated)} clips ready to preview, edit, copy social posts & download!",
            ai_used=any_ai_used,
            ai_note=" ".join(notes),
            encoder_used=encoder_used,
            request_options=req_opts,
            clips=generated,
        )

        final_job = self.get_job(job_id)
        if final_job:
            (job_dir / "job.json").write_text(final_job.model_dump_json(indent=2), encoding="utf-8")

    def update_clip(self, job_id: str, rank: int, patch: ClipUpdateRequest) -> GeneratedClip:
        """Trim and/or edit captions on a single already-generated clip and re-render just that clip."""
        job = self.get_job(job_id)
        if not job:
            raise KeyError("Job not found.")
        job_dir = OUTPUT_DIR / job_id
        target_idx = next((i for i, c in enumerate(job.clips) if c.rank == rank), -1)
        if target_idx < 0:
            raise KeyError(f"Clip #{rank} not found.")

        clip = job.clips[target_idx]
        opts = dict(job.request_options or {})

        # Resolve source video: job_dir/source.* first, else cached_source_path
        src_video = _find_source_in_dir(job_dir)
        if not src_video:
            ref = str(opts.get("cached_source_path") or "")
            if ref:
                cand = Path(ref)
                if cand.exists() and cand.is_file():
                    src_video = cand
        if not src_video:
            raise ValueError(
                "Original source video is no longer cached. Enable 'Keep source' or 'Reuse cache' before generating to trim or re-burn clips later."
            )

        src_info = probe_video(src_video)
        total_dur = src_info.duration if src_info.duration > 0 else job.duration

        new_start = clip.start if patch.start is None else max(0.0, min(total_dur - 1.0, float(patch.start)))
        new_end = clip.end if patch.end is None else min(total_dur, float(patch.end))
        if new_end <= new_start + 0.5:
            raise ValueError("End time must be at least 0.5s after start time.")
        trim_changed = (patch.start is not None and abs(new_start - clip.start) > 0.05) or (
            patch.end is not None and abs(new_end - clip.end) > 0.05
        )
        prepend_hook = (
            (clip.hook_start, clip.hook_end)
            if (not trim_changed and clip.hook_start is not None and clip.hook_end is not None and clip.hook_end > clip.hook_start + 0.8)
            else None
        )
        hook_dur = round(prepend_hook[1] - prepend_hook[0], 3) if prepend_hook else 0.0
        new_dur = round((new_end - new_start) + hook_dur, 2)

        aspect_ratio = AspectRatio(opts.get("aspect_ratio", AspectRatio.VERTICAL_CROP.value))
        quality = QualityPreset(opts.get("quality", QualityPreset.FAST.value))
        cap_lang = CaptionLanguage(opts.get("caption_language", CaptionLanguage.AUTO.value))
        hl_color = patch.highlight_color or HighlightColor(opts.get("highlight_color", HighlightColor.YELLOW.value))
        cap_style = patch.caption_style or CaptionStyle(opts.get("caption_style", CaptionStyle.CLASSIC.value))
        cap_pos = patch.caption_position or CaptionPosition(opts.get("caption_position", CaptionPosition.BOTTOM.value))
        cap_scale = float(patch.caption_scale if patch.caption_scale is not None else opts.get("caption_scale", 1.0))
        hook_banner = str(
            patch.hook_banner
            if patch.hook_banner is not None
            else (clip.hook_banner or opts.get("hook_banner", ""))
        ).strip()
        watermark = str(opts.get("watermark_text", ""))
        audio_boost = bool(opts.get("audio_boost", True))
        cam_src = CornerPosition(opts.get("streamer_cam_source", CornerPosition.OFF.value))
        cam_tgt = CornerPosition(opts.get("streamer_cam_target", CornerPosition.TOP_RIGHT.value))
        split_cam_pos = patch.split_cam_position or SplitCamPosition(
            opts.get("split_cam_position", SplitCamPosition.TOP.value)
        )
        enc_mode = EncoderMode(opts.get("encoder_mode", EncoderMode.AUTO.value))
        bg_music = resolve_bg_music(str(opts.get("bg_music_id", "")))
        music_vol = float(opts.get("music_volume", 0.14))

        words = list(clip.words or [])
        cap_source = clip.caption_source
        if patch.words is not None:
            words = list(patch.words)
            cap_source = "edited_manual"
        elif patch.caption_text is not None:
            words = align_edited_text_to_words(
                patch.caption_text,
                existing_words=words,
                clip_duration=new_dur,
                caption_language=cap_lang,
            )
            cap_source = "edited_manual" if words else "none"
        elif (patch.start is not None or patch.end is not None) and words:
            # Re-scale/clamp existing words smoothly if user adjusted clip trim range without editing text
            old_span = max(0.5, clip.end - clip.start)
            delta_s = new_start - clip.start
            shifted: list[WordTimestamp] = []
            for w in words:
                ws = round(w.start - delta_s, 3)
                we = round(w.end - delta_s, 3)
                if we > 0.0 and ws < new_dur:
                    ws_c = max(0.0, ws)
                    we_c = min(new_dur, max(ws_c + 0.05, we))
                    shifted.append(WordTimestamp(word=w.word, start=ws_c, end=we_c))
            words = shifted if shifted else align_edited_text_to_words(
                " ".join(w.word for w in words),
                existing_words=[],
                clip_duration=new_dur,
                caption_language=cap_lang,
            )

        work_dir = job_dir / f"_edit_{rank}_{uuid.uuid4().hex[:6]}"
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            w_out, h_out = target_dimensions(aspect_ratio, quality, src_info=src_info)
            out_mp4 = job_dir / clip.filename
            out_thumb = job_dir / (clip.thumb_filename or f"clip_{rank}.jpg")
            srt_name = clip.srt_filename or (f"clip_{rank}.srt" if words else None)
            srt_path = (job_dir / srt_name) if (srt_name and words) else None
            ass_path: Path | None = None
            preview = clip.transcript_preview

            if words or watermark.strip() or hook_banner.strip():
                ass_path = work_dir / f"edit_{rank}.ass"
                preview = write_highlighted_ass_and_srt(
                    words=words,
                    ass_path=ass_path,
                    srt_path=srt_path,
                    video_w=w_out,
                    video_h=h_out,
                    clip_duration=new_dur,
                    aspect_ratio=aspect_ratio,
                    highlight_color=hl_color,
                    caption_language=cap_lang,
                    watermark_text=watermark,
                    caption_style=cap_style,
                    caption_position=cap_pos,
                    caption_scale=cap_scale,
                    hook_banner=hook_banner,
                )

            render_clip(
                source_video=src_video,
                src_info=src_info,
                output_mp4=out_mp4,
                output_thumb=out_thumb,
                start=new_start,
                end=new_end,
                aspect_ratio=aspect_ratio,
                quality=quality,
                ass_path=ass_path,
                work_dir=work_dir,
                streamer_cam_source=cam_src,
                streamer_cam_target=cam_tgt,
                split_cam_position=split_cam_pos,
                audio_boost=audio_boost,
                prepend_hook=prepend_hook,
                encoder_mode=enc_mode,
                bg_music_path=bg_music,
                music_volume=music_vol,
            )

            ver = int(time.time())
            updated = clip.model_copy(
                update={
                    "start": round(new_start, 2),
                    "end": round(new_end, 2),
                    "duration": new_dur,
                    "words": words,
                    "caption_source": cap_source,
                    "transcript_preview": preview,
                    "hook_banner": hook_banner,
                    "hook_start": prepend_hook[0] if prepend_hook else None,
                    "hook_end": prepend_hook[1] if prepend_hook else None,
                    "srt_filename": srt_path.name if srt_path and srt_path.exists() else None,
                    "video_url": f"/output/{job_id}/{out_mp4.name}?v={ver}",
                    "srt_url": f"/output/{job_id}/{srt_path.name}?v={ver}" if srt_path and srt_path.exists() else None,
                    "thumb_url": f"/output/{job_id}/{out_thumb.name}?v={ver}" if out_thumb.exists() else None,
                }
            )
            job.clips[target_idx] = updated
            # Remove stale zip bundle so next Download All includes updated clip
            stale_zip = job_dir / f"clipcraft_{job_id}.zip"
            if stale_zip.exists():
                stale_zip.unlink(missing_ok=True)
            with self._lock:
                if job_id in self._jobs:
                    self._jobs[job_id] = job.model_copy(deep=True)
            (job_dir / "job.json").write_text(job.model_dump_json(indent=2), encoding="utf-8")
            return updated
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)


job_manager = JobManager()
