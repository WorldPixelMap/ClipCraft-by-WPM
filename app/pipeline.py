"""Job orchestration pipeline connecting multi-platform fetch / local file, non-AI picking, captions, and FFmpeg."""

from __future__ import annotations

import shutil
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.caption_engine import (
    get_clip_words,
    parse_custom_script_or_srt,
    transcribe_clip_offline_tiny,
    write_highlighted_ass_and_srt,
)
from app.models import (
    AspectRatio,
    CaptionEngineMode,
    CaptionLanguage,
    ClipRequest,
    CornerPosition,
    GeneratedClip,
    HighlightColor,
    JobStatus,
    MomentCandidate,
    QualityPreset,
    TranscriptSegment,
)
from app.moment_detector import detect_best_moments
from app.social_suggester import suggest_social_metadata
from app.video_renderer import probe_video, render_clip, target_dimensions
from app.youtube_service import VideoInspection, download_video_and_captions

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class JobManager:
    """Manages background clipping jobs, live progress, and persisted job history."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobStatus] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)

    def get_job(self, job_id: str) -> JobStatus | None:
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

    def list_recent_jobs(self, limit: int = 8) -> list[JobStatus]:
        """Return recent completed jobs from output/*/job.json sorted newest-first."""
        found: dict[str, tuple[float, JobStatus]] = {}
        for job_file in OUTPUT_DIR.glob("*/job.json"):
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

    def submit_job(self, req: ClipRequest, platform: str = "YouTube") -> JobStatus:
        job_id = uuid.uuid4().hex[:10]
        job = JobStatus(
            job_id=job_id,
            status="queued",
            stage="Queued",
            progress=2,
            platform=platform,
            message="Starting clipping job...",
        )
        with self._lock:
            self._jobs[job_id] = job
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
        )
        with self._lock:
            self._jobs[job_id] = job
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
    ) -> JobStatus:
        job = JobStatus(
            job_id=job_id,
            status="queued",
            stage="Queued",
            progress=15,
            platform="Pre-Edited Reel",
            video_title=title,
            video_id=job_id,
            message=f"Uploaded reel '{title}', preparing highlighted captions...",
        )
        with self._lock:
            self._jobs[job_id] = job
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
        )
        return job.model_copy(deep=True)

    def _run_url_job(self, job_id: str, req: ClipRequest) -> None:
        job_dir = OUTPUT_DIR / job_id
        work_dir = job_dir / "_work"
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.update_job(
                job_id,
                status="running",
                stage="Inspecting & Downloading",
                progress=6,
                message="Fetching metadata, heatmap & video stream...",
            )

            max_h = 1080 if req.quality.value == "1080p" else 720

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
            )

            self._process_video_to_clips(
                job_id=job_id,
                job_dir=job_dir,
                work_dir=work_dir,
                video_path=video_path,
                segments=segments,
                inspection=inspection,
                req=req,
            )
        except Exception as exc:
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

    def _run_local_job(self, job_id: str, video_path: Path, title: str, req: ClipRequest) -> None:
        job_dir = OUTPUT_DIR / job_id
        work_dir = job_dir / "_work"
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
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
            self._process_video_to_clips(
                job_id=job_id,
                job_dir=job_dir,
                work_dir=work_dir,
                video_path=video_path,
                segments=[],
                inspection=inspection,
                req=req,
            )
        except Exception as exc:
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
    ) -> None:
        """Add word-highlighted captions & upload copy kit to a full pre-edited reel without slicing."""
        job_dir = OUTPUT_DIR / job_id
        work_dir = job_dir / "_work"
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.update_job(
                job_id,
                status="running",
                stage="Inspecting Pre-Edited Reel",
                progress=25,
                message="Reading reel dimensions and audio track...",
            )
            src_info = probe_video(video_path)
            dur = src_info.duration
            if dur <= 0.2:
                raise ValueError("Could not determine duration of the uploaded reel.")
            self.update_job(job_id, duration=dur)

            ai_used = False
            caption_source = "none"
            if custom_script.strip():
                self.update_job(
                    job_id,
                    stage="Syncing Custom Script (Zero AI)",
                    progress=45,
                    message="Aligning your script/SRT words to speech energy...",
                )
                from app.moment_detector import analyze_audio_signal

                audio = analyze_audio_signal(video_path, dur)
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
                from app.caption_engine import _apply_language_format

                raw_words = transcribe_clip_offline_tiny(video_path, 0.0, dur, caption_language)
                words = _apply_language_format(raw_words, caption_language)
                ai_used = True
                caption_source = "offline_whisper_tiny" if words else "none"
                ai_note = "Inbuilt offline Whisper-tiny (int8 CPU) synced word-by-word captions on your reel."

            w_out, h_out = target_dimensions(aspect_ratio, quality, src_info=src_info)
            ass_path: Path | None = None
            srt_path: Path | None = None
            full_text = ""
            preview = ""

            if words or watermark_text.strip():
                ass_path = work_dir / "reel_1.ass"
                srt_path = (job_dir / "reel_captioned.srt") if words else None
                preview = write_highlighted_ass_and_srt(
                    words=words,
                    ass_path=ass_path,
                    srt_path=srt_path,
                    video_w=w_out,
                    video_h=h_out,
                    clip_duration=dur,
                    aspect_ratio=aspect_ratio,
                    highlight_color=highlight_color,
                    caption_language=caption_language,
                    watermark_text=watermark_text,
                )
                if words:
                    full_text = " ".join(w.word.strip() for w in words)

            self.update_job(
                job_id,
                stage="Burning Highlighted Captions",
                progress=70,
                message="Rendering captioned reel & generating upload copy kit...",
            )

            out_mp4 = job_dir / "reel_captioned.mp4"
            out_thumb = job_dir / "reel_captioned.jpg"
            render_clip(
                source_video=video_path,
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
            )

            moment = MomentCandidate(
                rank=1,
                start=0.0,
                end=round(dur, 2),
                duration=round(dur, 2),
                score=100.0,
                reasons=["Pre-Edited Reel", "Word Highlight Captions"],
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
                duration=round(dur, 2),
                score=100.0,
                reasons=["Full Pre-Edited Reel", "Highlighted Captions"],
                caption_source=caption_source,
                transcript_preview=preview,
                social=social,
            )

            self.update_job(
                job_id,
                status="completed",
                stage="Completed",
                progress=100,
                message="Your captioned reel & upload copy kit are ready!",
                ai_used=ai_used,
                ai_note=ai_note,
                clips=[clip_obj],
            )
            final_job = self.get_job(job_id)
            if final_job:
                (job_dir / "job.json").write_text(final_job.model_dump_json(indent=2), encoding="utf-8")
        except Exception as exc:
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
    ) -> None:
        """Shared moment selection, caption generation, face-tracked rendering, and upload kit builder."""
        self.update_job(
            job_id,
            stage="Picking Best Moments (No AI)",
            progress=44,
            message="Analyzing audio energy, heatmap peaks & transcript hooks...",
        )
        src_info = probe_video(video_path)
        true_duration = src_info.duration if src_info.duration > 0 else inspection.duration
        if true_duration <= 0:
            raise ValueError("Could not read video duration from file.")
        self.update_job(job_id, duration=true_duration)

        moments, audio_analysis = detect_best_moments(
            video_path=video_path,
            duration=true_duration,
            num_clips=req.num_clips,
            target_duration=req.clip_duration,
            heatmap=inspection.heatmap,
            segments=segments,
        )

        total = len(moments)
        generated: list[GeneratedClip] = []
        any_ai_used = False
        w_out, h_out = target_dimensions(req.aspect_ratio, req.quality)
        social_dump_lines: list[str] = [
            "Local ClipCraft by WPM — Social Upload Kit",
            f"Source ({inspection.platform}): {inspection.title} | Channel: {inspection.channel}",
            "=" * 64,
            "",
        ]

        for idx, m in enumerate(moments, start=1):
            base_pct = 50 + int(((idx - 1) / max(1, total)) * 46)
            self.update_job(
                job_id,
                stage=f"Generating Clip {idx}/{total}",
                progress=base_pct,
                message=f"Tracking faces, syncing captions & rendering clip {idx} of {total}...",
            )

            ass_path: Path | None = None
            srt_path: Path | None = None
            caption_source = "none"
            full_clip_text = m.title_snippet
            preview = m.title_snippet
            words = []

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
                )
                if ai_flag:
                    any_ai_used = True
                if words:
                    full_clip_text = " ".join(w.word.strip() for w in words)

            if words or req.watermark_text.strip():
                ass_path = work_dir / f"clip_{idx}.ass"
                srt_path = (job_dir / f"clip_{idx}.srt") if words else None
                rendered_preview = write_highlighted_ass_and_srt(
                    words=words,
                    ass_path=ass_path,
                    srt_path=srt_path,
                    video_w=w_out,
                    video_h=h_out,
                    clip_duration=m.duration,
                    aspect_ratio=req.aspect_ratio,
                    highlight_color=req.highlight_color,
                    caption_language=req.caption_language,
                    watermark_text=req.watermark_text,
                )
                if words:
                    preview = rendered_preview
            elif segments:
                seg_slice = [s.text for s in segments if s.end > m.start and s.start < m.end]
                if seg_slice:
                    full_clip_text = " ".join(seg_slice)

            social = suggest_social_metadata(
                clip_text=full_clip_text,
                video_title=inspection.title,
                channel=inspection.channel,
                platform=inspection.platform,
                moment=m,
                video_tags=inspection.tags,
                caption_language=req.caption_language,
            )

            social_dump_lines.extend(
                [
                    f"--- CLIP #{idx} ({m.start}s - {m.end}s) | Score {m.score}/100 ---",
                    f"YOUTUBE SHORTS:\n{social.copy_all_youtube}",
                    "",
                    f"FACEBOOK REELS:\n{social.copy_all_facebook}",
                    "",
                ]
            )

            out_mp4 = job_dir / f"clip_{idx}.mp4"
            out_thumb = job_dir / f"clip_{idx}.jpg"
            render_clip(
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
                audio_boost=req.audio_boost,
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
                    duration=m.duration,
                    score=m.score,
                    reasons=m.reasons,
                    caption_source=caption_source,
                    transcript_preview=preview,
                    social=social,
                )
            )

        if not req.highlighted_captions:
            ai_note = "Zero AI used (captions disabled; moments picked via deterministic signals)."
        elif any_ai_used:
            ai_note = (
                "Moments picked with Zero AI; inbuilt offline Whisper-tiny (int8 CPU) "
                "generated word-level captions."
            )
        elif req.caption_mode == CaptionEngineMode.NO_AI:
            ai_note = f"100% Zero AI (moments & captions built from {inspection.platform} + audio signals)."
        else:
            ai_note = f"100% Zero AI ({inspection.platform} captions + audio/heatmap signals — offline model not needed)."

        (job_dir / "social_posts.txt").write_text("\n".join(social_dump_lines), encoding="utf-8")

        self.update_job(
            job_id,
            status="completed",
            stage="Completed",
            progress=100,
            message=f"Generated {len(generated)} clips ready to preview, copy social posts & download!",
            ai_used=any_ai_used,
            ai_note=ai_note,
            clips=generated,
        )

        final_job = self.get_job(job_id)
        if final_job:
            (job_dir / "job.json").write_text(final_job.model_dump_json(indent=2), encoding="utf-8")


job_manager = JobManager()
