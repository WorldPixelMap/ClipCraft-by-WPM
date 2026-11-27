"""FastAPI backend serving Local ClipCraft by WPM UI, APIs, cookies auth, local upload, and SSE updates."""

from __future__ import annotations

import asyncio
import os
import subprocess
import uuid
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.caption_engine import is_offline_model_cached
from app.face_tracker import is_face_model_cached
from app.models import (
    AspectRatio,
    CaptionEngineMode,
    CaptionLanguage,
    ClipRequest,
    CornerPosition,
    HighlightColor,
    InspectRequest,
    InspectResponse,
    JobStatus,
    QualityPreset,
)
from app.pipeline import OUTPUT_DIR, job_manager
from app.youtube_auth import find_chromium_browser, youtube_auth_manager
from app.youtube_service import (
    COOKIES_FILE,
    find_ffmpeg_dir,
    has_cookies_file,
    inspect_video,
    validate_video_url,
)

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Local ClipCraft by WPM", version="2.0.0")


@app.middleware("http")
async def disable_static_cache(request: Request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, must-revalidate"
    return response


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")


@app.get("/")
def serve_index() -> FileResponse:
    """Serve the minimal offline-ready web interface."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.png")
@app.get("/favicon.ico")
def serve_favicon() -> FileResponse:
    """Serve project favicon.png cleanly at root."""
    fav = BASE_DIR / "favicon.png"
    if not fav.exists():
        raise HTTPException(status_code=404, detail="Favicon not found.")
    return FileResponse(fav, media_type="image/png")


@app.get("/api/health")
def health_check() -> dict:
    """Report runtime capabilities (FFmpeg, offline Whisper, face tracker, cookies status)."""
    return {
        "status": "ok",
        "brand": "Local ClipCraft by WPM",
        "ffmpeg_available": find_ffmpeg_dir() is not None,
        "offline_model_cached": is_offline_model_cached(),
        "face_model_cached": is_face_model_cached(),
        "cookies_file_loaded": has_cookies_file(),
        "browser_login_available": find_chromium_browser() is not None,
        "offline_model_name": "faster-whisper-tiny (int8 CPU)",
        "moment_engine": "Zero-AI Deterministic (Heatmap + Audio DSP + Transcript Heuristics)",
        "supported_platforms": ["YouTube", "Twitch", "Kick", "Facebook", "X / Twitter", "Vimeo", "Rumble", "1000+ yt-dlp sites"],
    }


@app.post("/api/cookies")
async def save_cookies(
    file: UploadFile | None = File(default=None),
    cookies_text: str = Form(default=""),
) -> dict:
    """Save a Netscape-format cookies.txt file locally for yt-dlp authenticated YouTube access."""
    raw = ""
    if file is not None:
        raw_bytes = await file.read(2 * 1024 * 1024)
        raw = raw_bytes.decode("utf-8", errors="replace")
    elif cookies_text.strip():
        raw = cookies_text

    lines = [ln for ln in raw.splitlines() if ln.strip()]
    has_cookie_rows = any("\t" in ln and not ln.lstrip().startswith("#") for ln in lines)
    if not lines or not has_cookie_rows:
        raise HTTPException(
            status_code=400,
            detail="Invalid cookies file. Export Netscape cookies.txt (must contain tab-separated cookie rows).",
        )

    COOKIES_FILE.write_text(raw.strip() + "\n", encoding="utf-8")
    return {"cookies_file_loaded": True}


@app.delete("/api/cookies")
def delete_cookies() -> dict:
    """Remove the locally stored cookies.txt and clear isolated login profile."""
    return youtube_auth_manager.clear_session()


@app.post("/api/youtube-login/start")
def start_youtube_login() -> dict:
    """Launch an isolated Edge/Chrome sign-in popup window so the user can log into YouTube."""
    try:
        return youtube_auth_manager.start_login()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/youtube-login/status")
def get_youtube_login_status() -> dict:
    """Poll 1-click YouTube login window state."""
    return youtube_auth_manager.get_status()


@app.post("/api/youtube-login/save")
def save_youtube_login_now() -> dict:
    """Immediately capture YouTube cookies from the open sign-in window."""
    try:
        return youtube_auth_manager.save_now()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/youtube-login/cancel")
def cancel_youtube_login() -> dict:
    """Close any active YouTube sign-in popup window."""
    return youtube_auth_manager.cancel_login()


@app.post("/api/inspect", response_model=InspectResponse)
def inspect_source_video(req: InspectRequest) -> InspectResponse:
    """Quickly inspect a YouTube, Twitch, Kick, or other supported stream link."""
    try:
        info = inspect_video(req.url, cookies_browser=req.cookies_browser)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    max_clips = max(1, min(10, int(info.duration // 12) or 1))
    return InspectResponse(
        video_id=info.video_id,
        platform=info.platform,
        title=info.title,
        channel=info.channel,
        duration=info.duration,
        thumbnail=info.thumbnail,
        has_heatmap=len(info.heatmap) > 0,
        heatmap_points=len(info.heatmap),
        caption_languages=info.caption_languages,
        has_captions=len(info.caption_languages) > 0,
        recommended_max_clips=max_clips,
    )


@app.post("/api/clips", response_model=JobStatus)
def create_clip_job(req: ClipRequest) -> JobStatus:
    """Submit an asynchronous multi-platform auto-clipping job."""
    try:
        _, platform = validate_video_url(req.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return job_manager.submit_job(req, platform=platform)


@app.post("/api/clips/upload", response_model=JobStatus)
async def create_local_clip_job(
    file: UploadFile = File(...),
    num_clips: int = Form(default=3),
    highlighted_captions: bool = Form(default=True),
    caption_language: CaptionLanguage = Form(default=CaptionLanguage.AUTO),
    aspect_ratio: AspectRatio = Form(default=AspectRatio.VERTICAL_CROP),
    highlight_color: HighlightColor = Form(default=HighlightColor.YELLOW),
    caption_mode: CaptionEngineMode = Form(default=CaptionEngineMode.AUTO),
    quality: QualityPreset = Form(default=QualityPreset.FAST),
    clip_duration: int = Form(default=35),
    streamer_cam_source: CornerPosition = Form(default=CornerPosition.OFF),
    streamer_cam_target: CornerPosition = Form(default=CornerPosition.TOP_RIGHT),
    watermark_text: str = Form(default=""),
    audio_boost: bool = Form(default=True),
) -> JobStatus:
    """Upload a local video file (.mp4/.mkv/.webm/.mov) and auto-clip it offline without any link."""
    orig_name = Path(file.filename or "local_video.mp4").name
    ext = Path(orig_name).suffix.lower()
    if ext not in {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi"}:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please select an MP4, MKV, WEBM, or MOV file.")

    job_id = uuid.uuid4().hex[:10]
    work_dir = OUTPUT_DIR / job_id / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    local_path = work_dir / f"local_source{ext}"

    with local_path.open("wb") as out_f:
        while chunk := await file.read(1024 * 1024):
            out_f.write(chunk)

    req = ClipRequest(
        url="https://example.com/local",
        num_clips=max(1, min(10, num_clips)),
        highlighted_captions=highlighted_captions,
        caption_language=caption_language,
        aspect_ratio=aspect_ratio,
        highlight_color=highlight_color,
        caption_mode=caption_mode,
        quality=quality,
        clip_duration=max(10, min(90, clip_duration)),
        streamer_cam_source=streamer_cam_source,
        streamer_cam_target=streamer_cam_target,
        watermark_text=watermark_text[:32],
        audio_boost=audio_boost,
    )
    title = Path(orig_name).stem or "Local Video"
    return job_manager.submit_local_file_job(job_id=job_id, video_path=local_path, title=title, req=req)


@app.post("/api/reels/caption", response_model=JobStatus)
async def caption_preedited_reel(
    file: UploadFile = File(...),
    caption_language: CaptionLanguage = Form(default=CaptionLanguage.AUTO),
    highlight_color: HighlightColor = Form(default=HighlightColor.YELLOW),
    aspect_ratio: AspectRatio = Form(default=AspectRatio.ORIGINAL),
    quality: QualityPreset = Form(default=QualityPreset.FAST),
    custom_script: str = Form(default=""),
    watermark_text: str = Form(default=""),
    audio_boost: bool = Form(default=True),
) -> JobStatus:
    """Upload a pre-edited reel and burn word-highlighted captions without slicing."""
    orig_name = Path(file.filename or "reel.mp4").name
    ext = Path(orig_name).suffix.lower()
    if ext not in {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi"}:
        raise HTTPException(status_code=400, detail="Unsupported reel format. Use MP4, MKV, WEBM, or MOV.")

    job_id = uuid.uuid4().hex[:10]
    work_dir = OUTPUT_DIR / job_id / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    local_path = work_dir / f"reel_source{ext}"

    with local_path.open("wb") as out_f:
        while chunk := await file.read(1024 * 1024):
            out_f.write(chunk)

    title = Path(orig_name).stem or "Pre-Edited Reel"
    return job_manager.submit_reel_caption_job(
        job_id=job_id,
        video_path=local_path,
        title=title,
        caption_language=caption_language,
        highlight_color=highlight_color,
        aspect_ratio=aspect_ratio,
        quality=quality,
        custom_script=custom_script,
        watermark_text=watermark_text[:32],
        audio_boost=audio_boost,
    )


@app.get("/api/jobs", response_model=list[JobStatus])
def list_recent_jobs() -> list[JobStatus]:
    """List recent persisted clipping jobs so users can reopen past exports."""
    return job_manager.list_recent_jobs(limit=8)


@app.get("/api/jobs/{job_id}", response_model=JobStatus)
def get_job_status(job_id: str) -> JobStatus:
    """Poll current status and clips for a job."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@app.get("/api/jobs/{job_id}/events")
async def stream_job_events(job_id: str) -> StreamingResponse:
    """Server-Sent Events stream pushing real-time job progress to the UI."""
    if not job_manager.get_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found.")

    async def event_generator():
        last_payload = ""
        while True:
            job = job_manager.get_job(job_id)
            if not job:
                break
            payload = job.model_dump_json()
            if payload != last_payload:
                yield f"data: {payload}\n\n"
                last_payload = payload
            if job.status in {"completed", "failed"}:
                break
            await asyncio.sleep(0.45)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/jobs/{job_id}/download-all")
def download_all_clips_zip(job_id: str) -> FileResponse:
    """Package generated MP4s, SRTs, and social_posts.txt into a disk-backed ZIP (low RAM)."""
    job_dir = OUTPUT_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job output not found.")

    zip_path = job_dir / f"clipcraft_{job_id}.zip"
    if not zip_path.exists():
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
            for fpath in sorted(job_dir.iterdir()):
                if fpath.is_file() and fpath.suffix.lower() in {".mp4", ".srt", ".txt"}:
                    zf.write(fpath, arcname=fpath.name)

    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=f"clipcraft_{job_id}.zip",
    )


@app.post("/api/jobs/{job_id}/open-folder")
def open_output_folder(job_id: str) -> dict:
    """Open the generated clips folder in Windows Explorer (local convenience)."""
    job_dir = OUTPUT_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Output folder not found.")
    if os.name == "nt":
        subprocess.Popen(["explorer", str(job_dir.resolve())])
    return {"opened": str(job_dir.resolve())}
