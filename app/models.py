"""Pydantic models and shared job state structures for Local ClipCraft by WPM."""

from __future__ import annotations

from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field


class AspectRatio(str, Enum):
    VERTICAL_CROP = "9:16"
    VERTICAL_FIT = "9:16-fit"
    LANDSCAPE = "16:9"
    ORIGINAL = "original"


class HighlightColor(str, Enum):
    YELLOW = "yellow"
    CYAN = "cyan"
    LIME = "lime"
    CORAL = "coral"


class CaptionEngineMode(str, Enum):
    AUTO = "auto"           # Zero-AI platform captions first, offline tiny model fallback if absent
    NO_AI = "no_ai"         # Strict zero-AI only (platform captions + audio-guided pacing)
    OFFLINE_AI = "offline"  # Always use inbuilt offline faster-whisper-tiny on selected clips


class CaptionLanguage(str, Enum):
    AUTO = "auto"
    ENGLISH = "en"
    HINDI = "hi"
    HINGLISH = "hinglish"


class CornerPosition(str, Enum):
    OFF = "off"
    AUTO = "auto"
    TOP_LEFT = "top-left"
    TOP_RIGHT = "top-right"
    BOTTOM_LEFT = "bottom-left"
    BOTTOM_RIGHT = "bottom-right"


class QualityPreset(str, Enum):
    FAST = "720p"
    HD = "1080p"


class InspectRequest(BaseModel):
    url: str = Field(..., min_length=5, max_length=600)
    cookies_browser: str = Field(default="", max_length=24)


class InspectResponse(BaseModel):
    video_id: str
    platform: str = "YouTube"
    title: str
    channel: str
    duration: float
    thumbnail: str | None = None
    has_heatmap: bool = False
    heatmap_points: int = 0
    caption_languages: list[str] = Field(default_factory=list)
    has_captions: bool = False
    recommended_max_clips: int = 10


class ClipRequest(BaseModel):
    url: str = Field(..., min_length=5, max_length=600)
    num_clips: int = Field(default=3, ge=1, le=10)
    highlighted_captions: bool = Field(default=True)
    caption_language: CaptionLanguage = Field(default=CaptionLanguage.AUTO)
    aspect_ratio: AspectRatio = Field(default=AspectRatio.VERTICAL_CROP)
    highlight_color: HighlightColor = Field(default=HighlightColor.YELLOW)
    caption_mode: CaptionEngineMode = Field(default=CaptionEngineMode.AUTO)
    quality: QualityPreset = Field(default=QualityPreset.FAST)
    clip_duration: int = Field(default=35, ge=10, le=90)
    streamer_cam_source: CornerPosition = Field(default=CornerPosition.OFF)
    streamer_cam_target: CornerPosition = Field(default=CornerPosition.TOP_RIGHT)
    watermark_text: str = Field(default="", max_length=32)
    audio_boost: bool = Field(default=True)
    cookies_browser: str = Field(default="", max_length=24)


class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float


class TranscriptSegment(BaseModel):
    text: str
    start: float
    end: float
    words: list[WordTimestamp] = Field(default_factory=list)
    has_word_timing: bool = False


class MomentCandidate(BaseModel):
    rank: int
    start: float
    end: float
    duration: float
    score: float
    heatmap_score: float = 0.0
    audio_score: float = 0.0
    transcript_score: float = 0.0
    reasons: list[str] = Field(default_factory=list)
    title_snippet: str = ""


class SocialSuggestion(BaseModel):
    title: str
    youtube_caption: str
    facebook_caption: str
    hashtags: list[str] = Field(default_factory=list)
    copy_all_youtube: str = ""
    copy_all_facebook: str = ""


class GeneratedClip(BaseModel):
    rank: int
    filename: str
    srt_filename: str | None = None
    thumb_filename: str | None = None
    video_url: str
    srt_url: str | None = None
    thumb_url: str | None = None
    start: float
    end: float
    duration: float
    score: float
    reasons: list[str] = Field(default_factory=list)
    caption_source: Literal[
        "youtube_word_sync",
        "youtube_audio_paced",
        "offline_whisper_tiny",
        "custom_script_paced",
        "none",
    ] = "none"
    transcript_preview: str = ""
    social: SocialSuggestion | None = None


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"] = "queued"
    stage: str = "Queued"
    progress: int = 0
    message: str = ""
    platform: str = "YouTube"
    video_title: str = ""
    video_id: str = ""
    duration: float = 0.0
    thumbnail: str | None = None
    ai_used: bool = False
    ai_note: str = ""
    clips: list[GeneratedClip] = Field(default_factory=list)
    error: str | None = None
