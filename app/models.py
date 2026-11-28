"""Pydantic models and shared job state structures for Local ClipCraft by WPM."""

from __future__ import annotations

from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field


class AspectRatio(str, Enum):
    VERTICAL_CROP = "9:16"
    VERTICAL_FIT = "9:16-fit"
    SPLIT_SCREEN = "split"
    LANDSCAPE = "16:9"
    ORIGINAL = "original"


class HighlightColor(str, Enum):
    YELLOW = "yellow"
    CYAN = "cyan"
    LIME = "lime"
    CORAL = "coral"


class CaptionStyle(str, Enum):
    CLASSIC = "classic"
    BOXED = "boxed"
    POP = "pop"
    MINIMAL = "minimal"


class CaptionPosition(str, Enum):
    BOTTOM = "bottom"
    CENTER = "center"
    TOP = "top"


class EncoderMode(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    NVENC = "nvenc"
    QSV = "qsv"
    AMF = "amf"


class SplitCamPosition(str, Enum):
    TOP = "top"
    BOTTOM = "bottom"


class LocalAiProvider(str, Enum):
    AUTO = "auto"
    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"


class CaptionEngineMode(str, Enum):
    AUTO = "auto"           # Zero-AI platform captions first, offline tiny model fallback if absent
    NO_AI = "no_ai"         # Strict zero-AI only (platform captions + audio-guided pacing)
    OFFLINE_AI = "offline"  # Always use inbuilt offline faster-whisper-tiny on selected clips


class CaptionLanguage(str, Enum):
    AUTO = "auto"
    ENGLISH = "en"
    HINDI = "hi"
    HINGLISH = "hinglish"


class SpeechModelPreset(str, Enum):
    AUTO = "auto"
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"


class CornerPosition(str, Enum):
    OFF = "off"
    AUTO = "auto"
    TOP_LEFT = "top-left"
    TOP_RIGHT = "top-right"
    BOTTOM_LEFT = "bottom-left"
    BOTTOM_RIGHT = "bottom-right"
    TOP_CENTER = "top-center"
    BOTTOM_CENTER = "bottom-center"
    MID_LEFT = "mid-left"
    MID_RIGHT = "mid-right"
    CENTER = "center"


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
    is_live: bool = False
    live_status: str = ""
    supports_dvr_from_start: bool = False


class ClipRequest(BaseModel):
    url: str = Field(..., min_length=5, max_length=600)
    num_clips: int = Field(default=3, ge=1, le=10)
    highlighted_captions: bool = Field(default=True)
    caption_language: CaptionLanguage = Field(default=CaptionLanguage.AUTO)
    aspect_ratio: AspectRatio = Field(default=AspectRatio.VERTICAL_CROP)
    highlight_color: HighlightColor = Field(default=HighlightColor.YELLOW)
    caption_style: CaptionStyle = Field(default=CaptionStyle.CLASSIC)
    caption_position: CaptionPosition = Field(default=CaptionPosition.BOTTOM)
    caption_scale: float = Field(default=1.0, ge=0.75, le=1.4)
    caption_mode: CaptionEngineMode = Field(default=CaptionEngineMode.AUTO)
    speech_model: SpeechModelPreset = Field(default=SpeechModelPreset.AUTO)
    quality: QualityPreset = Field(default=QualityPreset.FAST)
    clip_duration: int = Field(default=35, ge=10, le=90)
    live_capture_seconds: int = Field(default=120, ge=20, le=3600)
    live_from_start: bool = Field(default=False)
    manual_start: float | None = Field(default=None, ge=0.0)
    manual_end: float | None = Field(default=None, gt=0.0)
    trim_silence: bool = Field(default=False)
    hook_banner: str = Field(default="", max_length=80)
    auto_hook: bool = Field(default=True)
    streamer_cam_source: CornerPosition = Field(default=CornerPosition.OFF)
    streamer_cam_target: CornerPosition = Field(default=CornerPosition.TOP_RIGHT)
    split_cam_position: SplitCamPosition = Field(default=SplitCamPosition.TOP)
    watermark_text: str = Field(default="", max_length=32)
    audio_boost: bool = Field(default=True)
    keep_source: bool = Field(default=False)
    use_source_cache: bool = Field(default=True)
    encoder_mode: EncoderMode = Field(default=EncoderMode.AUTO)
    music_volume: float = Field(default=0.14, ge=0.0, le=0.6)
    bg_music_id: str = Field(default="", max_length=64)
    ai_provider: LocalAiProvider = Field(default=LocalAiProvider.AUTO)
    ollama_enabled: bool = Field(default=False)
    ollama_model: str = Field(default="", max_length=120)
    ollama_topic: str = Field(default="", max_length=120)
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
    hook_text: str = ""


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
        "edited_manual",
        "none",
    ] = "none"
    transcript_preview: str = ""
    hook_banner: str = ""
    hook_start: float | None = None
    hook_end: float | None = None
    words: list[WordTimestamp] = Field(default_factory=list)
    trimmed_silence_pct: float = 0.0
    social: SocialSuggestion | None = None


class ClipUpdateRequest(BaseModel):
    start: float | None = Field(default=None, ge=0.0)
    end: float | None = Field(default=None, gt=0.0)
    caption_text: str | None = Field(default=None, max_length=4000)
    words: list[WordTimestamp] | None = None
    hook_banner: str | None = Field(default=None, max_length=80)
    caption_style: CaptionStyle | None = None
    caption_position: CaptionPosition | None = None
    caption_scale: float | None = Field(default=None, ge=0.75, le=1.4)
    highlight_color: HighlightColor | None = None
    split_cam_position: SplitCamPosition | None = None


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"] = "queued"
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
    source_video_name: str | None = None
    source_video_url: str | None = None
    source_kept: bool = False
    used_cache: bool = False
    is_live_capture: bool = False
    encoder_used: str = "libx264"
    request_options: dict | None = None
    clips: list[GeneratedClip] = Field(default_factory=list)
    error: str | None = None
