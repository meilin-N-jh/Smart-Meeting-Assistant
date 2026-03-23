"""Pydantic schemas for API request/response models."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


# ============ Request Models ============

class TranscriptionRequest(BaseModel):
    """Request for audio transcription."""
    language: Optional[str] = Field(None, description="Language code (auto-detect if None)")
    beam_size: int = Field(5, description="Beam size for decoding")
    enable_diarization: bool = Field(True, description="Enable speaker diarization")


class TranslationRequest(BaseModel):
    """Request for translation."""
    text: str = Field(..., description="Text to translate")
    source_lang: str = Field("auto", description="Source language (auto for detection)")
    target_lang: str = Field("en", description="Target language")


class TranscriptTranslateRequest(BaseModel):
    """Request for transcript translation."""
    transcript: List[Dict[str, Any]] = Field(..., description="Transcript segments")
    source_lang: str = Field("auto", description="Source language")
    target_lang: str = Field("en", description="Target language")


class SummarizationRequest(BaseModel):
    """Request for meeting summarization."""
    transcript: str = Field(..., description="Meeting transcript text")
    transcript_segments: Optional[List[Dict[str, Any]]] = Field(None, description="Optional segments with timestamps")


class ActionItemsRequest(BaseModel):
    """Request for action item extraction."""
    transcript: str = Field(..., description="Meeting transcript text")
    transcript_segments: Optional[List[Dict[str, Any]]] = Field(None, description="Optional segments")


class SentimentRequest(BaseModel):
    """Request for sentiment analysis."""
    transcript: str = Field(..., description="Meeting transcript text")
    transcript_segments: Optional[List[Dict[str, Any]]] = Field(None, description="Optional segments")


class ProcessMeetingRequest(BaseModel):
    """Request for full meeting processing pipeline."""
    # Input type: "file", "text", or "mic"
    input_type: str = Field(..., description="Input type: file, text, or mic")
    file_path: Optional[str] = Field(None, description="Path to audio file (for file input)")
    text: Optional[str] = Field(None, description="Text transcript (for text input)")
    transcript_segments: Optional[List[Dict[str, Any]]] = Field(
        None, description="Optional precomputed transcript segments for text input"
    )
    language: Optional[str] = Field(None, description="Language code")
    translate_to: Optional[str] = Field(None, description="Translate to language (e.g., 'en', 'zh')")
    enable_diarization: bool = Field(True, description="Enable speaker diarization")


# ============ Response Models ============

class TranscriptSegment(BaseModel):
    """Single transcript segment."""
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Transcribed text")
    speaker: str = Field("Speaker 1", description="Speaker label")


class TranscriptionResponse(BaseModel):
    """Response for transcription."""
    text: str = Field(..., description="Full transcript text")
    segments: List[TranscriptSegment] = Field(default_factory=list, description="Timestamped segments")
    language: str = Field("en", description="Detected language")
    duration: Optional[float] = Field(None, description="Audio duration in seconds")
    timestamp: datetime = Field(default_factory=datetime.now)


class TranslationResponse(BaseModel):
    """Response for translation."""
    original_text: str = Field(..., description="Original text")
    translated_text: str = Field(..., description="Translated text")
    source_lang: str = Field(..., description="Source language")
    target_lang: str = Field(..., description="Target language")
    translated_segments: List[TranscriptSegment] = Field(default_factory=list, description="Speaker-aware translated segments")
    timestamp: datetime = Field(default_factory=datetime.now)


class SummaryResponse(BaseModel):
    """Response for summarization."""
    title: str = Field(..., description="Meeting title")
    overview: str = Field(..., description="Meeting overview")
    key_topics: List[str] = Field(default_factory=list, description="Key topics discussed")
    decisions: List[str] = Field(default_factory=list, description="Decisions made")
    blockers: List[str] = Field(default_factory=list, description="Blockers identified")
    next_steps: List[str] = Field(default_factory=list, description="Next steps")
    concise_summary: str = Field(..., description="Concise summary")
    timestamp: datetime = Field(default_factory=datetime.now)


class ActionItem(BaseModel):
    """Single action item."""
    id: int = Field(..., description="Action item ID")
    assignee: str = Field("unknown", description="Person assigned")
    task: str = Field(..., description="Task description")
    deadline: Optional[str] = Field(None, description="Deadline if specified")
    priority: str = Field("medium", description="Priority: high/medium/low")
    source_text: str = Field(..., description="Source text from transcript")
    speaker: Optional[str] = Field(None, description="Speaker who mentioned or assigned the task")
    time_range: Optional[str] = Field(None, description="Approximate segment time range, e.g. 01:12-01:25")
    confidence: float = Field(0.5, description="Confidence score 0-1")


class ActionItemsResponse(BaseModel):
    """Response for action items extraction."""
    action_items: List[ActionItem] = Field(default_factory=list, description="Extracted action items")
    count: int = Field(0, description="Total count")
    timestamp: datetime = Field(default_factory=datetime.now)


class EmotionalMoment(BaseModel):
    """Emotional moment in meeting."""
    timestamp: str = Field(..., description="Approximate timestamp")
    description: str = Field(..., description="Description of moment")
    speaker: str = Field(..., description="Speaker label")
    sentiment: str = Field("neutral", description="Sentiment: positive/negative/neutral")


class Agreement(BaseModel):
    """Agreement detection."""
    speaker: str = Field(..., description="Speaker who agreed")
    statement: str = Field(..., description="What they agreed to")
    evidence: str = Field(..., description="Quote from transcript")


class Disagreement(BaseModel):
    """Disagreement detection."""
    speaker: str = Field(..., description="Speaker who disagreed")
    statement: str = Field(..., description="What they disagreed with")
    evidence: str = Field(..., description="Quote from transcript")


class TensionPoint(BaseModel):
    """Tension point in meeting."""
    speakers: List[str] = Field(..., description="Speakers involved")
    topic: str = Field(..., description="Topic of tension")
    evidence: str = Field(..., description="Quote from transcript")


class SpeakerSignal(BaseModel):
    """Speaker-level engagement and sentiment summary."""
    speaker: str = Field(..., description="Speaker label")
    agreement_count: int = Field(0, description="Number of agreement signals")
    disagreement_count: int = Field(0, description="Number of disagreement signals")
    hesitation_count: int = Field(0, description="Number of hesitation signals")
    emotional_moment_count: int = Field(0, description="Number of emotional moments")
    tension_involvement_count: int = Field(0, description="Number of tension points involving the speaker")
    dominant_signal: str = Field("neutral", description="Dominant interaction signal for the speaker")


class SentimentResponse(BaseModel):
    """Response for sentiment analysis."""
    overall_sentiment: str = Field("neutral", description="Overall sentiment")
    engagement_level: str = Field("medium", description="Engagement level")
    emotional_moments: List[EmotionalMoment] = Field(default_factory=list)
    agreements: List[Agreement] = Field(default_factory=list)
    disagreements: List[Disagreement] = Field(default_factory=list)
    tension_points: List[TensionPoint] = Field(default_factory=list)
    hesitation_signals: List[Dict[str, str]] = Field(default_factory=list)
    evidence_quotes: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    speaker_signals: List[SpeakerSignal] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


class MeetingProcessingResponse(BaseModel):
    """Response for full meeting processing."""
    transcript: TranscriptionResponse = Field(..., description="Transcription result")
    translation: Optional[TranslationResponse] = Field(None, description="Translation if requested")
    summary: SummaryResponse = Field(..., description="Meeting summary")
    action_items: ActionItemsResponse = Field(..., description="Extracted action items")
    sentiment: SentimentResponse = Field(..., description="Sentiment analysis")
    processing_time: float = Field(..., description="Total processing time in seconds")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field("1.0.0", description="API version")
    timestamp: datetime = Field(default_factory=datetime.now)
    services: Dict[str, bool] = Field(default_factory=dict, description="Service availability")


class ErrorResponse(BaseModel):
    """Error response."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.now)
