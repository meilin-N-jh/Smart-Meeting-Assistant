"""Shared meeting processing pipeline."""

import re
import time
from typing import Any, List, Optional

from loguru import logger

from backend.models.schemas import (
    ActionItem,
    ActionItemsResponse,
    MeetingProcessingResponse,
    ProcessMeetingRequest,
    SentimentResponse,
    SummaryResponse,
    TranscriptSegment,
    TranscriptionResponse,
    TranslationResponse,
)
from backend.services.action_items_service import get_action_items_service
from backend.services.asr_service import get_asr_service
from backend.services.diarization_service import get_diarization_service
from backend.services.sentiment_service import get_sentiment_service
from backend.services.summarization_service import get_summarization_service
from backend.services.translation_service import get_translation_service
from backend.utils import merge_transcript_segments


class MeetingPipeline:
    """Shared pipeline for text/file meeting processing."""

    def normalize_segments(
        self,
        segments: List[dict],
        merge_gap: float = 1.5,
        merge_speaker_turns: bool = True,
    ) -> List[dict]:
        """Normalize transcript segments for downstream tasks."""
        cleaned: List[dict] = []
        for seg in segments or []:
            text = str(self._seg_value(seg, "text", "")).strip()
            if not text:
                continue

            start = float(self._seg_value(seg, "start", 0.0) or 0.0)
            end = float(self._seg_value(seg, "end", start) or start)
            if end < start:
                end = start

            speaker = str(self._seg_value(seg, "speaker", "Speaker 1")).strip() or "Speaker 1"
            cleaned.append({
                "start": start,
                "end": end,
                "text": text,
                "speaker": speaker,
            })

        cleaned.sort(key=lambda item: (item["start"], item["end"]))
        if not cleaned:
            return []

        if not merge_speaker_turns:
            return cleaned

        return merge_transcript_segments(
            cleaned,
            min_gap=0.2,
            same_speaker_gap=merge_gap,
        )

    def apply_speaker_diarization(
        self,
        audio_path: str,
        transcript_segments: List[dict],
    ) -> List[dict]:
        """Apply diarization and text fallback speaker assignment."""
        diar = get_diarization_service()
        diar_segments = diar.diarize(audio_path)
        aligned = diar.align_speakers_to_transcript(diar_segments, transcript_segments)

        unique_speakers = {
            str(seg.get("speaker", "")).strip()
            for seg in aligned
            if str(seg.get("speaker", "")).strip()
        }

        if len(unique_speakers) <= 1:
            aligned = diar.infer_speakers_from_transcript(aligned)
        else:
            aligned = diar.assign_speaker_labels(aligned)

        return self.normalize_segments(aligned)

    def transcribe_file(
        self,
        file_path: str,
        language: Optional[str],
        enable_diarization: bool,
    ) -> TranscriptionResponse:
        """Transcribe a local audio file path into a standardized response."""
        logger.info(
            "MeetingPipeline.transcribe_file start: path={}, language={}, diarization={}",
            file_path,
            language,
            enable_diarization,
        )
        asr = get_asr_service()
        result = asr.transcribe_file(file_path, language=language)
        raw_segments = result.get("segments", [])
        logger.info(
            "MeetingPipeline.transcribe_file after ASR: raw_segments={}, language={}",
            len(raw_segments),
            result.get("language", language or "en"),
        )
        if enable_diarization:
            normalized_segments = self.apply_speaker_diarization(file_path, raw_segments)
        else:
            normalized_segments = self.normalize_segments(
                raw_segments,
                merge_gap=0.0,
                merge_speaker_turns=False,
            )
        logger.info(
            "MeetingPipeline.transcribe_file after normalization: normalized_segments={}",
            len(normalized_segments),
        )

        transcript_segments = self._as_transcript_segments(normalized_segments)
        logger.info(
            "MeetingPipeline.transcribe_file after schema conversion: transcript_segments={}",
            len(transcript_segments),
        )
        transcript_text = self._build_plain_transcript_text(transcript_segments) or result.get("text", "")
        logger.info(
            "MeetingPipeline.transcribe_file before response model: transcript_chars={}",
            len(transcript_text),
        )

        response = TranscriptionResponse(
            text=transcript_text,
            segments=transcript_segments,
            language=result.get("language", language or "en"),
            duration=result.get("duration"),
        )
        logger.info("MeetingPipeline.transcribe_file completed")
        return response

    def build_text_transcript(
        self,
        text: str,
        transcript_segments: Optional[List[dict]],
        enable_diarization: bool,
        language: Optional[str],
    ) -> TranscriptionResponse:
        """Build a transcript response from raw text or precomputed segments."""
        if transcript_segments:
            normalized = self.normalize_segments(transcript_segments)
            if enable_diarization:
                unique_speakers = {
                    str(seg.get("speaker", "")).strip()
                    for seg in normalized
                    if str(seg.get("speaker", "")).strip()
                }
                if len(unique_speakers) <= 1:
                    diar = get_diarization_service()
                    normalized = self.normalize_segments(
                        diar.infer_speakers_from_transcript(normalized)
                    )
        else:
            normalized = self._infer_segments_from_text(text)

        transcript_segments_model = self._as_transcript_segments(normalized)
        transcript_text = text.strip() or self._build_plain_transcript_text(transcript_segments_model)
        return TranscriptionResponse(
            text=transcript_text,
            segments=transcript_segments_model,
            language=language or "en",
        )

    def process_request(self, request: ProcessMeetingRequest) -> MeetingProcessingResponse:
        """Run the full transcript -> analysis pipeline for text or file input."""
        start_time = time.time()
        logger.info("MeetingPipeline processing request: input_type={}", request.input_type)

        if request.input_type == "file":
            if not request.file_path:
                raise ValueError("file_path required for file input")
            transcript_response = self.transcribe_file(
                file_path=request.file_path,
                language=request.language,
                enable_diarization=request.enable_diarization,
            )
        elif request.input_type == "text":
            if not request.text:
                raise ValueError("text required for text input")
            transcript_response = self.build_text_transcript(
                text=request.text,
                transcript_segments=request.transcript_segments,
                enable_diarization=request.enable_diarization,
                language=request.language,
            )
        else:
            raise ValueError("Invalid input_type")

        transcript_segments_dump = [segment.model_dump() for segment in transcript_response.segments]
        transcript_text = transcript_response.text

        translation_response = None
        if request.translate_to:
            translation_response = self._build_translation_response(
                transcript_text=transcript_text,
                transcript_segments=transcript_segments_dump,
                source_lang=request.language or transcript_response.language or "auto",
                target_lang=request.translate_to,
            )

        summary_response = self._build_summary_response(transcript_text, transcript_segments_dump)
        action_items_response = self._build_action_items_response(transcript_text, transcript_segments_dump)
        sentiment_response = self._build_sentiment_response(transcript_text, transcript_segments_dump)

        return MeetingProcessingResponse(
            transcript=transcript_response,
            translation=translation_response,
            summary=summary_response,
            action_items=action_items_response,
            sentiment=sentiment_response,
            processing_time=time.time() - start_time,
        )

    def _build_translation_response(
        self,
        transcript_text: str,
        transcript_segments: List[dict],
        source_lang: str,
        target_lang: str,
    ) -> TranslationResponse:
        """Translate speaker-aware transcript segments."""
        service = get_translation_service()
        translated_segments_raw = service.translate_transcript(
            transcript=transcript_segments,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        translated_segments = [
            TranscriptSegment(
                start=float(seg.get("start", 0.0)),
                end=float(seg.get("end", seg.get("start", 0.0))),
                text=str(seg.get("text_translated") or seg.get("text") or "").strip(),
                speaker=str(seg.get("speaker", "Speaker 1")).strip() or "Speaker 1",
            )
            for seg in translated_segments_raw
            if str(seg.get("text_translated") or seg.get("text") or "").strip()
        ]
        translated_text = "\n".join(
            f"{segment.speaker}: {segment.text}"
            for segment in translated_segments
        )
        return TranslationResponse(
            original_text=transcript_text,
            translated_text=translated_text,
            source_lang=source_lang,
            target_lang=target_lang,
            translated_segments=translated_segments,
        )

    def _build_summary_response(
        self,
        transcript_text: str,
        transcript_segments: List[dict],
    ) -> SummaryResponse:
        """Build normalized summary response."""
        result = get_summarization_service().summarize(
            transcript=transcript_text,
            transcript_segments=transcript_segments,
        )
        if not isinstance(result, dict):
            result = {}
        return SummaryResponse(
            title=result.get("title", "Meeting Summary"),
            overview=result.get("overview", ""),
            key_topics=result.get("key_topics", []),
            decisions=result.get("decisions", []),
            blockers=result.get("blockers", []),
            next_steps=result.get("next_steps", []),
            concise_summary=result.get("concise_summary", ""),
        )

    def _build_action_items_response(
        self,
        transcript_text: str,
        transcript_segments: List[dict],
    ) -> ActionItemsResponse:
        """Build normalized action item response."""
        result = get_action_items_service().extract(
            transcript=transcript_text,
            transcript_segments=transcript_segments,
        )
        if not isinstance(result, list):
            result = []
        items = [ActionItem(**item) for item in result]
        return ActionItemsResponse(
            action_items=items,
            count=len(items),
        )

    def _build_sentiment_response(
        self,
        transcript_text: str,
        transcript_segments: List[dict],
    ) -> SentimentResponse:
        """Build normalized sentiment response."""
        result = get_sentiment_service().analyze(
            transcript=transcript_text,
            transcript_segments=transcript_segments,
        )
        if not isinstance(result, dict):
            result = {}
        return SentimentResponse(
            overall_sentiment=result.get("overall_sentiment", "neutral"),
            engagement_level=result.get("engagement_level", "medium"),
            emotional_moments=result.get("emotional_moments", []),
            agreements=result.get("agreements", []),
            disagreements=result.get("disagreements", []),
            tension_points=result.get("tension_points", []),
            hesitation_signals=result.get("hesitation_signals", []),
            evidence_quotes=result.get("evidence_quotes", []),
            recommendations=result.get("recommendations", []),
            speaker_signals=result.get("speaker_signals", []),
        )

    def _infer_segments_from_text(self, transcript_text: str) -> List[dict]:
        """Create rough segments from plain text input."""
        segments: List[dict] = []
        current_time = 0.0

        for line in transcript_text.splitlines():
            line = line.strip()
            if not line:
                continue

            speaker = "Speaker 1"
            text = line
            match = re.match(r"^([^:]+):\s*(.*)$", line)
            if match:
                potential_speaker = match.group(1).strip()
                if 1 <= len(potential_speaker) <= 20:
                    speaker = potential_speaker
                    text = match.group(2).strip()

            word_count = len(text.split())
            duration = max(min(word_count * 0.4, 15.0), 1.0)
            segments.append({
                "start": current_time,
                "end": current_time + duration,
                "text": text,
                "speaker": speaker,
            })
            current_time += duration + 0.5

        return self.normalize_segments(segments)

    def _as_transcript_segments(self, segments: List[dict]) -> List[TranscriptSegment]:
        """Convert raw dict segments into schema objects."""
        return [
            TranscriptSegment(
                start=float(seg.get("start", 0.0)),
                end=float(seg.get("end", seg.get("start", 0.0))),
                text=str(seg.get("text", "")).strip(),
                speaker=str(seg.get("speaker", "Speaker 1")).strip() or "Speaker 1",
            )
            for seg in segments
            if str(seg.get("text", "")).strip()
        ]

    def _build_plain_transcript_text(self, segments: List[TranscriptSegment]) -> str:
        """Build newline-delimited transcript text from segments."""
        return "\n".join(
            f"{segment.speaker}: {segment.text}"
            for segment in segments
            if segment.text.strip()
        )

    @staticmethod
    def _seg_value(seg: Any, key: str, default: Any = None) -> Any:
        """Read field from dict or object-like transcript segment."""
        if isinstance(seg, dict):
            return seg.get(key, default)
        return getattr(seg, key, default)


_meeting_pipeline: Optional[MeetingPipeline] = None


def get_meeting_pipeline() -> MeetingPipeline:
    """Get the shared meeting processing pipeline."""
    global _meeting_pipeline
    if _meeting_pipeline is None:
        _meeting_pipeline = MeetingPipeline()
    return _meeting_pipeline
