"""API routes for Smart Meeting Assistant."""

import asyncio
import json
import os
import tempfile
import wave
from pathlib import Path
from typing import List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from loguru import logger

from backend.models.schemas import (
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptSegment,
    TranslationRequest,
    TranscriptTranslateRequest,
    TranslationResponse,
    SummarizationRequest,
    SummaryResponse,
    ActionItemsRequest,
    ActionItemsResponse,
    ActionItem,
    SentimentRequest,
    SentimentResponse,
    ProcessMeetingRequest,
    MeetingProcessingResponse,
    HealthResponse,
    ErrorResponse,
)
from backend.services import (
    get_asr_service,
    get_diarization_service,
    get_translation_service,
    get_summarization_service,
    get_action_items_service,
    get_sentiment_service,
    get_llm_client,
    get_meeting_pipeline,
)
from backend.utils import merge_transcript_segments

router = APIRouter()


def _normalize_transcript_segments(
    segments: List[dict],
    merge_gap: float = 1.5,
    merge_speaker_turns: bool = True,
) -> List[dict]:
    """Normalize transcript segments for cleaner downstream processing."""
    cleaned: List[dict] = []
    for seg in segments or []:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        if end < start:
            end = start
        speaker = str(seg.get("speaker", "Speaker 1")).strip() or "Speaker 1"
        cleaned.append({
            "start": start,
            "end": end,
            "text": text,
            "speaker": speaker,
        })

    cleaned.sort(key=lambda x: (x.get("start", 0.0), x.get("end", 0.0)))
    if not cleaned:
        return []

    if not merge_speaker_turns:
        return cleaned

    merged = merge_transcript_segments(
        cleaned,
        min_gap=0.2,
        same_speaker_gap=merge_gap,
    )
    return merged


def _apply_speaker_diarization(
    audio_path: str,
    transcript_segments: List[dict],
) -> List[dict]:
    """Apply speaker diarization with text-based fallback."""
    diar = get_diarization_service()

    diar_segments = diar.diarize(audio_path)
    aligned = diar.align_speakers_to_transcript(diar_segments, transcript_segments)

    unique_speakers = {
        str(seg.get("speaker", "")).strip()
        for seg in aligned
        if str(seg.get("speaker", "")).strip()
    }

    # If audio diarization collapses to one speaker, infer turns from transcript text.
    if len(unique_speakers) <= 1:
        aligned = diar.infer_speakers_from_transcript(aligned)
    else:
        aligned = diar.assign_speaker_labels(aligned)

    return _normalize_transcript_segments(aligned)


def _write_pcm16_wav(
    pcm_bytes: bytes,
    sample_rate: int,
) -> str:
    """Write mono PCM16 bytes to a temporary WAV file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        with wave.open(tmp, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        return tmp.name


def _process_live_audio_window(
    pcm_bytes: bytes,
    sample_rate: int,
    language: str | None,
    enable_diarization: bool,
    committed_until: float,
    final_pass: bool = False,
    analysis_window_sec: float = 8.0,
    holdback_sec: float = 1.2,
) -> dict:
    """Transcribe and diarize a rolling PCM audio window for websocket live mode."""
    total_samples = len(pcm_bytes) // 2
    if total_samples <= 0 or sample_rate <= 0:
        return {
            "segments": [],
            "language": language or "auto",
            "committed_until": committed_until,
            "total_duration": 0.0,
        }

    total_duration = total_samples / float(sample_rate)
    window_start = max(0.0, total_duration - analysis_window_sec)
    start_sample = int(window_start * sample_rate)
    window_pcm = pcm_bytes[start_sample * 2:]
    if not window_pcm:
        return {
            "segments": [],
            "language": language or "auto",
            "committed_until": committed_until,
            "total_duration": total_duration,
        }

    wav_path = _write_pcm16_wav(window_pcm, sample_rate)
    try:
        asr = get_asr_service()
        result = asr.transcribe_file(wav_path, language=language)
        segments = result.get("segments", [])
        if enable_diarization:
            segments = _apply_speaker_diarization(wav_path, segments)
        else:
            segments = _normalize_transcript_segments(segments, merge_gap=0.25)

        adjusted = []
        for seg in segments:
            start = window_start + float(seg.get("start", 0.0))
            end = window_start + float(seg.get("end", start))
            text = str(seg.get("text", "")).strip()
            speaker = str(seg.get("speaker", "Speaker 1")).strip() or "Speaker 1"
            if not text:
                continue
            adjusted.append({
                "start": start,
                "end": end,
                "text": text,
                "speaker": speaker,
            })

        commit_boundary = total_duration if final_pass else max(0.0, total_duration - holdback_sec)
        new_segments = []
        for seg in adjusted:
            if seg["end"] <= committed_until + 0.05:
                continue
            if seg["end"] > commit_boundary:
                continue
            if seg["start"] < committed_until:
                seg["start"] = committed_until
            if seg["end"] <= seg["start"]:
                continue
            new_segments.append(seg)

        normalized = _normalize_transcript_segments(new_segments, merge_gap=0.2)
        if normalized:
            committed_until = max(committed_until, max(seg["end"] for seg in normalized))

        return {
            "segments": normalized,
            "language": result.get("language", language or "auto"),
            "committed_until": committed_until,
            "total_duration": total_duration,
        }
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


# ============ Health Check ============

@router.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    services = {
        "llm": False,
        "asr": False,
        "diarization": False,
    }

    # Check LLM
    try:
        llm = get_llm_client()
        services["llm"] = llm.is_available()
    except Exception as e:
        logger.warning(f"LLM health check failed: {e}")

    # Check ASR
    try:
        asr = get_asr_service()
        services["asr"] = asr.model is not None
    except Exception as e:
        logger.warning(f"ASR health check failed: {e}")

    # Check diarization (may fail if no token)
    try:
        diar = get_diarization_service()
        if hasattr(diar, "is_available"):
            services["diarization"] = diar.is_available()
        else:
            services["diarization"] = diar.model is not None
    except Exception as e:
        logger.warning(f"Diarization health check failed: {e}")

    all_healthy = all(services.values())
    status = "healthy" if all_healthy else "degraded"

    return HealthResponse(
        status=status,
        services=services,
    )


# ============ Transcription ============

@router.post("/api/transcribe/file", response_model=TranscriptionResponse)
async def transcribe_file(
    file: UploadFile = File(...),
    language: str = Form(None),
    enable_diarization: bool = Form(True),
):
    """Transcribe an uploaded audio file."""
    logger.info(f"Transcribing uploaded file: {file.filename}")

    # Save uploaded file to temp
    suffix = Path(file.filename or "").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        return get_meeting_pipeline().transcribe_file(
            file_path=tmp_path,
            language=language,
            enable_diarization=enable_diarization,
        )

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        os.unlink(tmp_path)


@router.post("/api/transcribe/mic", response_model=TranscriptionResponse)
async def transcribe_mic(
    file: UploadFile = File(...),
    language: str = Form(None),
):
    """Transcribe microphone-recorded audio."""
    # Similar to file transcription
    return await transcribe_file(file, language, enable_diarization=True)


@router.post("/api/transcribe/file/refine-speakers", response_model=TranscriptionResponse)
async def refine_transcript_speakers(
    file: UploadFile = File(...),
    transcript_segments: str = Form(...),
    language: str = Form(None),
):
    """Apply speaker diarization to an existing transcript without rerunning ASR."""
    logger.info(f"Refining speaker labels for uploaded file: {file.filename}")

    try:
        raw_segments = json.loads(transcript_segments)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid transcript_segments JSON: {e.msg}")

    if not isinstance(raw_segments, list):
        raise HTTPException(status_code=400, detail="transcript_segments must be a JSON array")

    suffix = Path(file.filename or "").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        normalized_segments = _normalize_transcript_segments(
            raw_segments,
            merge_gap=0.0,
            merge_speaker_turns=False,
        )
        if not normalized_segments:
            return TranscriptionResponse(
                text="",
                segments=[],
                language=language or "auto",
            )

        diarized_segments = _apply_speaker_diarization(tmp_path, normalized_segments)
        transcript_segments_model = [
            TranscriptSegment(
                start=float(seg.get("start", 0.0)),
                end=float(seg.get("end", seg.get("start", 0.0))),
                text=str(seg.get("text", "")).strip(),
                speaker=str(seg.get("speaker", "Speaker 1")).strip() or "Speaker 1",
            )
            for seg in diarized_segments
            if str(seg.get("text", "")).strip()
        ]
        transcript_text = " ".join(seg.text for seg in transcript_segments_model)
        return TranscriptionResponse(
            text=transcript_text,
            segments=transcript_segments_model,
            language=language or "auto",
        )
    except Exception as e:
        logger.error(f"Speaker refinement failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@router.websocket("/ws/live-transcript")
async def websocket_live_transcript(websocket: WebSocket):
    """WebSocket endpoint for low-latency live transcription."""
    await websocket.accept()
    logger.info("Live transcript websocket connected")

    sample_rate = 16000
    language = None
    enable_diarization = True
    pcm_buffer = bytearray()
    committed_until = 0.0
    last_processed_duration = 0.0
    process_step_sec = 0.9

    try:
        await websocket.send_json({
            "type": "status",
            "message": "connected",
        })

        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect()

            text_payload = message.get("text")
            bytes_payload = message.get("bytes")

            if text_payload is not None:
                data = json.loads(text_payload)
                msg_type = str(data.get("type", "")).strip().lower()

                if msg_type == "start":
                    sample_rate = int(data.get("sample_rate") or 16000)
                    raw_language = str(data.get("language") or "").strip().lower()
                    language = None if raw_language in {"", "auto"} else raw_language
                    enable_diarization = bool(data.get("enable_diarization", True))
                    committed_until = 0.0
                    last_processed_duration = 0.0
                    pcm_buffer = bytearray()
                    await websocket.send_json({
                        "type": "status",
                        "message": "listening",
                    })
                    continue

                if msg_type == "stop":
                    await websocket.send_json({
                        "type": "status",
                        "message": "finalizing",
                    })
                    result = await asyncio.to_thread(
                        _process_live_audio_window,
                        bytes(pcm_buffer),
                        sample_rate,
                        language,
                        enable_diarization,
                        committed_until,
                        True,
                    )
                    committed_until = result["committed_until"]
                    await websocket.send_json({
                        "type": "transcript_update",
                        "segments": result["segments"],
                        "language": result["language"],
                        "committed_until": committed_until,
                        "duration": result["total_duration"],
                        "final": True,
                    })
                    await websocket.send_json({
                        "type": "status",
                        "message": "stopped",
                    })
                    break

            if bytes_payload is not None:
                pcm_buffer.extend(bytes_payload)
                total_duration = (len(pcm_buffer) // 2) / float(sample_rate or 16000)
                if total_duration - last_processed_duration < process_step_sec:
                    continue

                result = await asyncio.to_thread(
                    _process_live_audio_window,
                    bytes(pcm_buffer),
                    sample_rate,
                    language,
                    enable_diarization,
                    committed_until,
                    False,
                )
                last_processed_duration = result["total_duration"]
                if result["segments"]:
                    committed_until = result["committed_until"]
                    await websocket.send_json({
                        "type": "transcript_update",
                        "segments": result["segments"],
                        "language": result["language"],
                        "committed_until": committed_until,
                        "duration": result["total_duration"],
                        "final": False,
                    })

    except WebSocketDisconnect:
        logger.info("Live transcript websocket disconnected")
    except Exception as e:
        logger.error(f"Live transcript websocket failed: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "detail": str(e),
            })
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ============ Translation ============

@router.post("/api/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """Translate text between languages."""
    logger.info(f"Translating: {request.source_lang} -> {request.target_lang}")

    try:
        service = get_translation_service()
        result = service.translate(
            text=request.text,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
        )
        if not isinstance(result, str):
            result = str(result) if result is not None else ""

        return TranslationResponse(
            original_text=request.text,
            translated_text=result,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            translated_segments=[],
        )

    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/translate/transcript", response_model=TranslationResponse)
async def translate_transcript(request: TranscriptTranslateRequest):
    """Translate transcript segments while preserving speaker turns."""
    logger.info(f"Translating transcript with {len(request.transcript)} segments: {request.source_lang} -> {request.target_lang}")

    try:
        service = get_translation_service()
        translated_segments = service.translate_transcript(
            transcript=request.transcript,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
        )
        transcript_segments = [
            TranscriptSegment(
                start=float(seg.get("start", 0.0)),
                end=float(seg.get("end", seg.get("start", 0.0))),
                text=str(seg.get("text_translated") or seg.get("text") or "").strip(),
                speaker=str(seg.get("speaker", "Speaker 1")).strip() or "Speaker 1",
            )
            for seg in translated_segments
            if str(seg.get("text_translated") or seg.get("text") or "").strip()
        ]
        translated_text = "\n".join(
            f"{seg.speaker}: {seg.text}"
            for seg in transcript_segments
        )
        original_text = "\n".join(
            f"{str(seg.get('speaker', 'Speaker 1')).strip() or 'Speaker 1'}: {str(seg.get('text', '')).strip()}"
            for seg in request.transcript
            if str(seg.get("text", "")).strip()
        )
        return TranslationResponse(
            original_text=original_text,
            translated_text=translated_text,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            translated_segments=transcript_segments,
        )
    except Exception as e:
        logger.error(f"Transcript translation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ Summarization ============

@router.post("/api/summarize", response_model=SummaryResponse)
async def summarize_meeting(request: SummarizationRequest):
    """Summarize a meeting transcript."""
    logger.info("Generating meeting summary")

    try:
        service = get_summarization_service()
        result = service.summarize(
            transcript=request.transcript,
            transcript_segments=request.transcript_segments,
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

    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ Action Items ============

@router.post("/api/action-items", response_model=ActionItemsResponse)
async def extract_action_items(request: ActionItemsRequest):
    """Extract action items from meeting transcript."""
    logger.info("Extracting action items")

    try:
        service = get_action_items_service()
        result = service.extract(
            transcript=request.transcript,
            transcript_segments=request.transcript_segments,
        )
        if not isinstance(result, list):
            result = []

        items = [
            ActionItem(**item) for item in result
        ]

        return ActionItemsResponse(
            action_items=items,
            count=len(items),
        )

    except Exception as e:
        logger.error(f"Action item extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ Sentiment Analysis ============

@router.post("/api/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """Analyze sentiment and engagement in meeting."""
    logger.info("Analyzing sentiment and engagement")

    try:
        service = get_sentiment_service()
        result = service.analyze(
            transcript=request.transcript,
            transcript_segments=request.transcript_segments,
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

    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ Full Pipeline ============

@router.post("/api/process-meeting", response_model=MeetingProcessingResponse)
async def process_meeting(request: ProcessMeetingRequest):
    """Process complete meeting: transcribe, translate, summarize, extract action items, analyze sentiment."""
    logger.info(f"Processing meeting: input_type={request.input_type}")

    try:
        return get_meeting_pipeline().process_request(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Meeting processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============ Export Action Items ============

@router.post("/export-action-items-ics")
async def export_action_items_ics(items: List[ActionItem]):
    """Export action items as iCalendar (.ics) format for calendar applications."""
    try:
        from fastapi import Response
        # Generate iCalendar content
        ics_lines = [
            "BEGIN:VCALENDAR",
            "VERSION:2.0",
            "PRODID:-//Smart Meeting Assistant//Action Items//EN",
            "CALSCALE:GREGORIAN",
        ]
        
        from datetime import datetime, timedelta
        now = datetime.now()
        tomorrow = now + timedelta(days=1)
        
        for item in items:
            # Parse deadline or use tomorrow as default
            if item.deadline:
                try:
                    # Try to parse deadline (flexible parsing)
                    from dateutil import parser as date_parser
                    event_date = date_parser.parse(item.deadline)
                except Exception:
                    # Fallback: use tomorrow if parsing fails
                    event_date = tomorrow
            else:
                event_date = tomorrow
            
            # Format dates in iCalendar format
            dtstart = event_date.strftime("%Y%m%dT%H%M%S")
            dtend = (event_date + timedelta(hours=1)).strftime("%Y%m%dT%H%M%S")
            dtstamp = now.strftime("%Y%m%dT%H%M%SZ")
            
            # Create a unique UID based on item ID and timestamp
            uid = f"action-item-{item.id}-{int(now.timestamp())}@smartmeeting.local"
            
            # Build event description with task metadata
            description = f"Assignee: {item.assignee}\\nPriority: {item.priority}\\nConfidence: {item.confidence}"
            if item.source_text:
                description += f"\\nSource: {item.source_text[:100]}"
            
            # Create VEVENT
            ics_lines.extend([
                "BEGIN:VEVENT",
                f"UID:{uid}",
                f"DTSTAMP:{dtstamp}",
                f"DTSTART:{dtstart}",
                f"DTEND:{dtend}",
                f"SUMMARY:{item.task}",
                f"DESCRIPTION:{description}",
                "STATUS:CONFIRMED",
                "END:VEVENT",
            ])
        
        ics_lines.append("END:VCALENDAR")
        ics_content = "\r\n".join(ics_lines)
        
        return Response(
            content=ics_content,
            media_type="text/calendar",
            headers={"Content-Disposition": "attachment; filename=action-items.ics"}
        )
    
    except Exception as e:
        logger.error(f"ICS export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export action items: {str(e)}")
