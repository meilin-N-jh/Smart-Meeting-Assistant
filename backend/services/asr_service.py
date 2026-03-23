"""ASR (Automatic Speech Recognition) service using faster-whisper."""

import os
import io
import tempfile
from typing import List, Dict, Any, Optional, AsyncIterator
import numpy as np
from loguru import logger

from backend.core.config import settings
from backend.services.llm_client import get_llm_client


class ASRService:
    """Speech-to-text service using faster-whisper.

    Supports:
    - Audio file transcription
    - Chunked transcription with timestamps
    - Speaker diarization integration
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
    ):
        """Initialize ASR service.

        Args:
            model_name: Whisper model name (tiny/base/small/medium/large)
            device: Device (cuda/cpu)
            compute_type: Compute type (float16/int8_float16/int8)
        """
        self.model_name = model_name or settings.asr_model
        self.device = device or settings.asr_device
        self.compute_type = compute_type or settings.asr_compute_type

        # Lazy load the model
        self._model = None

        logger.info(
            f"ASRService initialized: model={self.model_name}, "
            f"device={self.device}, compute_type={self.compute_type}"
        )

    @property
    def model(self):
        """Lazy load the Whisper model."""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
                self._model = WhisperModel(
                    self.model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                )
                logger.info(f"Whisper model loaded: {self.model_name}")
            except ImportError:
                logger.warning("faster-whisper not installed, using fallback")
                self._model = None
        return self._model

    def transcribe_file(
        self,
        audio_path: str,
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> Dict[str, Any]:
        """Transcribe an audio file.

        Args:
            audio_path: Path to audio file
            language: Language code (None for auto-detect)
            beam_size: Beam size for decoding
            vad_filter: Whether to use VAD filter

        Returns:
            Dict with 'text', 'segments', 'language'
        """
        logger.info(f"Transcribing file: {audio_path}")

        if self.model is None:
            return self._fallback_transcribe(audio_path)

        try:
            segments, info = self.model.transcribe(
                audio_path,
                language=language,
                beam_size=beam_size,
                vad_filter=vad_filter,
                word_timestamps=True,
            )

            segment_list = []
            full_text = []

            for segment in segments:
                seg_dict = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                }
                if hasattr(segment, "words") and segment.words:
                    seg_dict["words"] = [
                        {
                            "word": w.word,
                            "start": w.start,
                            "end": w.end,
                            "probability": w.probability,
                        }
                        for w in segment.words
                    ]

                segment_list.append(seg_dict)
                full_text.append(segment.text.strip())

            result = {
                "text": " ".join(full_text),
                "segments": segment_list,
                "language": info.language if info.language else language or "en",
                "language_probability": info.language_probability if hasattr(info, "language_probability") else None,
            }

            logger.info(
                f"Transcription complete: {len(segment_list)} segments, "
                f"language={result['language']}"
            )
            return result

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    def transcribe_audio_data(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Transcribe raw audio data.

        Args:
            audio_data: Raw audio bytes
            sample_rate: Audio sample rate
            language: Language code

        Returns:
            Transcription result dict
        """
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_data)
            temp_path = f.name

        try:
            return self.transcribe_file(temp_path, language=language)
        finally:
            os.unlink(temp_path)

    def transcribe_chunked(
        self,
        audio_path: str,
        chunk_duration: int = 30,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Transcribe audio in chunks for long files.

        Args:
            audio_path: Path to audio file
            chunk_duration: Duration of each chunk in seconds
            language: Language code

        Returns:
            Combined transcription result
        """
        logger.info(f"Chunked transcription: {audio_path}, chunk={chunk_duration}s")

        try:
            import soundfile as sf
            from pydub import AudioSegment
        except ImportError as e:
            logger.warning(f"Audio libraries not available: {e}, using direct transcription")
            return self.transcribe_file(audio_path, language=language)

        try:
            # Load audio
            audio = AudioSegment.from_file(audio_path)
            audio = audio.set_frame_rate(16000).set_channels(1)

            all_segments = []
            all_text = []
            total_duration = len(audio) / 1000  # ms to seconds

            # Process in chunks
            for start_ms in range(0, len(audio), chunk_duration * 1000):
                end_ms = min(start_ms + chunk_duration * 1000, len(audio))
                chunk = audio[start_ms:end_ms]

                # Save chunk to temp file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    chunk.export(f.name, format="wav")
                    chunk_path = f.name

                try:
                    result = self.transcribe_file(chunk_path, language=language)

                    # Adjust timestamps
                    offset = start_ms / 1000.0
                    for seg in result["segments"]:
                        seg["start"] += offset
                        seg["end"] += offset

                    all_segments.extend(result["segments"])
                    all_text.append(result["text"])
                finally:
                    os.unlink(chunk_path)

            return {
                "text": " ".join(all_text),
                "segments": all_segments,
                "language": language or "en",
                "duration": total_duration,
            }

        except Exception as e:
            logger.error(f"Chunked transcription failed: {e}")
            raise

    def _fallback_transcribe(self, audio_path: str) -> Dict[str, Any]:
        """Fallback transcription when faster-whisper is not available.

        Args:
            audio_path: Path to audio file

        Returns:
            Basic transcription result
        """
        logger.warning("Using fallback transcription - no model loaded")
        return {
            "text": "Transcription service not available. Please install faster-whisper.",
            "segments": [],
            "language": "en",
        }


# Global service instance
_asr_service: Optional[ASRService] = None


def get_asr_service() -> ASRService:
    """Get global ASR service instance."""
    global _asr_service
    if _asr_service is None:
        _asr_service = ASRService()
    return _asr_service
