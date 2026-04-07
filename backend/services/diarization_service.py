"""Speaker diarization service for speaker segmentation."""

import re
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from loguru import logger

from backend.core.config import settings


def _resolve_torch_device(device_name: str):
    import torch

    if not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name or "cuda")


class DiarizationService:
    """Speaker diarization service using pyannote.audio or whisperx.

    Identifies different speakers in audio and provides timestamps
    for speaker segments.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        hf_token: Optional[str] = None,
    ):
        """Initialize diarization service.

        Args:
            model_name: Diarization model name
            hf_token: HuggingFace token for model access
        """
        self.model_name = model_name or settings.diarization_model
        self.hf_token = hf_token or settings.diarization_hf_token
        self._model = None

        logger.info(f"DiarizationService initialized: model={self.model_name}")

    @property
    def model(self):
        """Lazy load the diarization model."""
        if self._model is None:
            try:
                # Try whisperx first (includes diarization)
                import whisperx
                if hasattr(whisperx, "DiarizationPipeline"):
                    self._model = {
                        "backend": "whisperx",
                        "module": whisperx,
                    }
                    logger.info("Using whisperx for diarization")
                else:
                    logger.warning(
                        "whisperx installed without DiarizationPipeline; "
                        "trying pyannote fallback"
                    )
                    raise ImportError("whisperx diarization API unavailable")
            except ImportError:
                try:
                    # Try pyannote.audio
                    from pyannote.audio import Pipeline
                    if self.hf_token:
                        try:
                            # Newer pyannote versions
                            pipeline = Pipeline.from_pretrained(
                                self.model_name,
                                token=self.hf_token,
                            )
                        except TypeError:
                            # Backward compatibility
                            pipeline = Pipeline.from_pretrained(
                                self.model_name,
                                use_auth_token=self.hf_token,
                            )
                    else:
                        pipeline = Pipeline.from_pretrained(self.model_name)
                        
                    diarization_device = _resolve_torch_device(settings.diarization_device)
                    if pipeline is not None:
                        pipeline.to(diarization_device)

                    self._model = {
                        "backend": "pyannote",
                        "pipeline": pipeline,
                    }
                    logger.info(f"Using pyannote for diarization on {diarization_device}")
                except ImportError:
                    logger.warning("No diarization library available, using text-based fallback")
                    self._model = {
                        "backend": "fallback",
                    }
                except Exception as e:
                    logger.warning(
                        f"pyannote unavailable ({e}), using text-based fallback"
                    )
                    self._model = {
                        "backend": "fallback",
                    }
        return self._model

    def is_available(self) -> bool:
        """Whether diarization capability is available (audio or text-based fallback)."""
        model = self.model
        if not model:
            return False
        return model.get("backend") in {"whisperx", "pyannote", "fallback"}

    def diarize(self, audio_path: str) -> List[Dict[str, Any]]:
        """Perform speaker diarization on audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            List of speaker segments with start, end, speaker
        """
        logger.info(f"Diarizing audio: {audio_path}")

        if self.model is None:
            return self._fallback_diarize(audio_path)

        try:
            backend = self.model.get("backend")

            if backend == "whisperx":
                whisperx = self.model["module"]
                dev_str = str(_resolve_torch_device(settings.diarization_device))
                diarize_pipeline = whisperx.DiarizationPipeline(
                    use_auth_token=self.hf_token,
                    device=dev_str,
                )
                diarization_result = diarize_pipeline(
                    audio_path,
                    min_speakers=1,
                    max_speakers=10,
                )
            elif backend == "pyannote":
                pipeline = self.model["pipeline"]
                audio_input = self._build_pyannote_audio_input(audio_path)
                if audio_input is not None:
                    diarization_result = pipeline(audio_input)
                else:
                    # Keep compatibility with pyannote default file-based loading.
                    diarization_result = pipeline(audio_path)
            elif backend == "fallback":
                return self._fallback_diarize(audio_path)
            else:
                return self._fallback_diarize(audio_path)

            annotation = diarization_result
            # pyannote>=3 may return DiarizeOutput instead of Annotation.
            if not hasattr(annotation, "itertracks") and hasattr(annotation, "speaker_diarization"):
                annotation = annotation.speaker_diarization

            if not hasattr(annotation, "itertracks"):
                raise RuntimeError("Unexpected pyannote diarization result type")

            segments: List[Dict[str, Any]] = []
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                segments.append({
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "speaker": str(speaker),
                })

            logger.info(f"Diarization complete: {len(segments)} segments")
            return segments

        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            return self._fallback_diarize(audio_path)

    def _build_pyannote_audio_input(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """Build in-memory audio input to bypass pyannote/torchcodec file decoding issues."""
        try:
            import torch
        except Exception:
            return None

        # Try torchaudio first.
        try:
            import torchaudio

            waveform, sample_rate = torchaudio.load(audio_path)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            return {
                "waveform": waveform.to(dtype=torch.float32),
                "sample_rate": int(sample_rate),
            }
        except Exception:
            pass

        # Fallback to pydub + numpy decoding.
        try:
            from pydub import AudioSegment

            audio = AudioSegment.from_file(audio_path)
            sample_rate = int(audio.frame_rate)
            channels = int(audio.channels)
            sample_width = int(audio.sample_width)

            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            if channels > 1:
                samples = samples.reshape((-1, channels)).T
                samples = samples.mean(axis=0, keepdims=True)
            else:
                samples = samples.reshape(1, -1)

            # Normalize PCM integers to [-1, 1].
            scale = float(2 ** (8 * sample_width - 1))
            if scale > 0:
                samples = samples / scale

            waveform = torch.from_numpy(samples).to(dtype=torch.float32)
            return {
                "waveform": waveform,
                "sample_rate": sample_rate,
            }
        except Exception as e:
            logger.warning(f"Failed to build in-memory audio for pyannote: {e}")
            return None

    def infer_speakers_from_transcript(
        self,
        transcript_segments: List[Dict[str, Any]],
        max_speakers: int = 6,
    ) -> List[Dict[str, Any]]:
        """Infer speaker turns from transcript text when audio diarization is unavailable.

        This is a fallback strategy that combines explicit name cues and LLM inference.
        """
        if not transcript_segments:
            return []

        # First pass: explicit name cues at segment start.
        # Support formats like "Alice: ...", "Alice, ...", "Alice - ...".
        # ASR often turns ":" into "," so we treat them equivalently.
        cue_pattern = re.compile(
            r"^\s*([A-Z][A-Za-z0-9_\-]{1,30})\s*[:,\-]\s+(.+)$"
        )
        disallowed_tokens = {
            "I", "We", "It", "This", "That", "These", "Those",
            "And", "But", "So", "Because", "If", "When", "While",
            "Hi", "Hello", "Okay", "Well", "Great", "Thanks",
        }

        cue_segments: List[Dict[str, Any]] = []
        cue_counts: Dict[str, int] = {}
        for seg in transcript_segments:
            text = str(seg.get("text", "")).strip()
            m = cue_pattern.match(text)
            if m:
                speaker_name = m.group(1).strip()
                cleaned_text = m.group(2).strip()
                if speaker_name not in disallowed_tokens:
                    cue_counts[speaker_name] = cue_counts.get(speaker_name, 0) + 1
                    cue_segments.append({
                        **seg,
                        "speaker": speaker_name,
                        "text": cleaned_text if cleaned_text else text,
                    })
                    continue
            cue_segments.append({**seg})

        # Activate cue-based assignment only when we have meaningful evidence.
        # This avoids accidentally treating sentence starters as speaker names.
        has_meaningful_cues = (
            len(cue_counts) >= 2 or any(v >= 2 for v in cue_counts.values())
        )
        if has_meaningful_cues:
            return self.assign_speaker_labels(cue_segments)

        # Second pass: LLM-based inference over segment sequence.
        try:
            from backend.services.llm_client import get_llm_client
            llm = get_llm_client()

            compact = []
            for idx, seg in enumerate(transcript_segments):
                compact.append({
                    "id": idx,
                    "start": round(float(seg.get("start", 0.0)), 2),
                    "end": round(float(seg.get("end", 0.0)), 2),
                    "text": str(seg.get("text", "")).strip()[:300],
                })

            prompt = (
                "Assign speaker IDs for meeting transcript segments.\n"
                "Use labels SPEAKER_01, SPEAKER_02, ... and keep IDs aligned.\n"
                f"Use at most {max_speakers} speakers.\n"
                "Return JSON only:\n"
                '{"assignments":[{"id":0,"speaker":"SPEAKER_01"}]}\n\n'
                f"Segments:\n{compact}"
            )

            result = llm.chat_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024,
            )

            assignments = result.get("assignments", []) if isinstance(result, dict) else []
            if not isinstance(assignments, list):
                assignments = []

            speaker_by_id = {}
            for item in assignments:
                if not isinstance(item, dict):
                    continue
                idx = item.get("id")
                spk = str(item.get("speaker", "")).strip()
                if isinstance(idx, int) and spk:
                    speaker_by_id[idx] = spk

            inferred = []
            for idx, seg in enumerate(transcript_segments):
                inferred.append({
                    **seg,
                    "speaker": speaker_by_id.get(idx, "SPEAKER_01"),
                })

            return self.assign_speaker_labels(inferred)

        except Exception as e:
            logger.warning(f"Text-based speaker inference failed, using single speaker fallback: {e}")
            return [
                {**seg, "speaker": "Speaker 1"}
                for seg in transcript_segments
            ]

    def align_speakers_to_transcript(
        self,
        diarization_segments: List[Dict[str, Any]],
        transcript_segments: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Align speaker labels with transcript segments.

        Args:
            diarization_segments: Speaker segments from diarization
            transcript_segments: Transcript segments from ASR

        Returns:
            Transcript segments with speaker labels
        """
        if not diarization_segments:
            # Return with default speaker if no diarization
            return [
                {**seg, "speaker": "Speaker 1"}
                for seg in transcript_segments
            ]

        # Assign speakers based on timestamp overlap
        result = []
        for seg in transcript_segments:
            seg_start = seg.get("start", 0)
            seg_end = seg.get("end", 0)

            # Find overlapping diarization segment
            speaker = "Speaker 1"  # Default
            for diar_seg in diarization_segments:
                if (diar_seg["start"] <= seg_start <= diar_seg["end"]) or \
                   (diar_seg["start"] <= seg_end <= diar_seg["end"]) or \
                   (seg_start <= diar_seg["start"] and seg_end >= diar_seg["end"]):
                    speaker = diar_seg.get("speaker", "Speaker 1")
                    break

            result.append({
                **seg,
                "speaker": speaker,
            })

        return result

    def assign_speaker_labels(
        self,
        segments: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Assign consistent speaker labels to segments.

        Args:
            segments: Segments with speaker info

        Returns:
            Segments with normalized speaker labels
        """
        # Map unique speakers to labels
        unique_speakers = set()
        for seg in segments:
            if "speaker" in seg:
                unique_speakers.add(seg["speaker"])

        speaker_map = {}
        label_idx = 1
        for speaker in sorted(unique_speakers):
            speaker_map[speaker] = f"Speaker {label_idx}"
            label_idx += 1

        # Apply mapping
        result = []
        for seg in segments:
            speaker = seg.get("speaker", "Speaker 1")
            result.append({
                **seg,
                "speaker": speaker_map.get(speaker, speaker),
            })

        return result

    def _fallback_diarize(self, audio_path: str) -> List[Dict[str, Any]]:
        """Fallback when no diarization model is available.

        Args:
            audio_path: Path to audio file

        Returns:
            Default speaker segments
        """
        logger.warning("Using fallback diarization")
        # Return a single segment covering the whole audio
        try:
            import soundfile as sf
            info = sf.info(audio_path)
            duration = info.duration
        except:
            duration = 3600  # Default 1 hour

        return [{
            "start": 0.0,
            "end": duration,
            "speaker": "Speaker 1",
        }]


# Global service instance
_diarization_service: Optional[DiarizationService] = None


def get_diarization_service() -> DiarizationService:
    """Get global diarization service instance."""
    global _diarization_service
    if _diarization_service is None:
        _diarization_service = DiarizationService()
    return _diarization_service
