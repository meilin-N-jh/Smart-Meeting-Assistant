"""Tests for Smart Meeting Assistant services."""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock


class TestLLMClient:
    """Tests for LLM client."""

    def test_llm_client_init(self):
        """Test LLM client initialization."""
        from backend.services.llm_client import LLMClient

        with patch('backend.services.llm_client.settings') as mock_settings:
            mock_settings.vllm_base_url = "http://localhost:8000/v1"
            mock_settings.vllm_api_key = "test"
            mock_settings.vllm_model = "test-model"

            client = LLMClient()
            assert client.base_url == "http://localhost:8000/v1"
            assert client.model == "test-model"

    def test_parse_json_response(self):
        """Test JSON parsing from LLM response."""
        from backend.services.llm_client import LLMClient

        with patch('backend.services.llm_client.settings') as mock_settings:
            mock_settings.vllm_base_url = "http://localhost:8000/v1"
            mock_settings.vllm_api_key = "test"
            mock_settings.vllm_model = "test-model"

            client = LLMClient()

            # Test basic JSON
            result = client._parse_json('{"key": "value"}')
            assert result == {"key": "value"}

            # Test JSON in markdown
            result = client._parse_json('```json\n{"key": "value"}\n```')
            assert result == {"key": "value"}

            # Test JSON with trailing comma
            result = client._parse_json('{"key": "value",}')
            assert result == {"key": "value"}


class TestPromptService:
    """Tests for prompt service."""

    def test_get_prompt_template(self):
        """Test prompt template retrieval."""
        from backend.services.prompt_service import get_prompt_template

        template = get_prompt_template("translation")
        assert "Translate" in template
        assert "{text}" in template

        template = get_prompt_template("summarization")
        assert "meeting" in template.lower()
        assert "JSON" in template

    def test_list_templates(self):
        """Test listing available templates."""
        from backend.services.prompt_service import list_prompt_templates

        templates = list_prompt_templates()
        assert "translation" in templates
        assert "summarization" in templates
        assert "action_items" in templates
        assert "sentiment" in templates


class TestConfig:
    """Tests for configuration."""

    def test_settings_defaults(self):
        """Test default settings."""
        from backend.core.config import Settings

        with patch.dict(os.environ, {}, clear=False):
            settings = Settings()
            assert settings.llm_profile == "7b-fp16"
            assert settings.vllm_base_url == "http://127.0.0.1:8400/v1"
            assert settings.vllm_model == "qwen2.5-7b-fp16"
            assert settings.vllm_conda_env == "qwen2.5"
            assert settings.port == 6493

    def test_settings_profile_override(self):
        """Test switching to the stronger 14B-AWQ profile."""
        from backend.core.config import Settings

        with patch.dict(os.environ, {"LLM_PROFILE": "14b-awq"}, clear=False):
            settings = Settings()
            assert settings.llm_profile == "14b-awq"
            assert settings.vllm_base_url == "http://127.0.0.1:8102/v1"
            assert settings.vllm_model == "Qwen2.5-14B-Instruct-AWQ"
            assert settings.vllm_conda_env == "awq"


class TestASRService:
    """Tests for ASR service."""

    def test_asr_service_init(self):
        """Test ASR service initialization."""
        from backend.services.asr_service import ASRService

        with patch('backend.services.asr_service.settings') as mock_settings:
            mock_settings.asr_model = "medium"
            mock_settings.asr_device = "cuda"
            mock_settings.asr_compute_type = "float16"

            service = ASRService()
            assert service.model_name == "medium"
            assert service.device == "cuda"


class TestDiarizationService:
    """Tests for diarization service."""

    def test_diarization_service_init(self):
        """Test diarization service initialization."""
        from backend.services.diarization_service import DiarizationService

        with patch('backend.services.diarization_service.settings') as mock_settings:
            mock_settings.diarization_model = "pyannote/speaker-diarization-3.1"
            mock_settings.diarization_hf_token = None

            service = DiarizationService()
            assert service.model_name == "pyannote/speaker-diarization-3.1"

    def test_assign_speaker_labels(self):
        """Test speaker label assignment."""
        from backend.services.diarization_service import DiarizationService

        service = DiarizationService()

        segments = [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_01"},
            {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_02"},
            {"start": 10.0, "end": 15.0, "speaker": "SPEAKER_01"},
        ]

        result = service.assign_speaker_labels(segments)
        speakers = [s["speaker"] for s in result]
        assert "Speaker 1" in speakers
        assert "Speaker 2" in speakers

    def test_align_speakers_to_transcript(self):
        """Test speaker alignment."""
        from backend.services.diarization_service import DiarizationService

        service = DiarizationService()

        diar_segments = [
            {"start": 0.0, "end": 10.0, "speaker": "SPEAKER_01"},
            {"start": 10.0, "end": 20.0, "speaker": "SPEAKER_02"},
        ]

        transcript_segments = [
            {"start": 0.0, "end": 5.0, "text": "Hello"},
            {"start": 5.0, "end": 12.0, "text": "World"},
        ]

        result = service.align_speakers_to_transcript(diar_segments, transcript_segments)
        assert len(result) == 2


class TestTranslationService:
    """Tests for translation service."""

    def test_translation_service_init(self):
        """Test translation service initialization."""
        from backend.services.translation_service import TranslationService

        with patch('backend.services.translation_service.get_llm_client') as mock_get_client:
            mock_client = Mock()
            mock_client.chat.return_value = "Translated text"
            mock_get_client.return_value = mock_client

            service = TranslationService()
            assert service.llm == mock_client


class TestSummarizationService:
    """Tests for summarization service."""

    def test_summarization_service_init(self):
        """Test summarization service initialization."""
        from backend.services.summarization_service import SummarizationService

        with patch('backend.services.summarization_service.get_llm_client') as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            service = SummarizationService()
            assert service.llm == mock_client
            assert service.chunk_size > 0


class TestActionItemsService:
    """Tests for action items service."""

    def test_parse_deadline(self):
        """Test deadline parsing."""
        from backend.services.action_items_service import ActionItemsService

        service = ActionItemsService()

        # Test various deadline formats
        assert service.parse_deadline("by January 15") == "January 15"
        assert service.parse_deadline("due Jan 15") == "Jan 15"
        assert service.parse_deadline("by end of week") == "end of week"

    def test_parse_priority(self):
        """Test priority parsing."""
        from backend.services.action_items_service import ActionItemsService

        service = ActionItemsService()

        assert service.parse_priority("urgent task") == "high"
        assert service.parse_priority("asap") == "high"
        assert service.parse_priority("low priority") == "low"
        assert service.parse_priority("normal task") == "medium"

    def test_enrich_items_adds_speaker_and_time_range(self):
        """Test action item enrichment with transcript metadata."""
        from backend.services.action_items_service import ActionItemsService

        service = ActionItemsService()
        items = [{
            "id": 1,
            "assignee": "unknown",
            "task": "Send the report",
            "deadline": None,
            "priority": "medium",
            "source_text": "",
            "confidence": 0.6,
        }]
        segments = [{
            "start": 62.0,
            "end": 75.0,
            "speaker": "Alice",
            "text": "I will send the report by Friday.",
        }]

        enriched = service._enrich_items_with_context(items, "Alice: I will send the report by Friday.", segments)

        assert enriched[0]["assignee"] == "Alice"
        assert enriched[0]["speaker"] == "Alice"
        assert enriched[0]["time_range"] == "01:02-01:15"


class TestSentimentService:
    """Tests for sentiment service."""

    def test_sentiment_service_init(self):
        """Test sentiment service initialization."""
        from backend.services.sentiment_service import SentimentService

        with patch('backend.services.sentiment_service.get_llm_client') as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            service = SentimentService()
            assert service.llm == mock_client


class TestSchemas:
    """Tests for API schemas."""

    def test_transcription_response_schema(self):
        """Test transcription response schema."""
        from backend.models.schemas import TranscriptionResponse, TranscriptSegment
        from datetime import datetime

        segments = [
            TranscriptSegment(start=0.0, end=5.0, text="Hello", speaker="Speaker 1")
        ]

        response = TranscriptionResponse(
            text="Hello",
            segments=segments,
            language="en"
        )

        assert response.text == "Hello"
        assert len(response.segments) == 1
        assert response.language == "en"

    def test_action_item_schema(self):
        """Test action item schema."""
        from backend.models.schemas import ActionItem

        item = ActionItem(
            id=1,
            assignee="John",
            task="Complete the report",
            deadline="Friday",
            priority="high",
            source_text="John will complete the report by Friday",
            speaker="Manager",
            time_range="00:10-00:18",
            confidence=0.9
        )

        assert item.id == 1
        assert item.assignee == "John"
        assert item.priority == "high"
        assert item.speaker == "Manager"
        assert item.time_range == "00:10-00:18"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
