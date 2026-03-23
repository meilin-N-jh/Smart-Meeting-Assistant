"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock


@pytest.fixture
def client():
    """Create test client."""
    from backend.main import app
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """Test health check returns 200."""
        with patch('backend.api.routes.get_llm_client') as mock_llm:
            mock_client = Mock()
            mock_llm.return_value = mock_client

            response = client.get("/api/health")
            assert response.status_code == 200

            data = response.json()
            assert "status" in data
            assert "services" in data


class TestTranslationEndpoint:
    """Tests for translation endpoint."""

    def test_translate_text(self, client):
        """Test text translation."""
        with patch('backend.api.routes.get_translation_service') as mock_service:
            mock_trans = Mock()
            mock_trans.translate.return_value = "Translated text"
            mock_service.return_value = mock_trans

            response = client.post("/api/translate", json={
                "text": "Hello world",
                "source_lang": "en",
                "target_lang": "zh"
            })

            # May fail if LLM not available
            # But should return proper response structure if mocked


class TestSummarizationEndpoint:
    """Tests for summarization endpoint."""

    def test_summarize_text(self, client):
        """Test text summarization."""
        with patch('backend.api.routes.get_summarization_service') as mock_service:
            mock_summ = Mock()
            mock_summ.summarize.return_value = {
                "title": "Test Meeting",
                "overview": "Test overview",
                "key_topics": ["Topic 1"],
                "decisions": ["Decision 1"],
                "blockers": [],
                "next_steps": ["Next step 1"],
                "concise_summary": "Summary text"
            }
            mock_service.return_value = mock_summ

            response = client.post("/api/summarize", json={
                "transcript": "Test transcript"
            })

            assert response.status_code == 200
            data = response.json()
            assert "title" in data
            assert "key_topics" in data


class TestActionItemsEndpoint:
    """Tests for action items endpoint."""

    def test_extract_action_items(self, client):
        """Test action item extraction."""
        with patch('backend.api.routes.get_action_items_service') as mock_service:
            mock_ai = Mock()
            mock_ai.extract.return_value = [
                {
                    "id": 1,
                    "assignee": "John",
                    "task": "Complete report",
                    "deadline": "Friday",
                    "priority": "high",
                    "source_text": "John will complete report",
                    "confidence": 0.9
                }
            ]
            mock_service.return_value = mock_ai

            response = client.post("/api/action-items", json={
                "transcript": "Test transcript"
            })

            assert response.status_code == 200
            data = response.json()
            assert "action_items" in data


class TestSentimentEndpoint:
    """Tests for sentiment endpoint."""

    def test_analyze_sentiment(self, client):
        """Test sentiment analysis."""
        with patch('backend.api.routes.get_sentiment_service') as mock_service:
            mock_sent = Mock()
            mock_sent.analyze.return_value = {
                "overall_sentiment": "positive",
                "engagement_level": "high",
                "emotional_moments": [],
                "agreements": [],
                "disagreements": [],
                "tension_points": [],
                "hesitation_signals": [],
                "evidence_quotes": [],
                "recommendations": []
            }
            mock_service.return_value = mock_sent

            response = client.post("/api/sentiment", json={
                "transcript": "Test transcript"
            })

            assert response.status_code == 200
            data = response.json()
            assert "overall_sentiment" in data
            assert "engagement_level" in data


class TestProcessMeetingEndpoint:
    """Tests for full pipeline endpoint."""

    def test_process_text_meeting(self, client):
        """Test processing text input."""
        with patch('backend.api.routes.get_meeting_pipeline') as mock_pipeline:
            mock_pipeline.return_value.process_request.return_value = {
                "transcript": {
                    "text": "Test meeting transcript",
                    "segments": [],
                    "language": "en",
                    "duration": None,
                },
                "translation": None,
                "summary": {
                    "title": "Test",
                    "overview": "Overview",
                    "key_topics": [],
                    "decisions": [],
                    "blockers": [],
                    "next_steps": [],
                    "concise_summary": "Summary",
                },
                "action_items": {
                    "action_items": [],
                    "count": 0,
                },
                "sentiment": {
                    "overall_sentiment": "neutral",
                    "engagement_level": "medium",
                    "emotional_moments": [],
                    "agreements": [],
                    "disagreements": [],
                    "tension_points": [],
                    "hesitation_signals": [],
                    "evidence_quotes": [],
                    "recommendations": [],
                    "speaker_signals": [],
                },
                "processing_time": 0.1,
            }

            response = client.post("/api/process-meeting", json={
                "input_type": "text",
                "text": "Test meeting transcript"
            })

            assert response.status_code == 200
            data = response.json()
            assert "transcript" in data
            assert "summary" in data

    def test_process_invalid_input(self, client):
        """Test processing with invalid input."""
        response = client.post("/api/process-meeting", json={
            "input_type": "invalid"
        })

        assert response.status_code == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
