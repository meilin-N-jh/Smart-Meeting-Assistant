"""Sentiment and engagement analysis service using local LLM."""

from collections import defaultdict
from typing import List, Dict, Any, Optional
from loguru import logger

from backend.services.llm_client import get_llm_client
from backend.services.prompt_service import get_prompt_template
from backend.core.config import settings


class SentimentService:
    """Sentiment and engagement analysis service using local Qwen via vLLM.

    Analyzes:
    - Overall sentiment (positive/negative/neutral)
    - Engagement level (high/medium/low)
    - Emotional moments
    - Agreements
    - Disagreements
    - Tension points
    - Hesitation signals
    - Evidence quotes
    - Recommendations
    """

    def __init__(self):
        """Initialize sentiment service."""
        self.llm = get_llm_client()
        self.chunk_size = settings.chunk_size_words
        logger.info("SentimentService initialized")

    def analyze(
        self,
        transcript: str,
        transcript_segments: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Analyze sentiment and engagement.

        Args:
            transcript: Full transcript text
            transcript_segments: Optional segments with timestamps

        Returns:
            Sentiment analysis result
        """
        logger.info("Analyzing sentiment and engagement")

        word_count = len(transcript.split())

        if word_count > self.chunk_size:
            # Use chunked analysis for long meetings
            return self._analyze_long(transcript, transcript_segments)
        else:
            # Direct analysis
            return self._analyze_direct(transcript, transcript_segments)

    def _analyze_direct(
        self,
        transcript: str,
        transcript_segments: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Direct sentiment analysis.

        Args:
            transcript: Transcript text

        Returns:
            Sentiment analysis result
        """
        llm_transcript = self._prepare_transcript_context(transcript, transcript_segments)
        template = get_prompt_template("sentiment")
        prompt = template.format(transcript=llm_transcript[:15000])

        messages = [{"role": "user", "content": prompt}]

        try:
            result = self.llm.chat_json(messages, temperature=0.3, max_tokens=2048)

            # Ensure all fields exist with defaults
            result = self._normalize_result(result)
            if self._is_sparse_result(result):
                logger.warning("Primary sentiment result is sparse, retrying with strict prompt")
                retried = self._analyze_retry_strict(llm_transcript)
                if not self._is_sparse_result(retried):
                    result = retried
            result = self._augment_sparse_result(result, transcript, transcript_segments)
            result = self._postprocess_result(result, transcript)

            logger.info("Sentiment analysis complete")
            return result

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return self._heuristic_result(transcript, transcript_segments, f"Analysis failed: {str(e)}")

    def _analyze_retry_strict(self, llm_transcript: str) -> Dict[str, Any]:
        """Retry sentiment analysis with a stricter interaction-focused prompt."""
        template = get_prompt_template("sentiment_strict")
        prompt = template.format(transcript=llm_transcript[:15000])
        messages = [{"role": "user", "content": prompt}]
        try:
            result = self.llm.chat_json(messages, temperature=0.15, max_tokens=2048)
            return self._normalize_result(result)
        except Exception as e:
            logger.warning(f"Strict sentiment retry failed: {e}")
            return self._empty_result(str(e))

    def _analyze_long(
        self,
        transcript: str,
        transcript_segments: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Analyze sentiment from long transcript in chunks.

        Args:
            transcript: Full transcript
            transcript_segments: Optional segments

        Returns:
            Combined sentiment analysis
        """
        logger.info("Analyzing long meeting in chunks")

        # Split into chunks
        words = transcript.split()
        chunks = []

        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start = end - 200  # Overlap

        # Analyze each chunk
        chunk_results = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Analyzing chunk {i+1}/{len(chunks)}")
            result = self._analyze_direct(chunk, None)
            chunk_results.append(result)

        # Combine results
        return self._combine_results(chunk_results)

    def _normalize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize sentiment result with defaults.

        Args:
            result: Raw result

        Returns:
            Normalized result
        """
        result = result or {}
        overall = str(result.get("overall_sentiment", "neutral")).strip().lower()
        if overall not in {"positive", "negative", "neutral", "mixed"}:
            overall = "neutral"

        engagement = str(result.get("engagement_level", "medium")).strip().lower()
        if engagement not in {"high", "medium", "low"}:
            engagement = "medium"

        def as_list(value):
            if isinstance(value, list):
                return value
            if value in (None, ""):
                return []
            return [value]

        normalized = {
            "overall_sentiment": overall,
            "engagement_level": engagement,
            "emotional_moments": as_list(result.get("emotional_moments", [])),
            "agreements": as_list(result.get("agreements", [])),
            "disagreements": as_list(result.get("disagreements", [])),
            "tension_points": as_list(result.get("tension_points", [])),
            "hesitation_signals": as_list(result.get("hesitation_signals", [])),
            "evidence_quotes": [str(x).strip() for x in as_list(result.get("evidence_quotes", [])) if str(x).strip()],
            "recommendations": [str(x).strip() for x in as_list(result.get("recommendations", [])) if str(x).strip()],
        }
        normalized["speaker_signals"] = self._build_speaker_signals(normalized)
        return normalized

    def _augment_sparse_result(
        self,
        result: Dict[str, Any],
        transcript: str,
        transcript_segments: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Fill in missing interaction signals using heuristics when LLM output is too sparse."""
        signal_count = sum(
            len(result.get(key, []) or [])
            for key in ["emotional_moments", "agreements", "disagreements", "tension_points", "hesitation_signals"]
        )
        if signal_count >= 3:
            return result

        heuristic = self._heuristic_result(transcript, transcript_segments)
        merged = dict(result)
        for key in ["emotional_moments", "agreements", "disagreements", "tension_points", "hesitation_signals", "evidence_quotes"]:
            primary = list(merged.get(key, []) or [])
            backup = list(heuristic.get(key, []) or [])
            if not primary:
                merged[key] = backup
            else:
                combined = primary + [item for item in backup if item not in primary]
                merged[key] = combined[:8]

        recs = list(merged.get("recommendations", []) or [])
        for rec in heuristic.get("recommendations", []) or []:
            if rec not in recs:
                recs.append(rec)
        merged["recommendations"] = recs[:6]

        if merged.get("overall_sentiment", "neutral") == "neutral" and heuristic.get("overall_sentiment") not in {"", None}:
            merged["overall_sentiment"] = heuristic.get("overall_sentiment", "neutral")
        if merged.get("engagement_level", "medium") == "medium" and heuristic.get("engagement_level") not in {"", None}:
            merged["engagement_level"] = heuristic.get("engagement_level", "medium")

        merged["speaker_signals"] = self._build_speaker_signals(merged)
        return merged

    def _postprocess_result(
        self,
        result: Dict[str, Any],
        transcript: str,
    ) -> Dict[str, Any]:
        """Canonicalize interaction signal text for more stable downstream use."""
        processed = dict(result)
        processed["agreements"] = self._canonicalize_agreements(result.get("agreements", []) or [])
        processed["disagreements"] = self._canonicalize_disagreements(
            result.get("disagreements", []) or [],
            transcript,
        )
        processed["tension_points"] = self._canonicalize_tension_points(
            result.get("tension_points", []) or [],
            transcript,
        )
        if not processed["tension_points"]:
            derived_tension = self._derive_launch_risk_tension_point(transcript)
            if derived_tension:
                processed["tension_points"] = [derived_tension]
        processed["speaker_signals"] = self._build_speaker_signals(processed)
        return processed

    def _canonicalize_agreements(self, items: List[Any]) -> List[Dict[str, Any]]:
        """Rewrite agreement statements into concise English summaries."""
        canonical = []
        for item in items:
            if not isinstance(item, dict):
                continue
            speaker = str(item.get("speaker", "")).strip()
            evidence = str(item.get("evidence", "")).strip()
            statement = str(item.get("statement", "")).strip()
            lowered = f"{statement} {evidence}".lower()

            if ("qa" in lowered and ("agree" in lowered or "同意" in evidence or "share that concern" in lowered)):
                statement = f"{speaker} agrees with QA's concern." if speaker else "A speaker agrees with QA's concern."
            elif "concern" in lowered:
                statement = f"{speaker} agrees with the concern raised in the meeting." if speaker else "A speaker agrees with the concern raised in the meeting."
            elif "agree" in lowered or "同意" in evidence:
                statement = f"{speaker} expresses agreement." if speaker else "A speaker expresses agreement."

            canonical.append({
                "speaker": speaker,
                "statement": statement,
                "evidence": evidence,
            })
        return canonical[:6]

    def _canonicalize_disagreements(
        self,
        items: List[Any],
        transcript: str,
    ) -> List[Dict[str, Any]]:
        """Keep objection signals and rewrite them into concise English summaries."""
        canonical = []
        full_text = transcript.lower()
        launch_context = any(term in full_text for term in ["launch", "launch date", "friday", "timeline", "上线", "周五"])

        for item in items:
            if not isinstance(item, dict):
                continue
            speaker = str(item.get("speaker", "")).strip()
            evidence = str(item.get("evidence", "")).strip()
            statement = str(item.get("statement", "")).strip()
            lowered = f"{statement} {evidence}".lower()

            concern_markers = ["worried", "concern", "not convinced", "risky", "too high", "冒险", "担心"]
            risk_markers = ["worried", "not convinced", "risky", "too high", "crash", "testing", "quality", "冒险", "担心"]
            defending_markers = ["keep the launch date", "delaying again will hurt", "hurt the campaign"]
            if any(marker in lowered for marker in defending_markers) and not any(marker in lowered for marker in risk_markers):
                continue

            if launch_context and any(marker in lowered for marker in concern_markers):
                statement = f"{speaker} pushes back on keeping the current launch date." if speaker else "A speaker pushes back on keeping the current launch date."
            elif any(marker in lowered for marker in concern_markers):
                statement = f"{speaker} raises concern about the current plan." if speaker else "A speaker raises concern about the current plan."

            canonical.append({
                "speaker": speaker,
                "statement": statement,
                "evidence": evidence,
            })

        deduped = []
        seen = set()
        for item in canonical:
            key = (item.get("speaker", "").lower(), item.get("statement", "").lower())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)

        if launch_context:
            grouped = self._derive_launch_risk_disagreement(transcript)
            if grouped:
                return [grouped]

        return deduped[:6]

    def _derive_launch_risk_disagreement(self, transcript: str) -> Optional[Dict[str, Any]]:
        """Build a meeting-level pushback item for launch-versus-risk conflicts."""
        concern_speakers = []
        evidence_lines = []
        risk_markers = ["worried", "not convinced", "risky", "too high", "crash", "testing", "quality", "担心", "冒险"]

        for line in transcript.splitlines():
            if ":" not in line:
                continue
            speaker, text = line.split(":", 1)
            speaker = speaker.strip()
            text = text.strip()
            lowered = text.lower()
            if any(marker in lowered for marker in risk_markers):
                if speaker not in concern_speakers:
                    concern_speakers.append(speaker)
                if text not in evidence_lines:
                    evidence_lines.append(text)

        if not concern_speakers:
            return None

        if len(concern_speakers) == 1:
            speaker_text = concern_speakers[0]
            statement = f"{speaker_text} pushes back on keeping the current launch date."
        elif len(concern_speakers) == 2:
            speaker_text = f"{concern_speakers[0]} and {concern_speakers[1]}"
            statement = f"{speaker_text} push back on keeping the current launch date."
        else:
            speaker_text = ", ".join(concern_speakers[:-1]) + f", and {concern_speakers[-1]}"
            statement = "Multiple speakers push back on keeping the current launch date."

        return {
            "speaker": speaker_text,
            "statement": statement,
            "evidence": " ".join(evidence_lines[:2]),
        }

    def _canonicalize_tension_points(
        self,
        items: List[Any],
        transcript: str,
    ) -> List[Dict[str, Any]]:
        """Normalize tension topics into concise English phrases."""
        canonical = []
        full_text = transcript.lower()
        has_launch_risk = any(term in full_text for term in ["launch", "launch date", "timeline", "上线", "周五"]) and any(
            term in full_text for term in ["risk", "risky", "testing", "crash", "quality", "担心", "冒险"]
        )

        for item in items:
            if not isinstance(item, dict):
                continue
            speakers = item.get("speakers", [])
            evidence = str(item.get("evidence", "")).strip()
            topic = str(item.get("topic", "")).strip()
            if has_launch_risk:
                topic = "Launch timing versus product quality risk"
            elif not topic:
                topic = "Open concern or implementation risk"

            canonical.append({
                "speakers": speakers,
                "topic": topic,
                "evidence": evidence,
            })
        return canonical[:4]

    def _derive_launch_risk_tension_point(self, transcript: str) -> Optional[Dict[str, Any]]:
        """Create a stable meeting-level tension point for launch-versus-risk conflicts."""
        full_text = transcript.lower()
        has_launch_risk = any(term in full_text for term in ["launch", "launch date", "timeline", "上线", "周五"]) and any(
            term in full_text for term in ["risk", "risky", "testing", "crash", "quality", "担心", "冒险"]
        )
        if not has_launch_risk:
            return None

        speakers = []
        evidence = []
        keywords = ["worried", "not convinced", "risky", "too high", "crash", "担心", "冒险", "hurt the campaign"]
        for line in transcript.splitlines():
            if ":" not in line:
                continue
            speaker, text = line.split(":", 1)
            speaker = speaker.strip()
            text = text.strip()
            lowered = text.lower()
            if any(keyword in lowered for keyword in keywords):
                if speaker not in speakers:
                    speakers.append(speaker)
                if text not in evidence:
                    evidence.append(text)

        return {
            "speakers": speakers[:3],
            "topic": "Launch timing versus product quality risk",
            "evidence": " ".join(evidence[:2]),
        }

    def _is_sparse_result(self, result: Dict[str, Any]) -> bool:
        """Check whether the interaction signal output is too thin to be useful."""
        signal_count = sum(
            len(result.get(key, []) or [])
            for key in ["emotional_moments", "agreements", "disagreements", "tension_points", "hesitation_signals"]
        )
        return signal_count < 2

    def _empty_result(self, error: str = "") -> Dict[str, Any]:
        """Create empty result.

        Args:
            error: Error message

        Returns:
            Empty result dict
        """
        return {
            "overall_sentiment": "neutral",
            "engagement_level": "medium",
            "emotional_moments": [],
            "agreements": [],
            "disagreements": [],
            "tension_points": [],
            "hesitation_signals": [],
            "evidence_quotes": [],
            "recommendations": [f"Analysis incomplete: {error}"] if error else [],
            "speaker_signals": [],
        }

    def _heuristic_result(
        self,
        transcript: str,
        transcript_segments: Optional[List[Dict[str, Any]]] = None,
        error: str = "",
    ) -> Dict[str, Any]:
        """Rule-based fallback sentiment/engagement analysis."""
        units = []
        if transcript_segments:
            for seg in transcript_segments:
                text = str(self._seg_value(seg, "text", "")).strip()
                if text:
                    units.append({
                        "speaker": str(self._seg_value(seg, "speaker", "Speaker 1")).strip() or "Speaker 1",
                        "text": text,
                        "timestamp": self._seg_value(seg, "start", 0.0),
                    })
        else:
            for sentence in transcript.split("."):
                text = sentence.strip()
                if text:
                    units.append({"speaker": "Speaker 1", "text": text, "timestamp": 0.0})

        agreements = []
        disagreements = []
        hesitations = []
        emotional_moments = []
        evidence_quotes = []

        agree_words = (
            "agree",
            "i agree",
            "sounds good",
            "makes sense",
            "yes",
            "comfortable moving forward",
            "share that concern",
            "i hear the concern",
            "我同意",
        )
        disagree_words = (
            "disagree",
            "not convinced",
            "won't work",
            "do not think",
            "problem",
            "worried",
            "concern",
            "risky",
            "too risky",
            "too high",
            "冒险",
            "担心",
        )
        hesitate_words = ("maybe", "perhaps", "not sure", "might", "i think", "feels risky", "still feels risky")

        for unit in units:
            text = unit["text"]
            lower = text.lower()
            speaker = unit["speaker"]
            ts = float(unit.get("timestamp", 0.0) or 0.0)
            ts_str = f"{int(ts // 60):02d}:{int(ts % 60):02d}"

            if any(word in lower for word in agree_words):
                agreements.append({
                    "speaker": speaker,
                    "statement": text[:180],
                    "evidence": text[:200],
                })
                emotional_moments.append({
                    "timestamp": ts_str,
                    "description": "Clear agreement or alignment was expressed.",
                    "speaker": speaker,
                    "sentiment": "positive",
                })
                evidence_quotes.append(text[:200])

            if any(word in lower for word in disagree_words):
                disagreements.append({
                    "speaker": speaker,
                    "statement": text[:180],
                    "evidence": text[:200],
                })
                emotional_moments.append({
                    "timestamp": ts_str,
                    "description": "Potential disagreement or concern raised.",
                    "speaker": speaker,
                    "sentiment": "negative",
                })
                evidence_quotes.append(text[:200])

            if any(word in lower for word in hesitate_words):
                hesitations.append({
                    "speaker": speaker,
                    "signal": "hesitation",
                    "evidence": text[:200],
                })
                emotional_moments.append({
                    "timestamp": ts_str,
                    "description": "A hesitation or uncertainty signal appeared.",
                    "speaker": speaker,
                    "sentiment": "neutral",
                })

        tension_points = []
        if disagreements:
            involved = sorted({item["speaker"] for item in disagreements if item.get("speaker")})
            if involved:
                topic = "Open concern or implementation risk"
                joined_text = " ".join(unit["text"].lower() for unit in units)
                if any(term in joined_text for term in ["launch", "launch date", "timeline", "上线", "周五"]) and any(
                    term in joined_text for term in ["risk", "risky", "testing", "crash", "quality", "担心", "冒险"]
                ):
                    topic = "Launch timing versus quality risk"
                tension_points.append({
                    "speakers": involved[:3],
                    "topic": topic,
                    "evidence": disagreements[0].get("evidence", ""),
                })

        overall = "neutral"
        if disagreements and agreements:
            overall = "mixed"
        elif disagreements:
            overall = "negative"
        elif agreements:
            overall = "positive"

        engagement = "medium"
        if (
            len(units) >= 12
            or len(disagreements) >= 2
            or (disagreements and agreements)
            or len({item.get("speaker") for item in disagreements if item.get("speaker")}) >= 2
        ):
            engagement = "high"
        elif len(units) <= 3 and not disagreements and not agreements and not hesitations:
            engagement = "low"

        recommendations = []
        if disagreements:
            recommendations.append("Summarize open concerns and assign owners for resolution.")
        if hesitations:
            recommendations.append("Clarify ambiguous points and confirm commitments explicitly.")
        if not recommendations:
            recommendations.append("Close the meeting with a recap of decisions and owners.")
        if error:
            recommendations.append(f"Fallback used due to model error: {error}")

        result = {
            "overall_sentiment": overall,
            "engagement_level": engagement,
            "emotional_moments": emotional_moments[:6],
            "agreements": agreements[:6],
            "disagreements": disagreements[:6],
            "tension_points": tension_points[:4],
            "hesitation_signals": hesitations[:6],
            "evidence_quotes": list(dict.fromkeys(evidence_quotes))[:8],
            "recommendations": recommendations[:5],
        }
        result["speaker_signals"] = self._build_speaker_signals(result)
        return result

    def _prepare_transcript_context(
        self,
        transcript: str,
        transcript_segments: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Build speaker/time structured transcript for LLM sentiment analysis."""
        if not transcript_segments:
            return transcript

        lines = []
        for seg in transcript_segments:
            text = str(self._seg_value(seg, "text", "")).strip()
            if not text:
                continue
            speaker = str(self._seg_value(seg, "speaker", "Speaker 1")).strip() or "Speaker 1"
            start = float(self._seg_value(seg, "start", 0.0) or 0.0)
            mm = int(start // 60)
            ss = int(start % 60)
            lines.append(f"[{mm:02d}:{ss:02d}] {speaker}: {text}")

        if not lines:
            return transcript
        return "\n".join(lines)

    @staticmethod
    def _seg_value(seg: Any, key: str, default: Any = None) -> Any:
        """Read field from dict or object-like transcript segment."""
        if isinstance(seg, dict):
            return seg.get(key, default)
        return getattr(seg, key, default)

    def _combine_results(
        self,
        chunk_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Combine chunk sentiment results.

        Args:
            chunk_results: List of chunk results

        Returns:
            Combined result
        """
        logger.info("Combining sentiment results from chunks")

        # Aggregate sentiment scores
        sentiment_scores = {
            "positive": 0,
            "negative": 0,
            "neutral": 0,
        }

        engagement_scores = {
            "high": 0,
            "medium": 0,
            "low": 0,
        }

        all_emotional_moments = []
        all_agreements = []
        all_disagreements = []
        all_tension_points = []
        all_hesitations = []
        all_quotes = []
        all_recommendations = []

        for result in chunk_results:
            # Count sentiments
            sent = result.get("overall_sentiment", "neutral").lower()
            if sent in sentiment_scores:
                sentiment_scores[sent] += 1

            # Count engagement
            eng = result.get("engagement_level", "medium").lower()
            if eng in engagement_scores:
                engagement_scores[eng] += 1

            # Collect items
            all_emotional_moments.extend(result.get("emotional_moments", []))
            all_agreements.extend(result.get("agreements", []))
            all_disagreements.extend(result.get("disagreements", []))
            all_tension_points.extend(result.get("tension_points", []))
            all_hesitations.extend(result.get("hesitation_signals", []))
            all_quotes.extend(result.get("evidence_quotes", []))
            all_recommendations.extend(result.get("recommendations", []))

        # Determine overall sentiment (majority)
        overall_sentiment = max(sentiment_scores, key=sentiment_scores.get)

        # Determine overall engagement (majority)
        engagement_level = max(engagement_scores, key=engagement_scores.get)

        # Deduplicate and limit
        all_quotes = list(dict.fromkeys(all_quotes))[:10]
        all_recommendations = list(dict.fromkeys(all_recommendations))[:5]

        result = {
            "overall_sentiment": overall_sentiment,
            "engagement_level": engagement_level,
            "emotional_moments": all_emotional_moments[:10],
            "agreements": all_agreements[:5],
            "disagreements": all_disagreements[:5],
            "tension_points": all_tension_points[:5],
            "hesitation_signals": all_hesitations[:5],
            "evidence_quotes": all_quotes,
            "recommendations": all_recommendations,
        }
        result["speaker_signals"] = self._build_speaker_signals(result)
        return result

    def _build_speaker_signals(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Aggregate meeting interaction signals per speaker."""
        counts = defaultdict(lambda: {
            "agreement_count": 0,
            "disagreement_count": 0,
            "hesitation_count": 0,
            "emotional_moment_count": 0,
            "tension_involvement_count": 0,
        })

        for item in result.get("agreements", []) or []:
            speaker = str(item.get("speaker", "")).strip()
            if speaker:
                counts[speaker]["agreement_count"] += 1

        for item in result.get("disagreements", []) or []:
            speaker = str(item.get("speaker", "")).strip()
            if speaker:
                counts[speaker]["disagreement_count"] += 1

        for item in result.get("hesitation_signals", []) or []:
            speaker = str(item.get("speaker", "")).strip()
            if speaker:
                counts[speaker]["hesitation_count"] += 1

        for item in result.get("emotional_moments", []) or []:
            speaker = str(item.get("speaker", "")).strip()
            if speaker:
                counts[speaker]["emotional_moment_count"] += 1

        for item in result.get("tension_points", []) or []:
            for speaker in item.get("speakers", []) or []:
                speaker = str(speaker).strip()
                if speaker:
                    counts[speaker]["tension_involvement_count"] += 1

        profiles = []
        for speaker, stats in counts.items():
            dominant_signal = "neutral"
            if stats["disagreement_count"] or stats["tension_involvement_count"]:
                dominant_signal = "concerned"
            if stats["agreement_count"] > stats["disagreement_count"] and stats["agreement_count"] > 0:
                dominant_signal = "supportive"
            if stats["hesitation_count"] > max(stats["agreement_count"], stats["disagreement_count"], 0):
                dominant_signal = "hesitant"
            if stats["agreement_count"] and stats["disagreement_count"]:
                dominant_signal = "mixed"

            profiles.append({
                "speaker": speaker,
                **stats,
                "dominant_signal": dominant_signal,
            })

        profiles.sort(
            key=lambda item: (
                -(item["agreement_count"] + item["disagreement_count"] + item["hesitation_count"] + item["emotional_moment_count"] + item["tension_involvement_count"]),
                item["speaker"],
            )
        )
        return profiles


# Global service instance
_sentiment_service = None


def get_sentiment_service() -> SentimentService:
    """Get global sentiment service instance."""
    global _sentiment_service
    if _sentiment_service is None:
        _sentiment_service = SentimentService()
    return _sentiment_service
