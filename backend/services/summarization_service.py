"""Summarization service using local LLM."""

import re
from typing import List, Dict, Any, Optional
from loguru import logger

from backend.services.llm_client import get_llm_client
from backend.services.prompt_service import get_prompt_template
from backend.core.config import settings


class SummarizationService:
    """Meeting summarization service using local Qwen via vLLM.

    Produces structured summaries with:
    - Title
    - Overview
    - Key topics
    - Decisions
    - Blockers
    - Next steps
    - Concise summary
    """

    def __init__(self):
        """Initialize summarization service."""
        self.llm = get_llm_client()
        self.chunk_size = settings.chunk_size_words
        self.chunk_overlap = settings.chunk_overlap_words
        logger.info("SummarizationService initialized")

    def summarize(
        self,
        transcript: str,
        transcript_segments: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Summarize a meeting transcript.

        Args:
            transcript: Full transcript text
            transcript_segments: Optional segments with timestamps

        Returns:
            Summary dict
        """
        logger.info("Starting meeting summarization")
        transcript = (transcript or "").strip()
        if not transcript:
            return self._heuristic_summary("")

        # Check if transcript is too long
        word_count = len(transcript.split())

        if word_count > self.chunk_size:
            # Use chunked summarization for long meetings
            logger.info(f"Long transcript ({word_count} words), using chunked approach")
            return self._summarize_long(transcript, transcript_segments)
        else:
            # Direct summarization
            return self._summarize_direct(transcript, transcript_segments)

    def _summarize_direct(
        self,
        transcript: str,
        transcript_segments: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Direct summarization for shorter transcripts.

        Args:
            transcript: Transcript text

        Returns:
            Summary dict
        """
        llm_transcript = self._prepare_transcript_context(transcript, transcript_segments)
        template = get_prompt_template("summarization")
        prompt = template.format(transcript=llm_transcript[:15000])  # Limit input

        messages = [{"role": "user", "content": prompt}]

        try:
            result = self.llm.chat_json(messages, temperature=0.3, max_tokens=2048)
            result = self._normalize_summary(result)

            if self._summary_is_empty(result):
                logger.warning("Primary summary is empty, retrying with strict prompt")
                retried = self._summarize_retry_strict(llm_transcript)
                if not self._summary_is_empty(retried):
                    result = retried

            if self._summary_is_empty(result):
                logger.warning("LLM summary still empty, using heuristic fallback")
                result = self._heuristic_summary(transcript, transcript_segments)

            result = self._deduplicate_summary_lists(result)

            logger.info("Summarization complete")
            return result
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return self._heuristic_summary(transcript, transcript_segments, error=str(e))

    def _summarize_long(
        self,
        transcript: str,
        transcript_segments: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Chunked summarization for long meetings.

        Args:
            transcript: Full transcript
            transcript_segments: Optional segments

        Returns:
            Combined summary
        """
        logger.info("Processing long meeting in chunks")

        chunks = self._split_into_chunks(transcript, transcript_segments)

        reconstructed_chunks = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Reconstructing chunk {i+1}/{len(chunks)}")
            reconstructed = self._reconstruct_chunk(chunk)
            reconstructed_chunks.append(reconstructed)

        reduced = self._reduce_reconstructed_notes(reconstructed_chunks)
        if not self._summary_is_empty(reduced):
            return self._deduplicate_summary_lists(reduced)

        chunk_summaries = []
        for reconstructed in reconstructed_chunks:
            chunk_summaries.append(self._summary_from_reconstruction(reconstructed))
        return self._combine_summaries(chunk_summaries)

    def _split_into_chunks(
        self,
        text: str,
        transcript_segments: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """Split text into overlapping chunks.

        Args:
            text: Input text

        Returns:
            List of text chunks
        """
        if transcript_segments:
            chunks = self._split_segment_contexts(transcript_segments)
            if chunks:
                return chunks

        words = text.split()
        chunks = []

        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end >= len(words):
                break
            start = max(end - self.chunk_overlap, start + 1)

        return chunks

    def _summarize_chunk(self, chunk: str) -> Dict[str, Any]:
        """Summarize a single chunk.

        Args:
            chunk: Text chunk

        Returns:
            Chunk summary
        """
        template = get_prompt_template("chunk_summarization")
        prompt = template.format(transcript=chunk[:5000])

        messages = [{"role": "user", "content": prompt}]

        try:
            result = self.llm.chat_json(messages, temperature=0.3, max_tokens=1536)
            return result
        except Exception as e:
            logger.warning(f"Chunk summarization failed: {e}")
            return {
                "summary": "",
                "key_points": [],
                "decisions": [],
                "action_items": [],
            }

    def _reconstruct_chunk(self, chunk: str) -> Dict[str, Any]:
        """Reconstruct a transcript chunk into structured notes before summarization."""
        template = get_prompt_template("reconstructive_notes")
        prompt = template.format(transcript=chunk[:7000])

        try:
            result = self.llm.chat_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1536,
            )
            return self._normalize_reconstruction(result)
        except Exception as e:
            logger.warning(f"Chunk reconstruction failed: {e}")
            return self._heuristic_reconstruction(chunk)

    def _reduce_reconstructed_notes(self, reconstructions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Produce a final summary from reconstructed chunk notes."""
        template = get_prompt_template("reconstructive_reduce")
        notes = self._format_reconstructed_notes(reconstructions)
        prompt = template.format(notes=notes[:15000])

        try:
            result = self.llm.chat_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=2048,
            )
            return self._normalize_summary(result)
        except Exception as e:
            logger.warning(f"Reconstruction reduce failed: {e}")
            return {}

    def _summary_from_reconstruction(self, reconstruction: Dict[str, Any]) -> Dict[str, Any]:
        """Map reconstructed notes into the combine_summaries shape."""
        return {
            "summary": reconstruction.get("segment_summary", ""),
            "key_points": reconstruction.get("discussion_points", []) or reconstruction.get("topic_labels", []),
            "decisions": reconstruction.get("decisions", []),
            "blockers": reconstruction.get("blockers", []),
            "next_steps": reconstruction.get("next_steps", []),
            "action_items": reconstruction.get("next_steps", []),
        }

    def _combine_summaries(
        self,
        chunk_summaries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Combine chunk summaries into final summary.

        Args:
            chunk_summaries: List of chunk summaries

        Returns:
            Combined summary
        """
        logger.info("Combining chunk summaries")

        # Collect all items
        all_key_points = []
        all_decisions = []
        all_blockers = []
        all_next_steps = []
        all_action_items = []

        for summary in chunk_summaries:
            if "key_points" in summary:
                all_key_points.extend(summary["key_points"])
            if "decisions" in summary:
                all_decisions.extend(summary["decisions"])
            if "blockers" in summary:
                all_blockers.extend(summary.get("blockers", []))
            if "next_steps" in summary:
                all_next_steps.extend(summary.get("next_steps", []))
            if "action_items" in summary:
                all_action_items.extend(summary["action_items"])

        # Deduplicate
        all_key_points = list(dict.fromkeys(all_key_points))[:10]
        all_decisions = list(dict.fromkeys(all_decisions))[:5]
        all_blockers = list(dict.fromkeys(all_blockers))[:5]
        all_next_steps = list(dict.fromkeys(all_next_steps))[:5]

        # Generate final summary using combined info
        combined_text = f"""Meeting covered the following topics:
{chr(10).join(f"- {p}" for p in all_key_points[:5])}

Decisions made:
{chr(10).join(f"- {d}" for d in all_decisions)}

Next steps:
{chr(10).join(f"- {n}" for n in all_next_steps)}"""

        # Extract title from first chunk summary
        title = "Meeting Summary"
        if chunk_summaries and "summary" in chunk_summaries[0]:
            first_summary = chunk_summaries[0].get("summary", "")
            if first_summary:
                # Use first sentence as potential title
                title = first_summary.split(".")[0][:100]

        # Generate concise summary
        concise_summary = self._generate_concise_summary(
            all_key_points,
            all_decisions,
            all_next_steps,
        )

        return {
            "title": title,
            "overview": f"Meeting with {len(all_key_points)} key topics discussed, "
                        f"{len(all_decisions)} decisions made, "
                        f"{len(all_next_steps)} next steps identified.",
            "key_topics": all_key_points,
            "decisions": all_decisions,
            "blockers": all_blockers,
            "next_steps": all_next_steps,
            "concise_summary": concise_summary,
        }

    def _split_segment_contexts(
        self,
        transcript_segments: List[Dict[str, Any]],
    ) -> List[str]:
        """Split transcript segments into speaker-aware chunk strings."""
        chunks: List[str] = []
        current_lines: List[str] = []
        current_words = 0
        overlap_lines: List[str] = []

        for seg in transcript_segments:
            text = str(self._seg_value(seg, "text", "")).strip()
            if not text:
                continue
            speaker = str(self._seg_value(seg, "speaker", "Speaker 1")).strip() or "Speaker 1"
            start = float(self._seg_value(seg, "start", 0.0) or 0.0)
            line = f"[{int(start // 60):02d}:{int(start % 60):02d}] {speaker}: {text}"
            line_words = len(text.split())

            if current_lines and current_words + line_words > self.chunk_size:
                chunks.append("\n".join(current_lines))
                overlap_lines = current_lines[-4:] if len(current_lines) > 4 else current_lines[:]
                current_lines = overlap_lines[:]
                current_words = sum(len(item.split()) for item in current_lines)

            current_lines.append(line)
            current_words += line_words

        if current_lines:
            chunk_text = "\n".join(current_lines)
            if not chunks or chunk_text != chunks[-1]:
                chunks.append(chunk_text)

        return chunks

    def _generate_concise_summary(
        self,
        key_topics: List[str],
        decisions: List[str],
        next_steps: List[str],
    ) -> str:
        """Generate a concise final summary.

        Args:
            key_topics: Key topics list
            decisions: Decisions list
            next_steps: Next steps list

        Returns:
            Concise summary text
        """
        parts = []

        if key_topics:
            topics_str = ", ".join(key_topics[:3])
            parts.append(f"Key topics covered: {topics_str}.")

        if decisions:
            decisions_str = "; ".join(decisions[:2])
            parts.append(f"Key decisions: {decisions_str}.")

        if next_steps:
            next_str = "; ".join(next_steps[:2])
            parts.append(f"Next steps: {next_str}.")

        return " ".join(parts)

    def _normalize_summary(self, result: Any) -> Dict[str, Any]:
        """Normalize summary payload to expected schema."""
        if not isinstance(result, dict):
            result = {}

        summary = {
            "title": str(result.get("title", "")).strip(),
            "overview": str(result.get("overview", "")).strip(),
            "key_topics": result.get("key_topics", []),
            "decisions": result.get("decisions", []),
            "blockers": result.get("blockers", []),
            "next_steps": result.get("next_steps", []),
            "concise_summary": str(result.get("concise_summary", "")).strip(),
        }

        for field in ["key_topics", "decisions", "blockers", "next_steps"]:
            value = summary.get(field, [])
            if isinstance(value, list):
                summary[field] = [str(v).strip() for v in value if str(v).strip()]
            elif value:
                summary[field] = [str(value).strip()]
            else:
                summary[field] = []

        if not summary["title"]:
            summary["title"] = "Meeting Summary"

        return summary

    def _normalize_reconstruction(self, result: Any) -> Dict[str, Any]:
        """Normalize reconstructed chunk notes."""
        if not isinstance(result, dict):
            result = {}

        normalized = {
            "segment_summary": str(result.get("segment_summary", "")).strip(),
            "topic_labels": result.get("topic_labels", []),
            "discussion_points": result.get("discussion_points", []),
            "decisions": result.get("decisions", []),
            "blockers": result.get("blockers", []),
            "next_steps": result.get("next_steps", []),
            "interaction_signals": result.get("interaction_signals", []),
        }
        for field in [
            "topic_labels",
            "discussion_points",
            "decisions",
            "blockers",
            "next_steps",
            "interaction_signals",
        ]:
            value = normalized.get(field, [])
            if isinstance(value, list):
                normalized[field] = [str(item).strip() for item in value if str(item).strip()]
            elif value:
                normalized[field] = [str(value).strip()]
            else:
                normalized[field] = []
        return normalized

    def _format_reconstructed_notes(self, reconstructions: List[Dict[str, Any]]) -> str:
        """Serialize reconstructed notes for the reducer prompt."""
        blocks = []
        for idx, reconstruction in enumerate(reconstructions, start=1):
            blocks.append(
                (
                    f"Chunk {idx}\n"
                    f"Summary: {reconstruction.get('segment_summary', '')}\n"
                    f"Topics: {', '.join(reconstruction.get('topic_labels', []))}\n"
                    f"Discussion: {', '.join(reconstruction.get('discussion_points', []))}\n"
                    f"Decisions: {', '.join(reconstruction.get('decisions', []))}\n"
                    f"Blockers: {', '.join(reconstruction.get('blockers', []))}\n"
                    f"Next steps: {', '.join(reconstruction.get('next_steps', []))}\n"
                    f"Signals: {', '.join(reconstruction.get('interaction_signals', []))}"
                )
            )
        return "\n\n".join(blocks)

    def _heuristic_reconstruction(self, chunk: str) -> Dict[str, Any]:
        """Fallback reconstruction when the LLM reconstruction step fails."""
        lines = [line.strip() for line in chunk.splitlines() if line.strip()]
        topic_labels = []
        discussion_points = []
        decisions = []
        blockers = []
        next_steps = []
        interaction_signals = []

        for line in lines[:12]:
            clean = re.sub(r"^\[\d{2}:\d{2}\]\s*", "", line)
            if len(clean.split()) >= 4 and len(discussion_points) < 6:
                discussion_points.append(clean[:180])
            if ":" in clean and len(topic_labels) < 4:
                speaker, statement = clean.split(":", 1)
                if statement.strip():
                    topic_labels.append(statement.strip()[:80])

            lower = clean.lower()
            if any(word in lower for word in ["decide", "agreed", "approved"]) and len(decisions) < 4:
                decisions.append(clean[:160])
            if any(word in lower for word in ["risk", "issue", "concern", "blocked", "waiting"]) and len(blockers) < 4:
                blockers.append(clean[:160])
            if any(word in lower for word in ["will", "need to", "by ", "tomorrow", "next week"]) and len(next_steps) < 4:
                next_steps.append(clean[:160])
            if any(word in lower for word in ["agree", "disagree", "not convinced", "concerned", "hesitate"]) and len(interaction_signals) < 4:
                interaction_signals.append(clean[:160])

        return self._normalize_reconstruction({
            "segment_summary": " ".join(discussion_points[:2])[:300],
            "topic_labels": topic_labels,
            "discussion_points": discussion_points,
            "decisions": decisions,
            "blockers": blockers,
            "next_steps": next_steps,
            "interaction_signals": interaction_signals,
        })

    def _deduplicate_summary_lists(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Deduplicate summary list fields while preserving order."""
        for field in ["key_topics", "decisions", "blockers", "next_steps"]:
            values = summary.get(field, [])
            if not isinstance(values, list):
                values = [str(values)] if values else []

            seen = set()
            unique = []
            for item in values:
                text = self._normalize_summary_item(str(item).strip())
                if not text:
                    continue
                key = self._canonical_summary_key(text)
                if key in seen:
                    continue
                seen.add(key)
                unique.append(text)

            if field == "next_steps":
                action_like = []
                action_pattern = re.compile(
                    r"\b(will|need to|plan to|follow up|prepare|send|deliver|complete|review|update)\b",
                    re.IGNORECASE,
                )
                for text in unique:
                    if action_pattern.search(text):
                        action_like.append(text)
                if action_like:
                    unique = action_like

            summary[field] = unique[:8]
        return summary

    def _normalize_summary_item(self, text: str) -> str:
        """Clean list item text to reduce transcript-like artifacts."""
        if not text:
            return ""
        text = re.sub(r"^\s*(speaker\s*\d+)\s*:\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^\s*([A-Z][A-Za-z0-9_\-]{1,20})\s*:\s*", "", text)
        text = re.sub(r"^\s*([A-Z][A-Za-z0-9_\-]{1,20})\s*,\s*", "", text)
        text = re.sub(r"^\s*(great update|okay|ok|alright|well|so)\s*[,.]?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _canonical_summary_key(self, text: str) -> str:
        """Build canonical key for stronger deduplication."""
        key = re.sub(r"[^a-z0-9\s]", "", text.lower())
        key = re.sub(r"\s+", " ", key).strip()
        return key

    def _summary_is_empty(self, summary: Dict[str, Any]) -> bool:
        """Whether the summary lacks useful content."""
        if not summary:
            return True
        has_overview = bool(summary.get("overview", "").strip())
        has_concise = bool(summary.get("concise_summary", "").strip())
        has_lists = any(summary.get(k) for k in ["key_topics", "decisions", "next_steps", "blockers"])
        return not (has_overview or has_concise or has_lists)

    def _summarize_retry_strict(self, transcript: str) -> Dict[str, Any]:
        """Second-pass summary prompt with stronger non-empty constraints."""
        prompt = (
            "You are a meeting assistant.\n"
            "Create a structured meeting summary in JSON.\n"
            "Important: do not return empty strings or empty arrays for all fields.\n"
            "If uncertain, provide best-effort concise content.\n\n"
            "Return JSON only with keys:\n"
            'title, overview, key_topics, decisions, blockers, next_steps, concise_summary.\n\n'
            f"Transcript:\n{transcript[:15000]}"
        )
        try:
            result = self.llm.chat_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1536,
            )
            return self._normalize_summary(result)
        except Exception as e:
            logger.warning(f"Strict summary retry failed: {e}")
            return {}

    def _heuristic_summary(
        self,
        transcript: str,
        transcript_segments: Optional[List[Dict[str, Any]]] = None,
        error: str = "",
    ) -> Dict[str, Any]:
        """Rule-based fallback summary to guarantee non-empty output."""
        transcript = (transcript or "").strip()
        if not transcript:
            return {
                "title": "Meeting Summary",
                "overview": "No transcript content was provided.",
                "key_topics": [],
                "decisions": [],
                "blockers": [],
                "next_steps": [],
                "concise_summary": "No transcript content was provided.",
            }

        text = re.sub(r"\s+", " ", transcript)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

        # Enrich sentence pool with speaker-attributed lines when available.
        if transcript_segments:
            attributed = []
            for seg in transcript_segments:
                seg_text = str(self._seg_value(seg, "text", "")).strip()
                if not seg_text:
                    continue
                speaker = str(self._seg_value(seg, "speaker", "Speaker")).strip()
                attributed.append(f"{speaker}: {seg_text}")
            sentences = attributed + sentences

        overview_sentences = sentences[:2] if sentences else [text[:220]]
        overview = " ".join(overview_sentences)

        decision_patterns = [
            r"\b(decide|decided|agreed|freeze|approved|finalized?)\b",
            r"\b(will|going to)\b.*\b(by|tomorrow|today|next)\b",
        ]
        blocker_patterns = [
            r"\b(blocker|risk|issue|problem|concern|worried|delay)\b",
            r"\b(waiting|blocked)\b",
        ]
        next_step_patterns = [
            r"\b(will|need to|plan to|follow up|prepare|finish|complete)\b",
            r"\b(by|tomorrow|today|next|monday|tuesday|wednesday|thursday|friday)\b",
        ]

        decisions = []
        blockers = []
        next_steps = []
        key_topics = []

        for s in sentences:
            s_lower = s.lower()
            if len(key_topics) < 5 and len(s.split()) >= 4:
                key_topics.append(s[:160])

            if any(re.search(p, s_lower) for p in decision_patterns) and len(decisions) < 5:
                decisions.append(s[:180])
            if any(re.search(p, s_lower) for p in blocker_patterns) and len(blockers) < 5:
                blockers.append(s[:180])
            if any(re.search(p, s_lower) for p in next_step_patterns) and len(next_steps) < 5:
                next_steps.append(s[:180])

        # Deduplicate while preserving order
        def dedup(items: List[str]) -> List[str]:
            return list(dict.fromkeys(items))

        key_topics = dedup(key_topics)[:5]
        decisions = dedup(decisions)[:5]
        blockers = dedup(blockers)[:5]
        next_steps = dedup(next_steps)[:5]

        concise_parts = []
        if decisions:
            concise_parts.append(f"Decisions: {'; '.join(decisions[:2])}.")
        if next_steps:
            concise_parts.append(f"Next steps: {'; '.join(next_steps[:2])}.")
        if not concise_parts:
            concise_parts.append(overview)
        if error:
            concise_parts.append(f"(Fallback summary generated due to model formatting issue: {error})")

        return {
            "title": "Meeting Summary",
            "overview": overview,
            "key_topics": key_topics,
            "decisions": decisions,
            "blockers": blockers,
            "next_steps": next_steps,
            "concise_summary": " ".join(concise_parts).strip(),
        }

    def _prepare_transcript_context(
        self,
        transcript: str,
        transcript_segments: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Build structured transcript context for better meeting-style summaries."""
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


# Global service instance
_summarization_service = None


def get_summarization_service() -> SummarizationService:
    """Get global summarization service instance."""
    global _summarization_service
    if _summarization_service is None:
        _summarization_service = SummarizationService()
    return _summarization_service
