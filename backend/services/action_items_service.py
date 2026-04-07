"""Action items extraction service using local LLM."""

import re
from typing import List, Dict, Any, Optional
from loguru import logger

from backend.services.llm_client import get_llm_client
from backend.services.prompt_service import get_prompt_template
from backend.core.config import settings


class ActionItemsService:
    """Action item extraction service using local Qwen via vLLM.

    Extracts:
    - Assignee
    - Task description
    - Deadline (if mentioned)
    - Priority (high/medium/low)
    - Source text
    - Confidence score
    """

    def __init__(self):
        """Initialize action items service."""
        self.llm = get_llm_client()
        self.chunk_size = settings.chunk_size_words
        logger.info("ActionItemsService initialized")

    def extract(
        self,
        transcript: str,
        transcript_segments: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Extract action items from transcript.

        Args:
            transcript: Full transcript text
            transcript_segments: Optional segments with timestamps

        Returns:
            List of action items
        """
        logger.info("Extracting action items")

        word_count = len(transcript.split())

        if word_count > self.chunk_size:
            # Use chunked extraction for long meetings
            return self._extract_long(transcript, transcript_segments)
        else:
            # Direct extraction
            return self._extract_direct(transcript, transcript_segments)

    def _extract_direct(
        self,
        transcript: str,
        transcript_segments: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Direct action item extraction.

        Args:
            transcript: Transcript text

        Returns:
            List of action items
        """
        llm_transcript = self._prepare_transcript_context(transcript, transcript_segments)
        template = get_prompt_template("action_items")
        prompt = template.format(transcript=llm_transcript[:15000])

        messages = [{"role": "user", "content": prompt}]

        try:
            result = self.llm.chat_json(messages, temperature=0.3, max_tokens=2048)

            items = []
            if isinstance(result, dict):
                if isinstance(result.get("action_items"), list):
                    items = result.get("action_items", [])
                elif isinstance(result.get("items"), list):
                    items = result.get("items", [])
            elif isinstance(result, list):
                items = result

            normalized_items = []
            for idx, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                normalized_items.append(self._normalize_item(item, idx + 1))

            # Heuristic fallback if model returned nothing useful
            if not normalized_items:
                normalized_items = self._extract_with_rules(transcript, transcript_segments)

            normalized_items = self._enrich_items_with_context(
                normalized_items,
                transcript,
                transcript_segments,
            )
            normalized_items = self._deduplicate_items(normalized_items)

            logger.info(f"Extracted {len(normalized_items)} action items")
            return normalized_items

        except Exception as e:
            logger.error(f"Action item extraction failed: {e}")
            return self._extract_with_rules(transcript, transcript_segments)

    def _extract_long(
        self,
        transcript: str,
        transcript_segments: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Extract action items from long transcript in chunks.

        Args:
            transcript: Full transcript
            transcript_segments: Optional segments

        Returns:
            Combined action items
        """
        logger.info("Extracting action items from long meeting")

        # Split into chunks
        words = transcript.split()
        chunks = []

        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start = end  # No overlap for action items to avoid duplication

        # Extract from each chunk
        all_items = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            items = self._extract_direct(chunk)

            # Offset IDs
            for item in items:
                item["id"] = len(all_items) + 1

            all_items.extend(items)

        # Deduplicate
        return self._deduplicate_items(all_items)

    def _deduplicate_items(
        self,
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Remove duplicate action items.

        Args:
            items: List of action items

        Returns:
            Deduplicated list
        """
        if not items:
            return []

        # Use task description for deduplication
        seen = set()
        unique_items = []

        for item in items:
            task_key = item.get("task", "").lower().strip()
            if task_key and task_key not in seen:
                seen.add(task_key)
                unique_items.append(item)

        # Re-number IDs
        for i, item in enumerate(unique_items):
            item["id"] = i + 1

        logger.info(f"Deduplicated to {len(unique_items)} action items")
        return unique_items

    def _normalize_item(self, item: Dict[str, Any], default_id: int) -> Dict[str, Any]:
        """Normalize one action item to schema-compatible values."""
        try:
            confidence = float(item.get("confidence", 0.5))
        except Exception:
            confidence = 0.5

        confidence = max(0.0, min(1.0, confidence))
        priority = str(item.get("priority", "medium")).strip().lower()
        if priority not in {"high", "medium", "low"}:
            priority = "medium"

        source_text = str(item.get("source_text", "")).strip()
        source_text = re.sub(r"^[A-Za-z0-9 _-]+:\s*", "", source_text)
        task = str(item.get("task") or item.get("description") or "").strip()
        assignee = str(item.get("assignee", "unknown")).strip() or "unknown"

        deadline_raw = item.get("deadline")
        deadline = str(deadline_raw).strip() if deadline_raw not in (None, "") else None
        if deadline:
            deadline = re.sub(r"^(by|before)\s+", "", deadline, flags=re.IGNORECASE).strip()
        speaker = str(item.get("speaker", "")).strip() or None
        time_range = str(item.get("time_range", "")).strip() or None

        return {
            "id": int(item.get("id", default_id)),
            "assignee": assignee,
            "task": task,
            "deadline": deadline,
            "priority": priority,
            "source_text": source_text,
            "speaker": speaker,
            "time_range": time_range,
            "confidence": confidence,
        }

    def _enrich_items_with_context(
        self,
        items: List[Dict[str, Any]],
        transcript: str,
        transcript_segments: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Backfill assignee/deadline/source using transcript context."""
        enriched = []

        for idx, item in enumerate(items):
            current = dict(item)
            current["id"] = idx + 1
            task_text = str(current.get("task", "")).strip()
            source_text = str(current.get("source_text", "")).strip()
            quote = source_text or task_text

            matched_seg = self._find_best_source_segment(quote, transcript_segments)
            if matched_seg and not source_text:
                current["source_text"] = str(self._seg_value(matched_seg, "text", "")).strip()

            if matched_seg and not current.get("speaker"):
                seg_speaker = str(self._seg_value(matched_seg, "speaker", "")).strip()
                current["speaker"] = seg_speaker or None

            if matched_seg and not current.get("time_range"):
                current["time_range"] = self._format_time_range(matched_seg)

            assignee = str(current.get("assignee", "unknown")).strip().lower()
            if assignee in {"", "unknown", "none", "n/a"}:
                if matched_seg and self._should_backfill_assignee_from_speaker(current):
                    seg_speaker = str(self._seg_value(matched_seg, "speaker", "")).strip()
                    current["assignee"] = seg_speaker or "unknown"
                else:
                    current["assignee"] = "unknown"

            if not current.get("deadline"):
                current["deadline"] = self.parse_deadline(
                    " ".join(
                        part for part in [task_text, current.get("source_text", ""), transcript]
                        if part
                    )
                )

            if not current.get("task"):
                current["task"] = (current.get("source_text") or "").strip()[:180]

            if not current.get("source_text"):
                current["source_text"] = current.get("task", "")

            if current.get("priority") not in {"high", "medium", "low"}:
                current["priority"] = self.parse_priority(
                    f"{current.get('task', '')} {current.get('source_text', '')}"
                )

            enriched.append(current)

        return enriched

    def _should_backfill_assignee_from_speaker(self, item: Dict[str, Any]) -> bool:
        """Only use the segment speaker as assignee for self-commitment language.

        This avoids converting anonymous or externally assigned tasks such as
        "Someone should follow up..." into the current speaker name.
        """
        evidence = " ".join(
            str(part or "").strip()
            for part in [item.get("task", ""), item.get("source_text", "")]
        ).lower()

        if not evidence:
            return False

        self_commitment_markers = [
            "i will",
            "i'll",
            "i can",
            "i can take",
            "i can do",
            "i'll handle",
            "i will handle",
            "i will update",
            "i will confirm",
            "i will send",
            "we will",
            "we'll",
        ]
        anonymous_markers = [
            "someone should",
            "someone needs to",
            "we should",
            "needs to be",
            "should be done",
            "please ",
        ]

        if any(marker in evidence for marker in anonymous_markers):
            return False

        return any(marker in evidence for marker in self_commitment_markers)

    def _find_best_source_segment(
        self,
        quote: str,
        transcript_segments: Optional[List[Dict[str, Any]]],
    ) -> Optional[Dict[str, Any]]:
        """Find transcript segment that best matches a quote/task snippet."""
        if not quote or not transcript_segments:
            return None

        q = str(quote).strip().lower()
        if not q:
            return None

        for seg in transcript_segments:
            text = str(self._seg_value(seg, "text", "")).strip().lower()
            if not text:
                continue
            if q in text or text in q:
                return seg

        q_words = set(re.findall(r"[a-z0-9]+", q))
        best_seg = None
        best_score = 0
        for seg in transcript_segments:
            text = str(self._seg_value(seg, "text", "")).strip().lower()
            words = set(re.findall(r"[a-z0-9]+", text))
            if not words:
                continue
            overlap = len(q_words & words)
            if overlap > best_score:
                best_score = overlap
                best_seg = seg
        return best_seg

    def _extract_with_rules(
        self,
        transcript: str,
        transcript_segments: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Rule-based backup extractor for explicit commitments."""
        candidates = []
        commitment_pattern = re.compile(
            r"\b(i will|i'll|we will|we'll|need to|must|action item|follow up|prepare|send|deliver|complete)\b",
            re.IGNORECASE,
        )

        source_units: List[Dict[str, Any]] = []
        if transcript_segments:
            for seg in transcript_segments:
                text = str(self._seg_value(seg, "text", "")).strip()
                if text:
                    source_units.append({
                        "text": text,
                        "speaker": str(self._seg_value(seg, "speaker", "unknown")),
                        "start": self._seg_value(seg, "start", None),
                        "end": self._seg_value(seg, "end", None),
                    })
        else:
            for sentence in re.split(r"(?<=[.!?])\s+", transcript):
                text = sentence.strip()
                if text:
                    source_units.append({"text": text, "speaker": "unknown", "start": None, "end": None})

        for unit in source_units:
            text = str(unit.get("text", "")).strip()
            if not text:
                continue
            if not commitment_pattern.search(text):
                continue

            candidates.append({
                "id": len(candidates) + 1,
                "assignee": str(unit.get("speaker", "unknown")).strip() or "unknown",
                "task": text[:180],
                "deadline": self.parse_deadline(text),
                "priority": self.parse_priority(text),
                "source_text": text,
                "speaker": str(unit.get("speaker", "unknown")).strip() or None,
                "time_range": self._format_time_range(unit),
                "confidence": 0.55,
            })

        return self._deduplicate_items(candidates)

    def _prepare_transcript_context(
        self,
        transcript: str,
        transcript_segments: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Build transcript with speaker/time metadata for LLM extraction."""
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

    def _format_time_range(self, seg: Any) -> Optional[str]:
        """Format segment boundaries as mm:ss-mm:ss when available."""
        start = self._seg_value(seg, "start", None)
        end = self._seg_value(seg, "end", None)
        if start in (None, "") or end in (None, ""):
            return None
        try:
            start_value = max(0.0, float(start))
            end_value = max(start_value, float(end))
        except Exception:
            return None
        return f"{self._format_mmss(start_value)}-{self._format_mmss(end_value)}"

    @staticmethod
    def _format_mmss(seconds: float) -> str:
        """Format seconds into mm:ss."""
        total = int(seconds)
        minutes = total // 60
        secs = total % 60
        return f"{minutes:02d}:{secs:02d}"

    def parse_deadline(self, text: str) -> Optional[str]:
        """Parse deadline from text.

        Args:
            text: Text containing potential deadline

        Returns:
            Parsed deadline or None
        """
        # Common deadline patterns
        patterns = [
            r"by\s+(\w+\s+\d{1,2})",  # by January 15
            r"due\s+(\w+\s+\d{1,2})",  # due January 15
            r"(\w+\s+\d{1,2})",  # January 15
            r"by\s+(end of\s+\w+)",  # by end of week
            r"(\d{1,2}/\d{1,2})",  # 01/15
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def parse_priority(self, text: str) -> str:
        """Parse priority from text.

        Args:
            text: Text containing potential priority

        Returns:
            Priority (high/medium/low)
        """
        text_lower = text.lower()

        if any(word in text_lower for word in ["urgent", "asap", "critical", "important", "high priority"]):
            return "high"
        elif any(word in text_lower for word in ["low priority", "when possible", "eventually"]):
            return "low"
        else:
            return "medium"


# Global service instance
_action_items_service = None


def get_action_items_service() -> ActionItemsService:
    """Get global action items service instance."""
    global _action_items_service
    if _action_items_service is None:
        _action_items_service = ActionItemsService()
    return _action_items_service
