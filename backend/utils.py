"""Utility functions for Smart Meeting Assistant."""

import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional


def format_timestamp(seconds: float) -> str:
    """Format seconds to HH:MM:SS format.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string
    """
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def parse_timestamp(timestamp_str: str) -> Optional[float]:
    """Parse timestamp string to seconds.

    Args:
        timestamp_str: Timestamp string (HH:MM:SS or MM:SS or seconds)

    Returns:
        Time in seconds or None if invalid
    """
    # Try HH:MM:SS format
    match = re.match(r"(\d+):(\d+):(\d+)", timestamp_str)
    if match:
        h, m, s = match.groups()
        return int(h) * 3600 + int(m) * 60 + int(s)

    # Try MM:SS format
    match = re.match(r"(\d+):(\d+)", timestamp_str)
    if match:
        m, s = match.groups()
        return int(m) * 60 + int(s)

    # Try plain number
    try:
        return float(timestamp_str)
    except ValueError:
        return None


def merge_transcript_segments(
    segments: List[Dict[str, Any]],
    min_gap: float = 0.5,
    same_speaker_gap: float = 2.0,
) -> List[Dict[str, Any]]:
    """Merge transcript segments that are close together.

    Args:
        segments: List of transcript segments
        min_gap: Minimum gap to consider merging (seconds)
        same_speaker_gap: Maximum gap for same speaker (seconds)

    Returns:
        Merged segments
    """
    if not segments:
        return []

    merged = [segments[0].copy()]

    for seg in segments[1:]:
        last = merged[-1]

        # Check if same speaker
        same_speaker = seg.get("speaker") == last.get("speaker")

        # Calculate gap
        gap = seg.get("start", 0) - last.get("end", 0)

        # Determine max gap threshold
        max_gap = same_speaker_gap if same_speaker else min_gap

        if gap <= max_gap and same_speaker:
            # Merge segments
            last["end"] = seg.get("end", last["end"])
            last["text"] = last.get("text", "") + " " + seg.get("text", "")
        else:
            merged.append(seg.copy())

    return merged


def extract_speaker_names(segments: List[Dict[str, Any]]) -> List[str]:
    """Extract unique speaker names from segments.

    Args:
        segments: List of transcript segments

    Returns:
        List of unique speaker names
    """
    speakers = set()
    for seg in segments:
        if "speaker" in seg:
            speakers.add(seg["speaker"])
    return sorted(list(speakers))


def calculate_speaking_time(
    segments: List[Dict[str, Any]],
    speaker: Optional[str] = None,
) -> float:
    """Calculate total speaking time.

    Args:
        segments: List of transcript segments
        speaker: Optional speaker to filter by

    Returns:
        Total speaking time in seconds
    """
    total = 0.0

    for seg in segments:
        if speaker and seg.get("speaker") != speaker:
            continue

        start = seg.get("start", 0)
        end = seg.get("end", 0)
        total += max(0, end - start)

    return total


def get_word_count(text: str) -> int:
    """Get word count for text.

    Args:
        text: Input text

    Returns:
        Word count
    """
    return len(text.split())


def truncate_text(text: str, max_words: int = 100, suffix: str = "...") -> str:
    """Truncate text to maximum word count.

    Args:
        text: Input text
        max_words: Maximum number of words
        suffix: Suffix to append if truncated

    Returns:
        Truncated text
    """
    words = text.split()
    if len(words) <= max_words:
        return text

    return " ".join(words[:max_words]) + suffix


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", filename)
    return sanitized


def validate_json_output(json_str: str) -> bool:
    """Validate if string is valid JSON.

    Args:
        json_str: JSON string to validate

    Returns:
        True if valid JSON
    """
    import json
    try:
        json.loads(json_str)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def clean_llm_output(text: str) -> str:
    """Clean LLM output by removing markdown formatting.

    Args:
        text: Raw LLM output

    Returns:
        Cleaned text
    """
    # Remove markdown code blocks
    text = re.sub(r"```[\w]*\n?", "", text)
    text = re.sub(r"```", "", text)

    # Remove trailing commas
    text = re.sub(r",\s*([}\]])", r"\1", text)

    return text.strip()
