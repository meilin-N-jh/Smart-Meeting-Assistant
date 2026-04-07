"""Translation service using local LLM."""

import re
from typing import List, Dict, Any, Optional
from loguru import logger

from backend.services.llm_client import get_llm_client
from backend.services.prompt_service import get_prompt_template


class TranslationService:
    """Translation service using local Qwen via vLLM.

    Supports:
    - Transcript translation
    - Summary translation
    - Preserving speaker labels and timestamps
    - English, Chinese, Japanese support
    """

    def __init__(self):
        """Initialize translation service."""
        self.llm = get_llm_client()
        logger.info("TranslationService initialized")

    def translate(
        self,
        text: str,
        source_lang: str = "auto",
        target_lang: str = "en",
    ) -> str:
        """Translate text from source to target language.

        Args:
            text: Text to translate
            source_lang: Source language (auto for detection)
            target_lang: Target language

        Returns:
            Translated text
        """
        if not text:
            return ""
        if source_lang == target_lang or target_lang == "auto":
            return text

        lang_map = {
            "en": "English",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "fr": "French",
            "de": "German",
            "es": "Spanish",
            "auto": "auto-detect",
        }

        source_display = lang_map.get(source_lang, source_lang)
        target_display = lang_map.get(target_lang, target_lang)

        logger.info(f"Translating: {source_display} -> {target_display}")

        template = get_prompt_template("translation")
        prompt = template.format(
            source_lang=source_display,
            target_lang=target_display,
            text=text,
        )

        messages = [{"role": "user", "content": prompt}]
        try:
            result = self.llm.chat(messages, temperature=0.2, max_tokens=2048)
            return self._clean_translation_output(result, text)
        except Exception as e:
            logger.warning(f"Translation failed, returning original text: {e}")
            return text

    def _clean_translation_output(self, result: str, fallback_text: str) -> str:
        """Clean common wrapper text from model translation output."""
        text = (result or "").strip()
        if not text:
            return fallback_text

        # Remove markdown fences.
        if text.startswith("```"):
            text = text.strip("`")
            if "\n" in text:
                text = text.split("\n", 1)[1]

        # Remove explicit labels occasionally returned by LLMs.
        prefixes = [
            "translation:",
            "translated text:",
            "here is the translation:",
        ]
        lowered = text.lower()
        for prefix in prefixes:
            if lowered.startswith(prefix):
                text = text[len(prefix):].strip()
                break

        return text or fallback_text

    def translate_transcript(
        self,
        transcript: List[Dict[str, Any]],
        source_lang: str = "auto",
        target_lang: str = "en",
    ) -> List[Dict[str, Any]]:
        """Translate transcript while preserving speaker labels and timestamps.

        Args:
            transcript: List of transcript segments
            source_lang: Source language
            target_lang: Target language

        Returns:
            Translated transcript segments
        """
        if source_lang == target_lang:
            return transcript

        logger.info(f"Translating transcript: {len(transcript)} segments")

        if source_lang == "en" and target_lang == "ja":
            return self._translate_transcript_per_segment(transcript, source_lang, target_lang)
        return self._translate_transcript_batched(transcript, source_lang, target_lang)

    def _translate_transcript_per_segment(
        self,
        transcript: List[Dict[str, Any]],
        source_lang: str,
        target_lang: str,
    ) -> List[Dict[str, Any]]:
        """Translate each segment independently for stable line mapping."""
        translated_segments = []
        for seg in transcript:
            source_text = str(seg.get("text", "")).strip()
            if self._should_preserve_source_segment(source_text, str(seg.get("speaker", "")), source_lang, target_lang):
                updated = seg.copy()
                updated["text_translated"] = source_text
                translated_segments.append(updated)
                continue
            translated = self.translate(
                source_text,
                source_lang=source_lang,
                target_lang=target_lang,
            )
            updated = seg.copy()
            updated["text_translated"] = self._clean_segment_translation(
                translated,
                seg.get("speaker", "Unknown"),
            )
            translated_segments.append(updated)

        return translated_segments

    def _translate_transcript_batched(
        self,
        transcript: List[Dict[str, Any]],
        source_lang: str,
        target_lang: str,
    ) -> List[Dict[str, Any]]:
        """Translate transcript in batches for better contextual fluency."""
        translated_segments = []

        batch_size = 10
        for i in range(0, len(transcript), batch_size):
            batch = transcript[i:i + batch_size]
            batch_text = "\n".join([
                f"[{seg.get('start', 0):.1f}s] {seg.get('speaker', 'Unknown')}: {seg.get('text', '')}"
                for seg in batch
            ])
            translated_text = self.translate(batch_text, source_lang, target_lang)
            lines = translated_text.split("\n")

            for j, line in enumerate(lines):
                if j >= len(batch):
                    continue
                seg = batch[j].copy()
                if "] " in line:
                    parts = line.split("] ", 1)
                    if len(parts) == 2:
                        translated_content = parts[1].split(": ", 1)[-1] if ": " in parts[1] else parts[1]
                        seg["text_translated"] = self._clean_segment_translation(
                            translated_content,
                            seg.get("speaker", "Unknown"),
                        )
                    else:
                        seg["text_translated"] = self._clean_segment_translation(
                            line,
                            seg.get("speaker", "Unknown"),
                        )
                else:
                    seg["text_translated"] = self._clean_segment_translation(
                        line,
                        seg.get("speaker", "Unknown"),
                    )
                translated_segments.append(seg)

        return translated_segments

    def _clean_segment_translation(self, text: str, speaker: str) -> str:
        """Remove duplicated speaker labels from translated utterances."""
        cleaned = (text or "").strip()
        if not cleaned:
            return cleaned

        speaker = str(speaker or "").strip()
        if speaker:
            for prefix in (f"{speaker}:", f"{speaker}："):
                if cleaned.lower().startswith(prefix.lower()):
                    cleaned = cleaned[len(prefix):].strip()

        # Some models translate the speaker label into the utterance, e.g. "リーダー：..."
        # Strip a short leading label followed by a colon when it looks like a duplicated name.
        if re.match(r"^[^\s:：]{1,12}[：:]\s*", cleaned):
            cleaned = re.sub(r"^[^\s:：]{1,12}[：:]\s*", "", cleaned, count=1).strip()

        return cleaned

    def _should_preserve_source_segment(
        self,
        text: str,
        speaker: str,
        source_lang: str,
        target_lang: str,
    ) -> bool:
        """Keep certain ownership statements in source form for stable line mapping."""
        if not text or source_lang != "en" or target_lang != "ja":
            return False

        speaker = str(speaker or "").strip()
        match = re.match(r"^([A-Z][a-zA-Z0-9_-]+)\s+will\s+.+", text)
        if not match:
            return False

        subject = match.group(1).strip()
        if speaker and subject.lower() == speaker.lower():
            return False

        return True

    def translate_summary(
        self,
        summary: Dict[str, Any],
        source_lang: str = "auto",
        target_lang: str = "en",
    ) -> Dict[str, Any]:
        """Translate summary while preserving structure.

        Args:
            summary: Summary dict
            source_lang: Source language
            target_lang: Target language

        Returns:
            Translated summary
        """
        if source_lang == target_lang:
            return summary

        logger.info("Translating summary")

        # Translate text fields
        text_fields = [
            "title", "overview", "concise_summary"
        ]

        translated = summary.copy()

        for field in text_fields:
            if field in translated and translated[field]:
                translated[f"{field}_translated"] = self.translate(
                    translated[field], source_lang, target_lang
                )

        # Translate lists
        list_fields = [
            "key_topics", "decisions", "blockers", "next_steps"
        ]

        for field in list_fields:
            if field in translated and translated[field]:
                translated[f"{field}_translated"] = [
                    self.translate(item, source_lang, target_lang)
                    for item in translated[field]
                ]

        return translated


# Global service instance
_translation_service = None


def get_translation_service() -> TranslationService:
    """Get global translation service instance."""
    global _translation_service
    if _translation_service is None:
        _translation_service = TranslationService()
    return _translation_service
