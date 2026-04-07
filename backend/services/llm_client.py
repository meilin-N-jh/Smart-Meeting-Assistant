"""LLM Client for local vLLM integration."""

import json
import re
import time
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI
from loguru import logger

from backend.core.config import settings


class LLMClient:
    """Client for interacting with local Qwen via vLLM.

    Provides methods for chat completions with structured JSON output,
    retry logic, and robust JSON parsing.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3,
    ):
        """Initialize LLM client.

        Args:
            base_url: vLLM base URL
            api_key: API key
            model: Model name
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.base_url = base_url or settings.vllm_base_url
        self.api_key = api_key or settings.vllm_api_key
        self.model = model or settings.vllm_model
        self.timeout = timeout
        self.max_retries = max_retries
        self._auto_switched_model = False

        # Disable environment proxy inheritance for local vLLM calls.
        # Some environments expose socks5h proxies that httpx does not accept.
        self.http_client = httpx.Client(
            timeout=self.timeout,
            trust_env=False,
        )

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=0,  # We handle retries ourselves
            http_client=self.http_client,
        )

        logger.info(
            f"LLMClient initialized: model={self.model}, base_url={self.base_url}"
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        """Send chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            Response content as string
        """
        current_max_tokens = max_tokens

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=current_max_tokens,
                    **kwargs
                )
                content = response.choices[0].message.content
                return content or ""
            except Exception as e:
                error_text = str(e)

                # vLLM returns 400 when max_tokens exceeds remaining context.
                if self._is_context_limit_error(error_text) and current_max_tokens > 256:
                    reduced = max(256, int(current_max_tokens * 0.7))
                    if reduced < current_max_tokens:
                        logger.warning(
                            "Context limit hit, reducing max_tokens from {} to {} and retrying",
                            current_max_tokens,
                            reduced,
                        )
                        current_max_tokens = reduced
                        continue

                # If configured model name does not exist on server, auto switch once.
                if self._is_model_not_found_error(error_text) and self._try_switch_model():
                    logger.warning("Model not found, switched to available model '{}'", self.model)
                    continue

                logger.error(f"LLM request failed (attempt {attempt}/{self.max_retries}): {e}")
                if attempt >= self.max_retries:
                    raise

                sleep_seconds = min(2 ** (attempt - 1), 8)
                time.sleep(sleep_seconds)

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send chat completion request and parse JSON response.

        Args:
            messages: List of message dicts
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            json_schema: Optional JSON schema for structured output

        Returns:
            Parsed JSON response as dict
        """
        # Add JSON output instruction to messages
        enhanced_messages = messages.copy()

        if json_schema:
            schema_str = json.dumps(json_schema, indent=2)
            system_msg = (
                f"You must output valid JSON matching this schema:\n{schema_str}\n"
                "Respond ONLY with valid JSON. Do not include reasoning, markdown, or code fences."
            )
        else:
            system_msg = (
                "You must output valid JSON only. "
                "Do not include reasoning, markdown, or code fences."
            )

        # Check if there's a system message
        if enhanced_messages and enhanced_messages[0].get("role") == "system":
            enhanced_messages[0]["content"] = (
                enhanced_messages[0]["content"] + "\n\n" + system_msg
            )
        else:
            enhanced_messages.insert(0, {"role": "system", "content": system_msg})

        # Make request
        response = self.chat(
            messages=enhanced_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Parse JSON with robust error handling
        return self._parse_json(response)

    def _parse_json(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response with robust error handling.

        Args:
            response: Raw response string

        Returns:
            Parsed JSON dict
        """
        candidates: List[str] = []
        text = (response or "").strip()
        if not text:
            raise ValueError("Invalid JSON response: empty response")

        # 1) Raw response
        candidates.append(text)

        # 2) Markdown JSON blocks
        candidates.extend(
            m.group(1).strip()
            for m in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
        )

        # 3) Any balanced JSON object chunks in output
        object_candidates = self._extract_balanced_chunks(text, "{", "}")
        array_candidates = self._extract_balanced_chunks(text, "[", "]")
        candidates.extend(object_candidates)
        candidates.extend(array_candidates)

        # Prefer later chunks (models often append final JSON near the end)
        seen = set()
        ordered_candidates = []
        for c in reversed(candidates):
            if c not in seen:
                seen.add(c)
                ordered_candidates.append(c)

        object_like = [c for c in ordered_candidates if c.lstrip().startswith("{")]
        array_like = [c for c in ordered_candidates if c.lstrip().startswith("[")]

        for candidate in object_like + array_like:
            parsed = self._try_json_load(candidate)
            if parsed is None:
                continue
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list):
                return {"items": parsed}

        logger.error(f"Failed to parse JSON: {text[:500]}")
        raise ValueError("Invalid JSON response: no valid JSON object found")

    def _try_json_load(self, text: str) -> Optional[Any]:
        """Try parsing JSON string with light cleanup."""
        candidate = text.strip()
        if not candidate:
            return None

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        # Remove trailing commas in objects/arrays.
        fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return None

    def _extract_balanced_chunks(self, text: str, start_char: str, end_char: str) -> List[str]:
        """Extract balanced JSON-like chunks while respecting quoted strings."""
        chunks: List[str] = []
        stack = 0
        start_idx: Optional[int] = None
        in_string = False
        escape = False

        for idx, ch in enumerate(text):
            if in_string:
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue

            if ch == start_char:
                if stack == 0:
                    start_idx = idx
                stack += 1
            elif ch == end_char and stack > 0:
                stack -= 1
                if stack == 0 and start_idx is not None:
                    chunks.append(text[start_idx:idx + 1])
                    start_idx = None

        return chunks

    @staticmethod
    def _is_context_limit_error(error_text: str) -> bool:
        lowered = error_text.lower()
        markers = [
            "max_tokens",
            "max_completion_tokens",
            "maximum context length",
            "context length",
            "too large",
        ]
        return any(marker in lowered for marker in markers)

    @staticmethod
    def _is_model_not_found_error(error_text: str) -> bool:
        lowered = error_text.lower()
        return "does not exist" in lowered or "model" in lowered and "not found" in lowered

    def _try_switch_model(self) -> bool:
        """Try to switch to first available model from vLLM server."""
        if self._auto_switched_model:
            return False

        try:
            models = self.client.models.list()
            data = getattr(models, "data", []) or []
            if not data:
                return False

            first_id = getattr(data[0], "id", None)
            if not first_id:
                return False

            self.model = first_id
            self._auto_switched_model = True
            return True
        except Exception as e:
            logger.warning(f"Failed to auto-switch model: {e}")
            return False

    def is_available(self) -> bool:
        """Return whether the configured vLLM endpoint is reachable."""
        try:
            models = self.client.models.list()
            data = getattr(models, "data", []) or []
            return bool(data)
        except Exception as e:
            logger.warning(f"LLM availability check failed: {e}")
            return False

    def chat_with_prompt_template(
        self,
        template_name: str,
        variables: Dict[str, Any],
        **kwargs
    ) -> str:
        """Chat with a prompt template.

        Args:
            template_name: Name of the prompt template
            variables: Template variables
            **kwargs: Additional arguments for chat()

        Returns:
            Response content
        """
        from backend.services.prompt_service import get_prompt_template

        template = get_prompt_template(template_name)
        prompt = template.format(**variables)

        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)

    def chat_json_with_template(
        self,
        template_name: str,
        variables: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Chat with prompt template expecting JSON response.

        Args:
            template_name: Name of the prompt template
            variables: Template variables
            **kwargs: Additional arguments for chat_json()

        Returns:
            Parsed JSON response
        """
        from backend.services.prompt_service import get_prompt_template

        template = get_prompt_template(template_name)
        prompt = template.format(**variables)

        messages = [{"role": "user", "content": prompt}]
        return self.chat_json(messages, **kwargs)


# Global client instance
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get global LLM client instance."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
