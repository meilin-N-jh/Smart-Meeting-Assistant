"""Core configuration module for Smart Meeting Assistant."""

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

LLM_PROFILE_DEFAULTS = {
    "7b-fp16": {
        "vllm_base_url": "http://127.0.0.1:8400/v1",
        "vllm_model": "qwen2.5-7b-fp16",
        "local_qwen_model_path": "/home/jiahuning2/LLM_Ability_Test/models/Qwen2.5-7B/Qwen2.5-7B-Instruct",
        "vllm_start_script": "/home/jiahuning2/LLM_Ability_Test/models/Qwen2.5-7B/start_vllm_fp16.sh",
        "vllm_conda_env": "qwen2.5",
    },
}


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        # Do not implicitly read .env for every Settings() call.
        # Global `settings` below loads project .env explicitly, while tests
        # can instantiate Settings() and get pure defaults.
        env_file=None,
        case_sensitive=False,
        extra="ignore"
    )

    llm_profile: str = Field(
        default="7b-fp16",
        description="Named LLM runtime profile. Supported: 7b-fp16",
    )

    # vLLM Configuration
    vllm_base_url: Optional[str] = Field(
        default=None,
        description="Base URL for vLLM API"
    )
    vllm_api_key: str = Field(
        default="EMPTY",
        description="API key for vLLM"
    )
    vllm_model: Optional[str] = Field(
        default=None,
        description="Model name for vLLM"
    )

    # Local Model Path
    local_qwen_model_path: Optional[str] = Field(
        default=None,
        description="Path to local Qwen model"
    )

    # vLLM Startup Script
    vllm_start_script: Optional[str] = Field(
        default=None,
        description="Path to vLLM startup script"
    )
    vllm_conda_env: Optional[str] = Field(
        default=None,
        description="Conda environment required by the selected vLLM profile",
    )

    # Server Configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=6493, description="Server port")
    debug: bool = Field(default=False, description="Debug mode")

    # ASR Configuration
    asr_model: str = Field(default="medium", description="Whisper model size")
    asr_device: str = Field(default="cuda", description="ASR device")
    asr_compute_type: str = Field(default="float16", description="ASR compute type")
    diarization_device: str = Field(default="cuda", description="Speaker diarization device")

    # Speaker Diarization
    diarization_model: str = Field(
        default="pyannote/speaker-diarization-3.1",
        description="Diarization model"
    )
    diarization_hf_token: Optional[str] = Field(
        default=None,
        description="HuggingFace token for diarization"
    )

    # Audio Configuration
    audio_chunk_duration: int = Field(
        default=5,
        description="Audio chunk duration in seconds"
    )
    audio_sample_rate: int = Field(
        default=16000,
        description="Audio sample rate"
    )

    # Meeting Processing
    chunk_size_words: int = Field(
        default=2000,
        description="Words per chunk for long meetings"
    )
    chunk_overlap_words: int = Field(
        default=200,
        description="Overlap words between chunks"
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: str = Field(default="meeting_assistant.log", description="Log file")

    # Paths
    project_root: Path = Field(default_factory=lambda: PROJECT_ROOT)
    sample_data_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "sample_data")
    prompts_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "prompts")

    @field_validator("debug", mode="before")
    @classmethod
    def normalize_debug_flag(cls, value):
        """Make DEBUG resilient to common non-boolean env values."""
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return bool(value)

        text = str(value).strip().lower()
        truthy = {"1", "true", "yes", "on", "debug", "dev", "development"}
        falsy = {"0", "false", "no", "off", "release", "prod", "production", ""}
        if text in truthy:
            return True
        if text in falsy:
            return False
        return False

    @field_validator("diarization_hf_token", mode="before")
    @classmethod
    def normalize_empty_token(cls, value):
        """Treat empty HF token as missing."""
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("llm_profile", mode="before")
    @classmethod
    def normalize_llm_profile(cls, value):
        """Normalize the LLM profile name."""
        text = str(value or "7b-fp16").strip().lower()
        aliases = {
            "fp16": "7b-fp16",
            "7b": "7b-fp16",
            "7b-fp16": "7b-fp16",
            "qwen2.5-7b-fp16": "7b-fp16",
            "qwen2.5-7b": "7b-fp16",
        }
        normalized = aliases.get(text, text)
        if normalized not in LLM_PROFILE_DEFAULTS:
            raise ValueError(f"Unsupported llm_profile '{value}'")
        return normalized

    @model_validator(mode="after")
    def apply_llm_profile_defaults(self):
        """Backfill runtime settings from the selected LLM profile."""
        defaults = LLM_PROFILE_DEFAULTS[self.llm_profile]
        if not self.vllm_base_url:
            self.vllm_base_url = defaults["vllm_base_url"]
        if not self.vllm_model:
            self.vllm_model = defaults["vllm_model"]
        if not self.local_qwen_model_path:
            self.local_qwen_model_path = defaults["local_qwen_model_path"]
        if not self.vllm_start_script:
            self.vllm_start_script = defaults["vllm_start_script"]
        if not self.vllm_conda_env:
            self.vllm_conda_env = defaults["vllm_conda_env"]
        return self


# Global settings instance
settings = Settings(_env_file=ENV_FILE if ENV_FILE.exists() else None)
