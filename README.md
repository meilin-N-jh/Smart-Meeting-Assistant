# Smart Meeting Assistant

A course-quality smart meeting assistant for real-time and offline meeting support. The system provides:

- speech-to-text transcription
- speaker attribution and refinement
- structured meeting summarization
- action item extraction
- multilingual transcript translation
- meeting sentiment and engagement analysis

The final stack uses a local `Qwen2.5-7B FP16` model served by vLLM, plus `faster-whisper` and `pyannote` for audio processing.

## Highlights

### Two-stage transcript pipeline

The system does not wait for the full meeting pipeline before showing text.

1. Fast ASR produces an initial transcript as early as possible.
2. Speaker refinement runs in the background.
3. Summary, action items, sentiment, and translation are refreshed from the refined transcript.

This makes the UI responsive while still preserving structured final outputs.

### Structured downstream tasks

The project is not prompt-only. It combines:

- task-specific prompts
- shared transcript schemas
- JSON extraction and repair
- post-processing and normalization
- light heuristic fallbacks where needed

This is important because the project depends on stable UI-facing structures rather than raw free-form model text.

## Features

### 1. Transcription

- live microphone recording
- uploaded audio files
- timestamped transcript segments
- background speaker refinement
- transcript evidence linking in the UI

### 2. Summarization

- overview
- concise summary
- key topics
- decisions
- blockers
- next steps

### 3. Action Items

- assignee extraction
- task extraction
- deadline extraction
- confidence scoring
- source-text grounding

### 4. Sentiment and Engagement

- overall sentiment
- engagement level
- agreement signals
- disagreement signals
- tension points
- emotionally significant moments

### 5. Translation

- transcript translation
- speaker and timestamp preservation
- structure-aware output formatting

## Architecture

High-level flow:

`Audio/Text Input -> Transcript Construction -> Speaker Refinement -> Parallel NLP Tasks -> Frontend Rendering`

Main backend modules:

- `backend/api/routes.py`
- `backend/services/meeting_pipeline.py`
- `backend/services/asr_service.py`
- `backend/services/diarization_service.py`
- `backend/services/summarization_service.py`
- `backend/services/action_items_service.py`
- `backend/services/sentiment_service.py`
- `backend/services/translation_service.py`
- `backend/services/llm_client.py`
- `backend/services/prompt_service.py`

Frontend:

- `frontend/index.html`

## Benchmark and Evaluation

The repository includes a compact course-style benchmark:

- `datasets/smart_meeting_benchmark_v1/`

Task files:

- `dataset_summarization.json`
- `dataset_action_items.json`
- `dataset_sentiment_engagement.json`
- `dataset_translation_multilingual.json`
- `audio_manifest.json`

Evaluation entrypoint:

```bash
cd CS6493
conda run -n cs6493 python scripts/evaluate.py --split dev --tasks summarization action_items sentiment translation
conda run -n cs6493 python scripts/evaluate.py --split test --tasks summarization action_items sentiment translation
```

Final benchmark snapshot:

- Dev
  - summarization: `key_topics F1 1.0`, `decisions F1 1.0`, `blockers F1 1.0`, `next_steps F1 1.0`
  - action items: `F1 1.0`
  - sentiment: `overall_sentiment_accuracy 1.0`, `engagement_level_accuracy 1.0`
  - translation: `line_similarity 1.0`, `speaker/timestamp preservation 1.0`
- Test
  - summarization: `key_topics F1 0.8`, `decisions F1 1.0`, `blockers F1 1.0`, `next_steps F1 1.0`
  - action items: `F1 1.0`
  - sentiment: `agreement/disagreement/tension F1 1.0`
  - translation: `line_similarity 0.8693`, `speaker/timestamp preservation 1.0`

## Runtime Configuration

Final model choice:

- `Qwen2.5-7B FP16`

Recommended runtime split:

- vLLM model server on GPU 7
- ASR and diarization on GPU 3

This avoids GPU contention between generation and audio processing.

## Installation

### 1. Start the model server

```bash
bash /home/jiahuning2/LLM_Ability_Test/models/Qwen2.5-7B/start_vllm_fp16.sh
```

### 2. Install Python dependencies

```bash
cd /home/jiahuning2/LLM_Ability_Test/CS6493
conda run -n cs6493 pip install -r requirements.txt
```

### 3. Configure environment

Copy `.env.example` to `.env` if needed and adjust values:

```bash
cp .env.example .env
```

Relevant settings:

```env
LLM_PROFILE=7b-fp16
VLLM_BASE_URL=http://127.0.0.1:8400/v1
VLLM_API_KEY=EMPTY
VLLM_MODEL=qwen2.5-7b-fp16
HOST=0.0.0.0
PORT=6493
ASR_MODEL=medium
ASR_DEVICE=cuda:3
DIARIZATION_DEVICE=cuda:3
```

## Running the App

Start the backend:

```bash
cd /home/jiahuning2/LLM_Ability_Test/CS6493
bash run_backend.sh
```

Then open:

- App: `http://127.0.0.1:6493`
- API docs: `http://127.0.0.1:6493/docs`

## Project Structure

```text
CS6493/
├── backend/
│   ├── api/
│   ├── core/
│   ├── models/
│   └── services/
├── frontend/
├── datasets/
├── docs/
├── scripts/
├── tests/
└── tools/
```

## Notes

- The production UI no longer exposes internal demo-recording controls.
- The project is optimized around stable structured outputs rather than raw generative text.
- Prompt engineering, schema normalization, and fallback logic are all part of the final system design.
