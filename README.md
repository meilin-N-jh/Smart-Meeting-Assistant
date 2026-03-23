# Smart Meeting Assistant

A comprehensive AI-powered meeting assistant that provides real-time transcription, summarization, translation, action item extraction, and sentiment analysis using local LLMs via vLLM. The default runtime profile is `Qwen2.5-7B FP16`, selected after local latency checks for this workflow. `Qwen2.5-14B-Instruct-AWQ` remains available when you want a stronger but slower model.

## Features

### 1. Real-time Speech-to-Text Transcription
- Capture live microphone input
- Support for uploaded audio files (WAV, MP3, M4A)
- Timestamped transcript output
- Speaker diarization and labeling
- Multiple speaker identification

### 2. Automatic Meeting Summarization
- Key topics extraction
- Decisions identification
- Blockers detection
- Next steps extraction
- Concise summary generation

### 3. Machine Translation
- Translate transcript and summary
- Preserve timestamps and speaker labels
- Support for English, Chinese, Japanese
- Extensible to other languages

### 4. Context-Aware Action Item Extraction
- Detect commitments and tasks
- Extract assignee, task, deadline, priority
- Handle ambiguous assignee with "unknown"
- Confidence scoring

### 5. Meeting Sentiment and Engagement Analysis
- Overall sentiment detection
- Engagement level assessment
- Agreement/disagreement detection
- Tension point identification
- Evidence quotes extraction

## Technology Stack

### Backend
- **FastAPI**: Modern Python web framework
- **vLLM**: Local LLM inference engine
- **Qwen2.5-7B FP16**: Default local language model for speed
- **Qwen2.5-14B-AWQ**: Higher-capacity alternative when latency is less important
- **Qwen2.5-7B-AWQ**: Lighter fallback model for constrained environments
- **faster-whisper**: Speech-to-text
- **whisperx**: Speaker diarization

### Frontend
- HTML/CSS/JavaScript (vanilla)
- Real-time microphone recording
- File upload support

## Project Structure

```
CS6493/
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py          # API endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py          # Configuration
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py         # Pydantic models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llm_client.py      # LLM client
│   │   ├── asr_service.py    # Speech-to-text
│   │   ├── diarization_service.py
│   │   ├── translation_service.py
│   │   ├── summarization_service.py
│   │   ├── action_items_service.py
│   │   ├── sentiment_service.py
│   │   └── prompt_service.py
│   └── main.py                # FastAPI app
├── frontend/
│   └── index.html             # Demo frontend
├── tests/
│   ├── __init__.py
│   ├── test_services.py
│   └── test_api.py
├── sample_data/
│   ├── sample_meeting_transcript.txt
│   ├── sample_meeting_zh.txt
│   └── sample_meeting_output.json
├── config/
├── prompts/
├── requirements.txt
├── .env.example
└── README.md
```

## Prerequisites

### 1. vLLM Model Server

Recommended default: start `Qwen2.5-7B FP16` in the `qwen2.5` environment.

```bash
bash /home/jiahuning2/LLM_Ability_Test/models/Qwen2.5-7B/start_vllm_fp16.sh
```

Higher-capacity alternative:

```bash
bash /home/jiahuning2/LLM_Ability_Test/models/Qwen2.5-14B/start_vllm_awq.sh
```

Manual environment variables for the default profile:

```bash
export LLM_PROFILE=7b-fp16
export VLLM_BASE_URL=http://127.0.0.1:8400/v1
export VLLM_API_KEY=EMPTY
export VLLM_MODEL=qwen2.5-7b-fp16
```

### 2. Python Environment

```bash
# Create conda environment
conda create -n qwen2.5 python=3.10
conda activate qwen2.5

# Install dependencies
cd /home/jiahuning2/LLM_Ability_Test/CS6493
pip install -r requirements.txt
```

### 3. Additional Dependencies (Optional)

For full ASR and diarization support:

```bash
pip install faster-whisper
pip install whisperx
# For pyannote (requires HF token):
# pip install pyannote.audio
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# LLM runtime profile
LLM_PROFILE=7b-fp16

# vLLM Configuration
VLLM_BASE_URL=http://127.0.0.1:8400/v1
VLLM_API_KEY=EMPTY
VLLM_MODEL=qwen2.5-7b-fp16

# Server
HOST=0.0.0.0
PORT=6493
DEBUG=false

# ASR (optional)
ASR_MODEL=medium
ASR_DEVICE=cuda

# Audio
AUDIO_CHUNK_DURATION=5
AUDIO_SAMPLE_RATE=16000
```

## Running the Application

### 1. Start vLLM (First)

Make sure vLLM is running with the selected profile. Default:

```bash
# In one terminal
cd /home/jiahuning2/LLM_Ability_Test/models/Qwen2.5-7B
bash start_vllm_fp16.sh
```

### 2. Start the Meeting Assistant

```bash
# In another terminal
cd /home/jiahuning2/LLM_Ability_Test/CS6493
conda run -n cs6493 env LLM_PROFILE=7b-fp16 python -m backend.main
```

Or with uvicorn:

```bash
cd /home/jiahuning2/LLM_Ability_Test/CS6493
conda run -n qwen2.5 uvicorn backend.main:app --host 0.0.0.0 --port 6493
```

### 3. Access the Demo

Open your browser:
- Frontend: http://localhost:6493
- API Docs: http://localhost:6493/docs

## API Usage

### Health Check

```bash
curl http://localhost:6493/api/health
```

### Transcribe Audio File

```bash
curl -X POST http://localhost:6493/api/transcribe/file \
  -F "file=@meeting.wav" \
  -F "language=en"
```

### Translate Text

```bash
curl -X POST http://localhost:6493/api/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello everyone, welcome to the meeting",
    "source_lang": "en",
    "target_lang": "zh"
  }'
```

### Summarize Meeting

```bash
curl -X POST http://localhost:6493/api/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "Alice: Good morning... Bob: Thank you..."
  }'
```

### Extract Action Items

```bash
curl -X POST http://localhost:6493/api/action-items \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "Bob will finish the report by Friday..."
  }'
```

### Analyze Sentiment

```bash
curl -X POST http://localhost:6493/api/sentiment \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "I agree with this approach..."
  }'
```

### Full Pipeline

```bash
curl -X POST http://localhost:6493/api/process-meeting \
  -H "Content-Type: application/json" \
  -d '{
    "input_type": "text",
    "text": "Meeting transcript here...",
    "translate_to": "zh"
  }'
```

## Demo Modes

### Demo A: Audio File Upload
Upload an audio file and get:
- Full transcription with speaker labels
- Translation (if requested)
- Summary
- Action items
- Sentiment analysis

### Demo B: Text Transcript Input
Paste existing transcript text and get:
- Translation
- Summary
- Action items
- Sentiment analysis

### Demo C: Microphone Input
Record from microphone and get:
- Real-time transcription
- Speaker identification
- Full analysis pipeline

## Sample Data

Test with sample data:

```bash
# Test with sample transcript
cat sample_data/sample_meeting_transcript.txt

# Test with Chinese sample
cat sample_data/sample_meeting_zh.txt

# Sample output
cat sample_data/sample_meeting_output.json
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_services.py -v

# Run with coverage
pytest tests/ --cov=backend --cov-report=html
```

## Error Handling

The application includes comprehensive error handling:
- Invalid input validation
- LLM response parsing with fallback
- Service availability checks
- Detailed error messages

## Known Limitations

1. **Microphone Recording**: Requires HTTPS or localhost for browser MediaRecorder
2. **Long Meetings**: Processed in chunks for memory efficiency
3. **Speaker Diarization**: Requires either whisperx or pyannote.audio
4. **Offline Mode**: Works without internet once dependencies are installed

## Future Optimizations

1. WebSocket streaming for real-time transcription
2. Multi-language support expansion
3. Custom prompt templates
4. Meeting history storage
5. Export to various formats (PDF, DOCX)
6. Integration with calendar APIs

## License

MIT License

## Contact

For issues and questions, please open an issue on the project repository.
