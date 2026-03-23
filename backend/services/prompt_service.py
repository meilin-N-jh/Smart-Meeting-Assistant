"""Prompt templates for LLM tasks."""

from typing import Dict

# Translation prompt template
TRANSLATION_TEMPLATE = """You are a multilingual meeting assistant.
Translate the following meeting content from {source_lang} to {target_lang}.

Rules:
1. Preserve speaker labels and timestamps exactly if they appear.
2. Preserve line breaks and paragraph structure.
3. Keep commitments, dates, and task ownership accurate.
4. Return only translated text, with no explanations.

Original:
{text}

Translation:"""

# Summarization prompt template
SUMMARIZATION_TEMPLATE = """You are a professional meeting assistant.
Analyze the following meeting transcript and produce a structured summary for participants who did not attend.

Meeting Transcript:
{transcript}

Provide a summary in JSON format with the following structure:
{{
    "title": "Meeting Title",
    "overview": "High-level summary of the meeting in 2-4 sentences",
    "key_topics": ["Distinct topic 1", "Distinct topic 2"],
    "decisions": ["Concrete decision 1", "Concrete decision 2"],
    "blockers": ["Open risk or blocker 1"],
    "next_steps": ["Follow-up step with owner/deadline if known"],
    "concise_summary": "Short executive-style recap in 3-6 sentences"
}}

Rules:
- Focus on synthesis, not verbatim transcript copying.
- Prefer concrete outcomes over generic statements.
- If no explicit decision exists, keep decisions as [].
- Keep each list item concise and non-duplicated.

Respond ONLY with valid JSON."""

RECONSTRUCTIVE_NOTES_TEMPLATE = """You are reconstructing a meeting before summarization.
Turn the transcript chunk into compact structured meeting notes that preserve discussion flow,
speaker intent, decisions, blockers, disagreements, and follow-ups.

Meeting Transcript Chunk:
{transcript}

Return JSON only:
{{
    "segment_summary": "2-4 sentence reconstruction of what happened in this chunk",
    "topic_labels": ["Topic 1", "Topic 2"],
    "discussion_points": ["Important discussion point 1", "Important discussion point 2"],
    "decisions": ["Decision made in this chunk"],
    "blockers": ["Open blocker or risk"],
    "next_steps": ["Follow-up step with owner/deadline if present"],
    "interaction_signals": ["agreement/disagreement/hesitation signal with speaker if clear"]
}}

Rules:
- Reconstruct the conversation into clean notes; do not copy long transcript spans.
- Preserve ownership, deadlines, and unresolved questions.
- Keep each list concise and non-duplicated.

Respond ONLY with valid JSON."""

RECONSTRUCTIVE_REDUCE_TEMPLATE = """You are a meeting summarizer working from reconstructed notes.
Synthesize the following chunk reconstructions into one final meeting summary.

Reconstructed Notes:
{notes}

Return JSON only:
{{
    "title": "Meeting Title",
    "overview": "High-level summary of the full meeting in 2-4 sentences",
    "key_topics": ["Distinct topic 1", "Distinct topic 2"],
    "decisions": ["Concrete decision 1", "Concrete decision 2"],
    "blockers": ["Open risk or blocker 1"],
    "next_steps": ["Follow-up step with owner/deadline if known"],
    "concise_summary": "Short executive-style recap in 3-6 sentences"
}}

Rules:
- Consolidate across chunks rather than repeating each chunk.
- Prefer decisions and next steps that matter to participants after the meeting.
- Preserve important unresolved tension or risk if present.

Respond ONLY with valid JSON."""

# Action items extraction template
ACTION_ITEMS_TEMPLATE = """You are a meeting action item assistant.
Analyze the following transcript and extract concrete commitments, follow-ups, and assigned tasks.

Meeting Transcript:
{transcript}

Extract all action items in JSON format:
{{
    "action_items": [
        {{
            "id": 1,
            "assignee": "Person name/speaker label or 'unknown' if unclear",
            "task": "Specific deliverable/action (not vague)",
            "deadline": "Deadline phrase if mentioned, otherwise null",
            "priority": "high/medium/low",
            "source_text": "The exact supporting quote from transcript",
            "confidence": 0.0-1.0
        }}
    ]
}}

Rules:
- Include only real commitments or task assignments.
- Do not extract background discussion as action items.
- Keep task phrasing concise and execution-oriented.

Respond ONLY with valid JSON."""

# Sentiment analysis template
SENTIMENT_TEMPLATE = """You are a meeting sentiment and engagement assistant.
Analyze emotional dynamics and interaction quality from the transcript.
This is not generic sentiment classification. Focus on meeting interaction signals.

Meeting Transcript:
{transcript}

Provide analysis in JSON format:
{{
    "overall_sentiment": "positive/negative/neutral/mixed",
    "engagement_level": "high/medium/low",
    "emotional_moments": [
        {{
            "timestamp": "approx time in meeting",
            "description": "Description of emotional moment",
            "speaker": "Speaker label",
            "sentiment": "positive/negative/neutral"
        }}
    ],
    "agreements": [
        {{
            "speaker": "Speaker label",
            "statement": "What they agreed to",
            "evidence": "Exact quote"
        }}
    ],
    "disagreements": [
        {{
            "speaker": "Speaker label",
            "statement": "What they disagreed with",
            "evidence": "Exact quote"
        }}
    ],
    "tension_points": [
        {{
            "speakers": ["Speaker 1", "Speaker 2"],
            "topic": "Topic of tension",
            "evidence": "Exact quote"
        }}
    ],
    "hesitation_signals": [
        {{
            "speaker": "Speaker label",
            "signal": "Hesitation indicator",
            "evidence": "Exact quote"
        }}
    ],
    "speaker_signals": [
        {{
            "speaker": "Speaker label",
            "agreement_count": 0,
            "disagreement_count": 0,
            "hesitation_count": 0,
            "emotional_moment_count": 0,
            "tension_involvement_count": 0,
            "dominant_signal": "supportive/concerned/hesitant/mixed/neutral"
        }}
    ],
    "evidence_quotes": ["Quote 1", "Quote 2"],
    "recommendations": ["Facilitator recommendation 1", "Facilitator recommendation 2"]
}}

Rules:
- Focus on interaction signals: agreement, disagreement, tension, hesitation, emotional shifts.
- Prefer evidence-backed observations over vague labels.
- If a speaker clearly supports a proposal, put it under agreements.
- If a speaker raises concern, skepticism, or pushback, put it under disagreements or tension_points.
- If a speaker sounds uncertain or non-committal, put it under hesitation_signals.
- Include emotionally significant moments whenever someone strongly agrees, objects, hesitates, or surfaces risk.
- Include evidence quotes for major claims whenever possible.
- Keep recommendations practical for the next meeting.

Respond ONLY with valid JSON."""

SENTIMENT_STRICT_TEMPLATE = """You are a meeting interaction analyst.
Your job is to identify concrete interaction signals in a meeting transcript.
Do NOT return only broad labels like neutral or medium without evidence.

Meeting Transcript:
{transcript}

Return JSON with this exact structure:
{{
    "overall_sentiment": "positive/negative/neutral/mixed",
    "engagement_level": "high/medium/low",
    "emotional_moments": [
        {{
            "timestamp": "approx time in meeting",
            "description": "Emotionally significant moment",
            "speaker": "Speaker label",
            "sentiment": "positive/negative/neutral"
        }}
    ],
    "agreements": [{{"speaker": "Speaker label", "statement": "summary", "evidence": "quote"}}],
    "disagreements": [{{"speaker": "Speaker label", "statement": "summary", "evidence": "quote"}}],
    "tension_points": [{{"speakers": ["Speaker 1", "Speaker 2"], "topic": "topic", "evidence": "quote"}}],
    "hesitation_signals": [{{"speaker": "Speaker label", "signal": "hesitation", "evidence": "quote"}}],
    "speaker_signals": [
        {{
            "speaker": "Speaker label",
            "agreement_count": 0,
            "disagreement_count": 0,
            "hesitation_count": 0,
            "emotional_moment_count": 0,
            "tension_involvement_count": 0,
            "dominant_signal": "supportive/concerned/hesitant/mixed/neutral"
        }}
    ],
    "evidence_quotes": ["quote 1", "quote 2"],
    "recommendations": ["recommendation 1"]
}}

Strict requirements:
- Extract at least 2 interaction signals if the transcript contains them.
- Use speaker labels from the transcript whenever possible.
- Every disagreement, hesitation, or tension point should include evidence.
- If the meeting is mostly calm and informational, still identify any notable agreement, caution, or uncertainty signals you can find.

Respond ONLY with valid JSON."""

# Long meeting chunk processing template
CHUNK_SUMMARIZATION_TEMPLATE = """Summarize this portion of a meeting. This is part of a longer meeting.

Meeting Transcript Chunk:
{transcript}

Provide a concise summary of this chunk:
{{
    "summary": "Summary of this chunk",
    "key_points": ["Point 1", "Point 2"],
    "decisions": ["Decision 1"],
    "action_items": ["Action 1"]
}}

Respond ONLY with valid JSON."""

# Meeting title extraction template
TITLE_EXTRACTION_TEMPLATE = """Extract a concise title for this meeting based on the transcript.

Meeting Transcript:
{transcript}

Provide a title in JSON format:
{{
    "title": "Meeting Title"
}}

Respond ONLY with valid JSON."""


# Template registry
PROMPT_TEMPLATES: Dict[str, str] = {
    "translation": TRANSLATION_TEMPLATE,
    "summarization": SUMMARIZATION_TEMPLATE,
    "reconstructive_notes": RECONSTRUCTIVE_NOTES_TEMPLATE,
    "reconstructive_reduce": RECONSTRUCTIVE_REDUCE_TEMPLATE,
    "action_items": ACTION_ITEMS_TEMPLATE,
    "sentiment": SENTIMENT_TEMPLATE,
    "sentiment_strict": SENTIMENT_STRICT_TEMPLATE,
    "chunk_summarization": CHUNK_SUMMARIZATION_TEMPLATE,
    "title_extraction": TITLE_EXTRACTION_TEMPLATE,
}


def get_prompt_template(name: str) -> str:
    """Get a prompt template by name.

    Args:
        name: Template name

    Returns:
        Template string

    Raises:
        ValueError: If template not found
    """
    if name not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown prompt template: {name}")
    return PROMPT_TEMPLATES[name]


def list_prompt_templates() -> list:
    """List all available prompt templates.

    Returns:
        List of template names
    """
    return list(PROMPT_TEMPLATES.keys())
