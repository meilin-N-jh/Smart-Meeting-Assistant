"""Prompt templates for LLM tasks."""

from typing import Dict

# Translation prompt template
TRANSLATION_TEMPLATE = """You are a multilingual meeting assistant.
Translate the following meeting content from {source_lang} to {target_lang}.

Rules:
1. Preserve speaker labels and timestamps exactly if they appear.
2. Preserve line breaks and paragraph structure.
3. Keep commitments, dates, and task ownership accurate.
4. If the input line format is `[timestamp] Speaker: utterance`, keep `[timestamp] Speaker:` unchanged and translate only `utterance`.
5. Do not repeat, translate, or prepend the speaker label inside the translated utterance.
6. If an utterance is already in the target language, keep it as-is.
7. Return only translated text, with no explanations.

Original:
{text}

Translation:"""

# Summarization prompt template
SUMMARIZATION_TEMPLATE = """You are a professional meeting assistant.
Return exactly one JSON object summarizing the meeting transcript below.

Meeting Transcript:
{transcript}

Use exactly this schema:
{{
    "title": "short meeting title",
    "overview": "2-3 sentence overview",
    "key_topics": ["short topic phrase"],
    "decisions": ["explicit decision only"],
    "blockers": ["unresolved risk or blocker"],
    "next_steps": ["future action or follow-up"],
    "concise_summary": "short executive recap"
}}

Rules:
- Output JSON only. No markdown. No explanation.
- `decisions` must contain only explicit meeting decisions.
- `blockers` must contain unresolved issues, risks, or dependencies.
- `next_steps` must contain concrete future actions.
- Use [] for any empty list field.
- Keep list items concise and non-duplicated.
- Use short noun phrases for `key_topics`.
- For multilingual transcripts, write every field in concise English.
- Make `key_topics` specific meeting concepts, such as "payment patch incident", "rollback decision", or "timeout investigation", not generic fragments like "issue" or "discussion".
- Write `blockers` as unresolved statements, for example: "The timeout issue causing the error spike is not isolated yet."
- Put only one future action in each `next_steps` item. Split separate actions into separate list items.
- Put items such as "prepare", "review", "follow up", or "investigate" under `next_steps`, not `decisions`, unless the transcript explicitly says that action itself was approved as a decision.
- Keep `decisions` for statements like "we decided", "today's decision is", "approved", or "finalized".
- In `decisions`, keep the main object from the transcript and prefer wording like "Roll back the payment patch immediately." instead of shorter phrases like "rollback the patch now".

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
- Keep conditional plans explicit. For example, "ship Tuesday if legal approves Monday" should stay conditional.
- If someone raises concern or disagreement, include it in blockers or interaction_signals rather than hiding it.

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
- If no explicit meeting decision exists, keep decisions as [].

Respond ONLY with valid JSON."""

# Action items extraction template
ACTION_ITEMS_TEMPLATE = """You are a meeting action item assistant.
Analyze the following transcript and extract concrete commitments, follow-ups, and assigned tasks.

Meeting Transcript:
{transcript}

Return exactly one JSON object in this format:
{{
    "action_items": [
        {{
            "id": 1,
            "assignee": "Person name/speaker label or 'unknown' if unclear",
            "task": "Short normalized verb phrase",
            "deadline": "Exact deadline phrase if mentioned, otherwise null",
            "priority": "high/medium/low",
            "source_text": "The exact supporting quote from transcript",
            "confidence": 0.0-1.0
        }}
    ]
}}

Rules:
- Include only real commitments or task assignments.
- Do not extract background discussion as action items.
- Return JSON only. No markdown. No explanation.
- `task` must be a concise action phrase, not a full sentence. Good: "Follow up with the vendor". Bad: "Someone should follow up with the vendor this week."
- Preserve deadline phrases exactly when they appear, such as "this week", "tonight", "tomorrow morning", "by Friday", or "next Tuesday".
- If the transcript says "someone should", "we should", or gives a task without a confirmed owner, set `assignee` to "unknown".
- If a manager assigns a task and the assignee later accepts it, prefer the acceptance or commitment sentence as `source_text`.
- If there is no acceptance sentence, use the clearest assignment sentence as `source_text`.
- Keep one task per item and avoid duplicates.
- Use `medium` unless urgency is explicit.
- Use `null` for missing deadlines.

Examples:
Transcript line: "Someone should follow up with the vendor this week."
Output item:
{{"id": 1, "assignee": "unknown", "task": "Follow up with the vendor", "deadline": "this week", "priority": "medium", "source_text": "Someone should follow up with the vendor this week.", "confidence": 0.82}}

Transcript lines:
- "Bob, update the roadmap tonight."
- "Okay, I will update it tonight."
Output item:
{{"id": 1, "assignee": "Bob", "task": "Update the roadmap", "deadline": "tonight", "priority": "medium", "source_text": "Okay, I will update it tonight.", "confidence": 0.93}}

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
- Return JSON only. No markdown. No explanation.
- Focus on interaction signals: agreement, disagreement, tension, hesitation, emotional shifts.
- Prefer evidence-backed observations over vague labels.
- If a speaker clearly supports a proposal, put it under agreements.
- If a speaker raises concern, skepticism, or pushback, put it under disagreements or tension_points.
- If a speaker sounds uncertain or non-committal, put it under hesitation_signals.
- Include emotionally significant moments whenever someone strongly agrees, objects, hesitates, or surfaces risk.
- Include evidence quotes for major claims whenever possible.
- Keep recommendations practical for the next meeting.
- Use `engagement_level="high"` when multiple speakers actively react to the same decision, risk, or conflict.
- Use `overall_sentiment="mixed"` when the meeting contains both forward pressure and explicit concern, objection, or risk escalation.
- For disagreement, capture pushback against a proposal, timeline, or decision even if the speaker does not literally say "I disagree".
- In `statement`, `description`, `topic`, and `recommendations`, write concise English summaries even if the evidence quote is multilingual.
- For tension_points, summarize the contested issue, such as schedule pressure versus product risk.
- When a speaker says they share another person's concern, record that under `agreements` and keep the concern itself under `disagreements` or `tension_points`.
- In `disagreements`, list the speaker(s) who push back or raise the concern, not the speaker defending the original plan.
- If the transcript is a calm status check with little reaction, it is acceptable for signal lists to stay empty.

Example pattern:
- "Let's commit to the current launch date."
- "I am not convinced this will work if testing slips again."
- "I share that concern. The current timeline still feels risky."
Expected interpretation:
- `overall_sentiment`: "mixed"
- `engagement_level`: "high"
- `agreements`: support for the concern
- `disagreements`: pushback on the launch decision
- `tension_points`: launch timing versus testing risk

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
- Mark `engagement_level` as `high` when several speakers respond to the same contested topic.
- Mark `overall_sentiment` as `mixed` when urgency or support coexists with concern, risk, or resistance.
- Treat phrases like "I am worried", "I am not convinced", "too risky", "too high", and "I share that concern" as strong interaction signals.
- If one speaker supports another person's concern, include that under `agreements`.
- If people are split between staying on schedule and reducing risk, include a `tension_points` item about schedule versus quality/risk.
- Write `statement`, `description`, `topic`, and `recommendations` in concise English, even if the quote itself is Chinese or mixed-language.
- In `disagreements`, identify the speaker raising the objection or concern. Do not use the speaker defending the current plan as the disagreement entry unless they are explicitly objecting to someone else.

Example output style for a conflict-heavy meeting:
{{
    "overall_sentiment": "mixed",
    "engagement_level": "high",
    "emotional_moments": [
        {{
            "timestamp": "00:01",
            "description": "QA raises a concrete quality risk that shifts the meeting tone.",
            "speaker": "QA",
            "sentiment": "negative"
        }}
    ],
    "agreements": [
        {{
            "speaker": "Engineering Lead",
            "statement": "Engineering Lead agrees with QA's concern about launch risk.",
            "evidence": "我同意 QA 的担心，如果今天不解决这个问题，周五上线太冒险了。"
        }}
    ],
    "disagreements": [
        {{
            "speaker": "QA",
            "statement": "QA pushes back on keeping the current launch date because crash risk remains high.",
            "evidence": "I am worried because the crash rate is still too high."
        }},
        {{
            "speaker": "Engineering Lead",
            "statement": "Engineering Lead also pushes back on launching Friday until the issue is fixed.",
            "evidence": "我同意 QA 的担心，如果今天不解决这个问题，周五上线太冒险了。"
        }}
    ],
    "tension_points": [
        {{
            "speakers": ["PM", "QA", "Engineering Lead"],
            "topic": "Launch timing versus product quality risk",
            "evidence": "I am worried because the crash rate is still too high."
        }}
    ],
    "hesitation_signals": [],
    "speaker_signals": [],
    "evidence_quotes": [
        "I am worried because the crash rate is still too high."
    ],
    "recommendations": [
        "Resolve the crash-rate blocker before confirming the launch date."
    ]
}}

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
