# Smart Meeting Benchmark v1

This folder contains the project evaluation dataset for the Smart Meeting Assistant.
It follows the same task split used in `dataset_make.zip`, but converts the templates
into a project-ready benchmark with train/dev/test partitions.

## Files

- `dataset_summarization.json`: structured summary benchmark
- `dataset_action_items.json`: action item extraction benchmark
- `dataset_sentiment_engagement.json`: meeting dynamics benchmark
- `dataset_translation_multilingual.json`: multilingual translation benchmark
- `audio_manifest.json`: references to local audio assets already in this repo

## Intended Usage

- `train` cases:
  - prompt drafting
  - few-shot example selection
  - schema debugging
- `dev` cases:
  - prompt iteration
  - fallback rule tuning
  - threshold selection
- `test` cases:
  - final report numbers only
  - do not hand-tune prompts on these cases

## Recommended Experiment Flow

1. Start with zero-shot prompts on all `dev` cases.
2. Use `train` cases marked as `few_shot_candidate=true` to build few-shot prompts.
3. Tune only on `train` and `dev`.
4. Freeze prompts and evaluate once on `test`.

## Coverage

The benchmark intentionally includes:

- single-language English meetings
- Chinese and mixed-language meetings
- clear action items and ambiguous assignments
- supportive discussions, hesitation, and tension
- short structured meetings and multi-topic meetings

## Audio References

`audio_manifest.json` points to the local generated meeting audio already checked into the repo.
Use it for ASR and speaker-diarization demos; use the task-specific JSON files for NLP evaluation.
