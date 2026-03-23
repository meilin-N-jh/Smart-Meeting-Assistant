"""Generate a synthetic 3-person meeting audio sample for local testing."""

from __future__ import annotations

import asyncio
from pathlib import Path

import imageio_ffmpeg
from edge_tts import Communicate
from pydub import AudioSegment


OUT_DIR = Path("/home/jiahuning2/LLM_Ability_Test/CS6493/test_assets/generated_audio")
TRANSCRIPT_PATH = OUT_DIR / "three_person_meeting_transcript.txt"
WAV_PATH = OUT_DIR / "three_person_meeting_3min.wav"
MP3_PATH = OUT_DIR / "three_person_meeting_3min.mp3"


SPEAKERS = {
    "Emma": {"voice": "en-US-AriaNeural", "rate": "+18%"},
    "Daniel": {"voice": "en-US-GuyNeural", "rate": "+16%"},
    "Maya": {"voice": "en-GB-SoniaNeural", "rate": "+14%"},
}


DIALOGUE = [
    ("Emma", "Thanks for joining, everyone. Today we need to lock the timeline for the customer analytics dashboard and confirm what goes into the April release."),
    ("Daniel", "I reviewed the current engineering status this morning. The data ingestion service is stable, but the permissions panel still has two blocking bugs."),
    ("Maya", "From the design side, the main dashboard and drill down flow are ready. I still need feedback on the mobile layout and the export interaction."),
    ("Emma", "Okay, so the dashboard itself is on track, but permissions and some usability details are still open. Let's start with launch scope."),
    ("Daniel", "If we keep role based permissions, CSV export, and the three core charts, we can likely ship by April eighteenth. If we add custom alert rules, I would push the release by at least one more week."),
    ("Maya", "I agree with that tradeoff. Alert rules are valuable, but they also touch settings, notifications, and empty states, so the user experience work is bigger than it first appears."),
    ("Emma", "Then let's treat custom alerts as phase two unless someone has a strong objection."),
    ("Daniel", "No objection from me. I would rather release a stable version than carry too many moving pieces into QA."),
    ("Maya", "Same here. We can still mention alerts in the roadmap slide so stakeholders know they are coming soon."),
    ("Emma", "Great. Decision one: April release includes dashboard, filters, role based permissions, and CSV export. Custom alerts move to the following sprint."),
    ("Emma", "Next, I want to talk about risk. Daniel, what exactly is blocking the permissions panel?"),
    ("Daniel", "There are two issues. First, admin users sometimes lose access after a role update because the cache is not invalidated correctly. Second, audit log entries are delayed by a few seconds, which fails one of the security acceptance checks."),
    ("Emma", "How confident are you that both can be fixed this week?"),
    ("Daniel", "The cache issue is straightforward. I can fix and test that by Wednesday. The audit log delay is trickier because it touches the message queue worker. I think I can finish by Friday, but I am not fully certain."),
    ("Maya", "If the audit log is still unstable by Friday, should we hide that section in the admin panel and keep the rest of the permissions workflow available?"),
    ("Daniel", "Technically yes, that would reduce risk. We would need a small copy change and one conditional check in the UI."),
    ("Emma", "Let's keep that as the fallback plan. Maya, please prepare a hidden state for the audit log section just in case."),
    ("Maya", "Sure. I will send a revised permissions mockup by tomorrow afternoon, including the fallback state and the mobile layout notes."),
    ("Emma", "Thank you. Sales also wants a one page release brief next Monday, so I need screenshots and a short list of supported use cases."),
    ("Maya", "I can provide the screenshots by Thursday evening after I finish the mobile spacing review."),
    ("Daniel", "I will send the technical summary by Thursday noon, including refresh timing and the permissions matrix."),
    ("Emma", "Good. One final concern is responsiveness in the meeting assistant demo. If transcript updates come too slowly, people may think the system is stuck."),
    ("Daniel", "I am not convinced we can make it perfectly word by word this week, but we can make it feel live by sending shorter updates and keeping the analysis panels in the background."),
    ("Maya", "That is fine for the demo. I will simplify the top of the interface so the transcript becomes the visual focus."),
    ("Emma", "Excellent. Final recap: Daniel fixes cache invalidation by Wednesday and sends the support summary by Thursday noon. Maya sends updated mockups tomorrow and screenshots by Thursday evening. I will draft the sales brief on Monday."),
    ("Daniel", "That matches my notes."),
    ("Maya", "Same here."),
    ("Emma", "Great, thanks both. Let's do a quick async check in on Wednesday morning and decide whether the fallback is needed."),
]


async def synthesize_line(speaker: str, text: str, mp3_path: Path) -> None:
    cfg = SPEAKERS[speaker]
    communicate = Communicate(text=text, voice=cfg["voice"], rate=cfg["rate"])
    await communicate.save(str(mp3_path))


async def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
    AudioSegment.ffmpeg = AudioSegment.converter
    AudioSegment.ffprobe = AudioSegment.converter

    transcript_lines = []
    combined = AudioSegment.silent(duration=500)

    for idx, (speaker, text) in enumerate(DIALOGUE, start=1):
        part_path = OUT_DIR / f"segment_{idx:02d}_{speaker.lower()}.mp3"
        await synthesize_line(speaker, text, part_path)
        segment = AudioSegment.from_file(part_path, format="mp3")

        if speaker == "Emma":
            segment = segment + 1
        elif speaker == "Daniel":
            segment = segment - 1
        else:
            segment = segment + 2

        combined += segment
        combined += AudioSegment.silent(duration=220 if idx % 4 else 380)
        transcript_lines.append(f"{speaker}: {text}")

    combined.export(MP3_PATH, format="mp3")
    combined.export(WAV_PATH, format="wav")
    TRANSCRIPT_PATH.write_text("\n".join(transcript_lines) + "\n")

    duration_sec = len(combined) / 1000.0
    print(f"Wrote {WAV_PATH}")
    print(f"Wrote {MP3_PATH}")
    print(f"Wrote {TRANSCRIPT_PATH}")
    print(f"Duration: {duration_sec:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
