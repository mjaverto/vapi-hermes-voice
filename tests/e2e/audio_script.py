"""The callee's side of the call: scripted utterances, exact silence gaps, real audio.

The gaps are the instrument. Every deadline in ``deadlines.py`` is measured from the
moment the callee stopped talking, so the script must leave enough silence after each
utterance for the *failure* to be observable: a 6 s gap cannot see a 10 s stall, it can
only truncate it into the next turn and turn a clear failure into a confusing one.

Audio is synthesised locally with macOS ``say`` + ``afconvert`` (or ``ffmpeg``) into
16 kHz mono pcm_s16le, which is what the Vapi websocket transport is opened with. Files
are cached under ``tests/e2e/audio/`` and keyed by content, so a re-run costs nothing
and two runs stream byte-identical audio.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

__all__ = ["SAMPLE_RATE", "Scenario", "Step", "SCENARIOS", "AudioUnavailable", "synthesize"]

SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2
DEFAULT_VOICE = "Samantha"
AUDIO_DIR = Path(__file__).parent / "audio"


class AudioUnavailable(RuntimeError):
    """No local text-to-speech toolchain to build the callee's audio with."""


@dataclass(frozen=True, slots=True)
class Step:
    """One scripted utterance and the silence that follows it.

    ``gap_after_s`` is silence the harness streams *after* the utterance. It has to
    exceed the deadline being measured by a healthy margin, otherwise a breach is
    clipped by the next utterance instead of measured.
    """

    text: str
    gap_after_s: float
    label: str


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    description: str
    steps: tuple[Step, ...]
    lead_silence_s: float = 0.6
    tail_silence_s: float = 1.0
    # Which callee turn the R2 acknowledgement is expected on; -1 = the last one, and
    # None = this script asks nothing that needs a lookup, so none is due.
    ack_turn_index: int | None = -1

    @property
    def scripted_seconds(self) -> float:
        """Silence budget only; speech length is whatever the synthesiser produces."""
        return self.lead_silence_s + self.tail_silence_s + sum(s.gap_after_s for s in self.steps)


# The callee says "Hello?", waits long enough for a 10 s stall to be visible, then asks
# something that forces a Hermes tool round trip (which is what makes an acknowledgement
# necessary in the first place), then waits long enough to prove the 10 s cooldown.
DEADLINES = Scenario(
    name="deadlines",
    description="R1 reason-for-calling deadline, then R2 acknowledgement deadline + cooldown",
    steps=(
        # 12 s: the reported failure was "10+ seconds of silence", and a window that
        # cannot contain the failure cannot measure it.
        Step("Hello?", 12.0, "hello"),
        # 14 s: the acknowledgement is due within 2 s, and proving the 10 s cooldown
        # needs at least 10 s of listening after it lands.
        Step("What medication did the vet prescribe Marvin?", 14.0, "lookup-question"),
    ),
)

# The Flux hold-open bug specifically. On live call 01a02524 the callee's repeated
# "Hello?" re-armed Deepgram Flux's 5000 ms eotTimeout while eotThreshold 0.8 was never
# reached, so a 96 ms utterance ending at 3.33 s produced no final transcript until
# 14.29 s and the assistant did not speak until 16.92 s. Three greetings ~1 s apart
# reproduce that re-arming; the assertion is on the gap after the LAST one.
FLUX_STORM = Scenario(
    name="flux_storm",
    description="repeated greetings re-arming the transcriber's end-of-turn timeout",
    steps=(
        Step("Hello?", 1.0, "hello-1"),
        Step("Hello?", 1.0, "hello-2"),
        Step("Hello? Are you there?", 14.0, "hello-3"),
    ),
    # Greetings need answering, not researching: no tool call, so no holding line.
    ack_turn_index=None,
)

SCENARIOS: dict[str, Scenario] = {s.name: s for s in (DEADLINES, FLUX_STORM)}


def _pcm_from_wav(path: Path) -> bytes:
    with wave.open(str(path)) as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (
            SAMPLE_RATE,
            1,
            SAMPLE_WIDTH,
        ):
            raise AudioUnavailable(
                f"{path} is {w.getframerate()} Hz / {w.getnchannels()} ch / "
                f"{w.getsampwidth() * 8}-bit; the transport is opened as "
                f"{SAMPLE_RATE} Hz mono pcm_s16le"
            )
        return w.readframes(w.getnframes())


def _convert(src: Path, dst: Path) -> None:
    if shutil.which("afconvert"):
        cmd = ["afconvert", "-f", "WAVE", "-d", f"LEI16@{SAMPLE_RATE}", "-c", "1",
               str(src), str(dst)]
    elif shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1",
               "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(dst)]
    else:
        raise AudioUnavailable(
            "need afconvert (macOS) or ffmpeg to resample the synthesised audio to "
            f"{SAMPLE_RATE} Hz mono pcm_s16le"
        )
    subprocess.run(cmd, check=True, capture_output=True)  # noqa: S603


def synthesize(text: str, *, voice: str = DEFAULT_VOICE, cache_dir: Path | None = None) -> bytes:
    """``text`` as 16 kHz mono pcm_s16le, cached on disk and keyed by content."""
    out_dir = cache_dir or AUDIO_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{voice}|{SAMPLE_RATE}|{text}".encode()).hexdigest()[:16]
    wav = out_dir / f"{key}.wav"
    if not wav.exists():
        if not shutil.which("say"):
            raise AudioUnavailable(
                "macOS `say` not found: this harness synthesises the callee's voice "
                f"locally. Pre-render {SAMPLE_RATE} Hz mono pcm_s16le WAVs into "
                f"{out_dir} named <sha256(voice|rate|text)[:16]>.wav, or run on macOS."
            )
        aiff = out_dir / f"{key}.aiff"
        try:
            subprocess.run(  # noqa: S603
                ["say", "-v", voice, "-o", str(aiff), text], check=True, capture_output=True
            )
            _convert(aiff, wav)
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or b"").decode(errors="replace")[:400]
            raise AudioUnavailable(f"synthesising {text!r} failed: {stderr}") from exc
        finally:
            aiff.unlink(missing_ok=True)
    return _pcm_from_wav(wav)


def silence(seconds: float) -> bytes:
    frames = int(round(seconds * SAMPLE_RATE))
    return b"\x00" * (frames * SAMPLE_WIDTH)
