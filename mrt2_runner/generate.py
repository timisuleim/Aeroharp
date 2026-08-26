"""
generate.py

Reads a control_sequence.json (produced by map_to_mrt2.py), groups
consecutive frames that share the same style_prompt into segments, and
generates one continuous audio clip per segment using Magenta RealTime 2 --
loading the model ONCE so we don't pay the ~55s compile cost per segment.

Two refinements over the baseline version:

1. STATE CHAINING: each segment's generate() call passes in the `state`
   returned by the previous segment's call, instead of starting fresh each
   time. This should make transitions between gestures sound like a
   continuation rather than an abrupt cut.

2. PITCH/INTENSITY VIA PROMPT TEXT: MRT2's pitch/notes conditioning
   (PIANOROLL_WITH_ONSETS) is a token-based representation with no public
   encoder function in this package version -- building it from scratch
   would mean guessing at internal token IDs with no way to verify
   correctness. Instead, each segment's average intensity and pitch are
   translated into descriptive words appended to the style prompt (e.g.
   "energetic, high-pitched bright upbeat synth pop"). This is a real,
   working mapping, just a coarser one than raw MIDI conditioning would be.

Usage:
    python generate.py control_sequence.json

Output:
    output/<clip_stem>_full.wav          -- all segments concatenated
    output/<clip_stem>_segment_XX.wav    -- each segment individually
"""

import sys
import json
import os
import numpy as np

from magenta_rt import paths
from magenta_rt import MagentaRT2Jax
from magenta_rt.config import MUSICCOCA

MODEL_NAME = "mrt2_small"  # must match a checkpoint you've actually downloaded via `mrt models download`

CONTROL_FPS = 20  # matches record.py's FPS -- rate control_sequence.json's frames are at
MRT_FPS = 25       # fixed by MagentaRT2Jax.generate()'s frame rate

OUTPUT_DIR = "output"

# Rough MIDI note range used by map_to_mrt2.py (MIDI_NOTE_MIN/MAX there)
PITCH_LOW_THRESHOLD = 58
PITCH_HIGH_THRESHOLD = 74

# Intensity is 0-127, like a MIDI CC value (see map_to_mrt2.py's `scale()` call)
INTENSITY_LOW_THRESHOLD = 42
INTENSITY_HIGH_THRESHOLD = 85


def load_control_sequence(path):
    with open(path) as f:
        return json.load(f)


def group_into_segments(events):
    """
    Groups consecutive control events that share the same style_prompt.
    Also carries along each event's pitch_midi/intensity so we can average
    them per segment for the prompt-modifier step.
    """
    if not events:
        return []

    segments = []
    current_prompt = events[0]["style_prompt"]
    start_frame = events[0]["frame"]
    pitches = [events[0]["pitch_midi"]]
    intensities = [events[0]["intensity"]]
    count = 1

    for event in events[1:]:
        if event["style_prompt"] == current_prompt:
            count += 1
            pitches.append(event["pitch_midi"])
            intensities.append(event["intensity"])
        else:
            segments.append({
                "style_prompt": current_prompt,
                "start_frame": start_frame,
                "num_control_frames": count,
                "avg_pitch": float(np.mean(pitches)),
                "avg_intensity": float(np.mean(intensities)),
            })
            current_prompt = event["style_prompt"]
            start_frame = event["frame"]
            pitches = [event["pitch_midi"]]
            intensities = [event["intensity"]]
            count = 1

    segments.append({
        "style_prompt": current_prompt,
        "start_frame": start_frame,
        "num_control_frames": count,
        "avg_pitch": float(np.mean(pitches)),
        "avg_intensity": float(np.mean(intensities)),
    })
    return segments


def build_modified_prompt(base_prompt, avg_pitch, avg_intensity):
    """
    Folds average pitch/intensity for a segment into descriptive words
    prepended to the base style prompt. This is the "simple" mapping we
    chose over raw MIDI/pianoroll conditioning -- coarser, but real.
    """
    modifiers = []

    if avg_intensity >= INTENSITY_HIGH_THRESHOLD:
        modifiers.append("energetic")
    elif avg_intensity <= INTENSITY_LOW_THRESHOLD:
        modifiers.append("soft")

    if avg_pitch >= PITCH_HIGH_THRESHOLD:
        modifiers.append("high-pitched")
    elif avg_pitch <= PITCH_LOW_THRESHOLD:
        modifiers.append("low-pitched")

    if not modifiers:
        return base_prompt
    return f"{', '.join(modifiers)} {base_prompt}"


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate.py path/to/control_sequence.json")
        return

    control_path = sys.argv[1]
    if not os.path.exists(control_path):
        print(f"Control sequence not found: {control_path}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    clip_stem = os.path.splitext(os.path.basename(control_path))[0]

    events = load_control_sequence(control_path)
    segments = group_into_segments(events)

    print(f"Loaded {len(events)} control events -> {len(segments)} style segments")
    for i, seg in enumerate(segments):
        duration_sec = seg["num_control_frames"] / CONTROL_FPS
        modified_prompt = build_modified_prompt(seg["style_prompt"], seg["avg_pitch"], seg["avg_intensity"])
        print(f"  segment {i}: '{modified_prompt}' -- {seg['num_control_frames']} frames "
              f"(~{duration_sec:.2f}s, avg_pitch={seg['avg_pitch']:.1f}, avg_intensity={seg['avg_intensity']:.1f})")

    # Load the model ONCE -- this is the ~55s compile cost, paid a single time
    print(f"\nLoading MagentaRT2Jax model ({MODEL_NAME}, this compiles once, ~55s)...")
    mrt = MagentaRT2Jax(
        size=MODEL_NAME,
        checkpoint=None,
        temperature=1.3,
        top_k=40,
        cfg_scales={"musiccoca": 3.0, "notes": 0.1, "drums": 1.0},
    )
    print("Model loaded.\n")

    segment_wavs = []
    running_state = None  # carried across generate() calls for smoother transitions

    for i, seg in enumerate(segments):
        duration_sec = seg["num_control_frames"] / CONTROL_FPS
        mrt_frames = max(1, int(duration_sec * MRT_FPS))
        modified_prompt = build_modified_prompt(seg["style_prompt"], seg["avg_pitch"], seg["avg_intensity"])

        print(f"Generating segment {i} ('{modified_prompt}', {mrt_frames} MRT frames)...")
        embedding = mrt.embed_style(modified_prompt, use_mapper=True)
        wav, running_state = mrt.generate(
            conditioning={MUSICCOCA.key: embedding},
            frames=mrt_frames,
            state=running_state,  # None for the first segment, carried state after that
        )

        segment_path = os.path.join(OUTPUT_DIR, f"{clip_stem}_segment_{i:02d}.wav")
        wav.write(segment_path)
        print(f"  saved {segment_path}")

        segment_wavs.append(wav)

    full_audio = np.concatenate([w.samples for w in segment_wavs], axis=0)
    full_path = os.path.join(OUTPUT_DIR, f"{clip_stem}_full.wav")
    segment_wavs[0].__class__(full_audio, segment_wavs[0].sample_rate).write(full_path)

    print(f"\nDone. Full stitched output: {full_path}")


if __name__ == "__main__":
    main()
