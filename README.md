# AeroHarp

A gesture-controlled live music instrument. A webcam tracks hand movement, and hand position and shape become real-time control signals for [Magenta RealTime 2](https://github.com/magenta/magenta-realtime), an open source live music model — so you can "play" AI-generated music with your hands instead of a keyboard.

CS 352 Final Project, Northwestern University.

## What it does

Most AI music tools work by typing a prompt and waiting for a finished clip. AeroHarp tries to make it feel like playing an instrument instead: a webcam captures your hand, [MediaPipe](https://developers.google.com/mediapipe) tracks its landmarks, a classifier turns that into gesture and motion data, and that data controls Magenta RealTime 2 to generate music live in response.

## Repo structure
