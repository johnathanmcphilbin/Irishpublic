# Irish Usage Tester

Small browser prototype that records one conversation, runs two separate transcription paths, and then estimates Irish versus English usage from the combined result.

## What it does

- Records microphone audio in the browser.
- Sends the recording to `https://recognition.abair.ie/v3-5/transcribe` for an Irish-oriented transcript path.
- Runs a local browser-side English recognizer using Transformers.js and `Xenova/whisper-tiny.en`.
- Keeps the Abair and English transcripts separate.
- Reassigns uncertain words with a simple comparison pass and builds a best-guess merged transcript.
- Estimates Irish percent, English percent, and an Irish-confidence score using transcript heuristics.

## Limits

- This is not a true bilingual speech-language classifier.
- The Irish and English percentages are inferred from transcript text, not directly from acoustic language identification.
- The merged transcript is a best guess from two imperfect ASR outputs, not a verified reconstruction.
- The first run may take longer because the English recognizer model downloads into the browser.
- If either recognizer fails or hears code-switching badly, the final split will still be noisy.

## Run locally

Use any local web server so the browser can access the microphone. One simple option is:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000` in a browser.

## Notes

- The page now shows three transcript views: Abair, English, and a merged best-guess conversation.
- The English recognizer loads in the browser, so it does not require a local backend.
- Word confidence values shown in the UI are heuristic confidence values from the reconciliation logic, not recognizer-supplied per-word confidence scores.