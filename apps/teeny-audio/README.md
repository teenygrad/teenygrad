# teeny-audio

Audio inference APIs for [teenygrad](https://teenygrad.org), served over HTTP by
[`teeny-api`](https://docs.teenygrad.org/api/teenygrad/teeny_api/) with its `audio` feature.
Nothing here knows about HTTP.

- one module per OpenAI audio API, each with its request and response types, the capabilities
  a model declares, `validate` (refusing what a model cannot do, by parameter), the trait models
  implement, and an echo model as its reference implementation: `transcription`
  (`TranscriptionModel`, for `/v1/audio/transcriptions` and `/translations`) and `speech`
  (`SpeechModel`, for `/v1/audio/speech`)
- `codec`: decoding uploads into `Pcm` samples (WAV so far) and encoding samples (WAV, raw
  16-bit PCM), mixed to mono and resampled for the model, so no audio model repeats it. The
  compressed containers (FLAC, MP3, Ogg, ...) are recognised and refused by name.

```bash
cargo test -p teeny-audio
```

No model implementations yet: `teeny-api --echo` serves the echo models (`teeny-echo-transcribe`,
which describes the recording, and `teeny-echo-speak`, which beeps once a word) for trying the
endpoints.

## License

Apache-2.0. See [LICENSE](https://github.com/teenygrad/teenygrad/blob/main/LICENSE).
