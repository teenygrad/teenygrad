# teeny-image

The image generation API for [teenygrad](https://teenygrad.org), served over HTTP by
[`teeny-api`](https://docs.teenygrad.org/api/teenygrad/teeny_api/) with its `image` feature.
Nothing here knows about HTTP.

- `ImageModel`, the trait diffusion-class models implement for `/v1/images/generations`,
  `/edits` and `/variations`, with its request types, the capabilities a model declares,
  `validate` (refusing what a model cannot do, by parameter) and `EchoImageModel`, its
  reference implementation
- `Rgba` pixels, and decoding uploads and encoding results (`decode`, `encode`,
  `encode_stream`), so no image model repeats them. PNG so far; JPEG and WebP are refused by
  name.

```bash
cargo test -p teeny-image
```

No model implementations yet: `teeny-api --echo` serves `teeny-echo-image`, which paints the
prompt as a colour, for trying the endpoints.

## License

Apache-2.0. See [LICENSE](https://github.com/teenygrad/teenygrad/blob/main/LICENSE).
