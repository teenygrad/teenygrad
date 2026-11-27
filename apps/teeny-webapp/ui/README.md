# teeny-webapp UI

The Vue 3 app `teeny-webapp` embeds: `/chat` (a chat with a model teeny-api serves, with
tools) and `/console` (teeny-api's `/v1/events`). `cargo build -p teeny-webapp` builds it with
npm; work on it here with Vite's dev server, against a running teeny-api:

```sh
npm ci
npm run dev                                      # http://localhost:5173, API at http://127.0.0.1:8000
VITE_API_URL=http://board:8000 npm run dev       # another API
```

```sh
npm run type-check && npm run lint:check && npm run format:check && npm test
```

`src/lib` holds the logic (SSE parsing, tool declarations, putting replies together, following
server events), unit-tested in `src/lib/__tests__`; `src/stores` the app's state (Pinia);
`src/views` the two pages.
