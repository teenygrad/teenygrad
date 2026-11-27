/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** The teeny-api `npm run dev` talks to (default `http://127.0.0.1:8000`). */
  readonly VITE_API_URL?: string
}
