/** Where the app finds teeny-api, served by teeny-webapp as `/config.json`. */
export interface AppConfig {
  apiUrl: string
}

/** The API the dev server (`npm run dev`) talks to: `VITE_API_URL`, else a local teeny-api. */
const DEV_API_URL: string = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000'

/** Reads `/config.json`, falling back to the dev API when it is missing (under `npm run dev`). */
export async function loadConfig(): Promise<AppConfig> {
  try {
    const response = await fetch('/config.json', { cache: 'no-store' })
    if (response.ok) {
      const config = (await response.json()) as Partial<AppConfig>
      if (typeof config.apiUrl === 'string' && config.apiUrl !== '') {
        return { apiUrl: config.apiUrl.replace(/\/+$/, '') }
      }
    }
  } catch {
    // No config: the dev server.
  }
  return { apiUrl: DEV_API_URL }
}
