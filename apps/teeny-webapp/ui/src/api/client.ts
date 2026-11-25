import type { ChatChunk } from '@/lib/chat'
import type { ModelEntry } from '@/lib/events'
import { DONE, SseParser } from '@/lib/sse'

/** An error the API answered with, or why it could not be reached. */
export class ApiError extends Error {}

async function failure(response: Response): Promise<ApiError> {
  try {
    const body = (await response.json()) as { error?: { message?: string } }
    if (body.error?.message) {
      return new ApiError(body.error.message)
    }
  } catch {
    // Not an OpenAI-style error body.
  }
  return new ApiError(`the API answered ${response.status} ${response.statusText}`)
}

/** teeny-api's OpenAI-compatible endpoints, at `baseUrl`. */
export class ApiClient {
  constructor(readonly baseUrl: string) {}

  /** `/v1/events`, as a websocket URL. */
  eventsUrl(): string {
    return `${this.baseUrl.replace(/^http/, 'ws')}/v1/events`
  }

  async listModels(): Promise<ModelEntry[]> {
    let response: Response
    try {
      response = await fetch(`${this.baseUrl}/v1/models`)
    } catch (e) {
      throw new ApiError(`cannot reach the API at ${this.baseUrl}: ${(e as Error).message}`)
    }
    if (!response.ok) {
      throw await failure(response)
    }
    return ((await response.json()) as { data: ModelEntry[] }).data
  }

  /** Streams a chat completion of `body`, handing each chunk to `onChunk`, until it is done. */
  async streamChat(
    body: object,
    onChunk: (chunk: ChatChunk) => void,
    signal?: AbortSignal,
  ): Promise<void> {
    let response: Response
    try {
      response = await fetch(`${this.baseUrl}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal,
      })
    } catch (e) {
      if (signal?.aborted) {
        return
      }
      throw new ApiError(`cannot reach the API at ${this.baseUrl}: ${(e as Error).message}`)
    }
    if (!response.ok || !response.body) {
      throw await failure(response)
    }
    const reader = response.body.pipeThrough(new TextDecoderStream()).getReader()
    const parser = new SseParser()
    for (;;) {
      const { value, done } = await reader.read()
      if (done) {
        return
      }
      for (const data of parser.feed(value)) {
        if (data === DONE) {
          return
        }
        onChunk(JSON.parse(data) as ChatChunk)
      }
    }
  }
}
