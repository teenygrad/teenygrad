/**
 * Server-sent events, as the OpenAI streaming endpoints send them: `data:` lines, events
 * separated by a blank line, the stream ended by `data: [DONE]`.
 */
export class SseParser {
  private buffer = ''

  /** Feeds `text` (any slice of the stream) and returns the data of each event it completes. */
  feed(text: string): string[] {
    this.buffer += text.replace(/\r\n?/g, '\n')
    const out: string[] = []
    let end: number
    while ((end = this.buffer.indexOf('\n\n')) !== -1) {
      const event = this.buffer.slice(0, end)
      this.buffer = this.buffer.slice(end + 2)
      const data = event
        .split('\n')
        .filter((line) => line.startsWith('data:'))
        .map((line) => line.slice(5).replace(/^ /, ''))
      if (data.length > 0) {
        out.push(data.join('\n'))
      }
    }
    return out
  }
}

/** The stream's terminator. */
export const DONE = '[DONE]'
