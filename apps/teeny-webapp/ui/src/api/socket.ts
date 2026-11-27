import type { ServerEvent } from '@/lib/events'

export type SocketState = 'connecting' | 'open' | 'closed'

/** How long to wait before reconnecting, doubling to at most the last. */
const BACKOFF_MS = [500, 1000, 2000, 5000]

/**
 * `/v1/events` at `url`, reconnecting whenever it drops, until `close`. Each event goes to
 * `onEvent`, and each change of connection to `onState`.
 */
export class EventSocket {
  private socket: WebSocket | null = null
  private attempt = 0
  private retry: ReturnType<typeof setTimeout> | null = null
  private closed = false

  constructor(
    private readonly url: string,
    private readonly onEvent: (event: ServerEvent) => void,
    private readonly onState: (state: SocketState) => void,
  ) {
    this.open()
  }

  private open(): void {
    this.onState('connecting')
    const socket = new WebSocket(this.url)
    this.socket = socket
    socket.onopen = () => {
      this.attempt = 0
      this.onState('open')
    }
    socket.onmessage = (message) => {
      try {
        this.onEvent(JSON.parse(String(message.data)) as ServerEvent)
      } catch {
        // Not an event; nothing to show.
      }
    }
    socket.onclose = () => {
      this.socket = null
      this.onState('closed')
      if (!this.closed) {
        const delay = BACKOFF_MS[Math.min(this.attempt, BACKOFF_MS.length - 1)]
        this.attempt += 1
        this.retry = setTimeout(() => this.open(), delay)
      }
    }
  }

  close(): void {
    this.closed = true
    if (this.retry !== null) {
      clearTimeout(this.retry)
    }
    this.socket?.close()
  }
}
