import type { Timings } from './timings'

/** A model, as `/v1/models` lists it. */
export interface ModelEntry {
  id: string
  object: string
  created: number
  owned_by: string
}

/** An event from `/v1/events`, by `type`; `time` is in Unix milliseconds. */
export type ServerEvent = { time: number } & (
  | { type: 'hello'; version: string; models: ModelEntry[] }
  | { type: 'heartbeat' }
  | { type: 'model.loading'; model: string; detail: string }
  | { type: 'model.added'; model: ModelEntry }
  | { type: 'model.failed'; model: string; error: string }
  | { type: 'request.started'; id: string; method: string; path: string }
  | {
      type: 'request.finished'
      id: string
      method: string
      path: string
      status: number
      duration_ms: number
    }
  | { type: 'request.stats'; id: string; model: string; timings: Timings }
  | { type: 'lagged'; missed: number }
)

/** A model the console shows: served, loading (with what it is doing), or failed. */
export interface ModelState {
  id: string
  status: 'served' | 'loading' | 'failed'
  detail: string
  owned_by: string
}

/**
 * An API request the console shows; `status` and `duration_ms` once it is answered, and
 * the model and its timings once a generation finishes.
 */
export interface RequestState {
  id: string
  method: string
  path: string
  started: number
  status: number | null
  duration_ms: number | null
  model: string | null
  timings: Timings | null
}

/** What the console knows about the server, from its events. */
export interface ServerState {
  version: string | null
  models: ModelState[]
  /** The most recent first. */
  requests: RequestState[]
  /** The most recent first. */
  log: ServerEvent[]
  lastSeen: number | null
  missed: number
}

/** How many requests, and events, the console keeps. */
export const KEEP_REQUESTS = 100
export const KEEP_EVENTS = 200

export function emptyState(): ServerState {
  return { version: null, models: [], requests: [], log: [], lastSeen: null, missed: 0 }
}

function upsertModel(state: ServerState, model: ModelState): void {
  const at = state.models.findIndex((m) => m.id === model.id)
  if (at === -1) {
    state.models.push(model)
  } else {
    state.models[at] = model
  }
}

/** Applies `event` to `state`, in place. */
export function applyEvent(state: ServerState, event: ServerEvent): void {
  state.lastSeen = event.time
  if (event.type !== 'heartbeat') {
    state.log.unshift(event)
    state.log.splice(KEEP_EVENTS)
  }
  switch (event.type) {
    case 'hello':
      state.version = event.version
      // A reconnect starts over from the server's own list.
      state.models = event.models.map((m) => ({
        id: m.id,
        status: 'served',
        detail: '',
        owned_by: m.owned_by,
      }))
      break
    case 'model.loading':
      upsertModel(state, { id: event.model, status: 'loading', detail: event.detail, owned_by: '' })
      break
    case 'model.added': {
      // A model loads under its tag and is served under its canonical id: the loading
      // entry becomes the served one.
      state.models = state.models.filter(
        (m) => !(m.status === 'loading' && event.model.id.startsWith(m.id)),
      )
      upsertModel(state, {
        id: event.model.id,
        status: 'served',
        detail: '',
        owned_by: event.model.owned_by,
      })
      break
    }
    case 'model.failed':
      upsertModel(state, { id: event.model, status: 'failed', detail: event.error, owned_by: '' })
      break
    case 'request.started':
      state.requests.unshift({
        id: event.id,
        method: event.method,
        path: event.path,
        started: event.time,
        status: null,
        duration_ms: null,
        model: null,
        timings: null,
      })
      state.requests.splice(KEEP_REQUESTS)
      break
    case 'request.finished': {
      const request = state.requests.find((r) => r.id === event.id)
      if (request) {
        request.status = event.status
        request.duration_ms = event.duration_ms
      }
      break
    }
    case 'request.stats': {
      const request = state.requests.find((r) => r.id === event.id)
      if (request) {
        request.model = event.model
        request.timings = event.timings
      }
      break
    }
    case 'lagged':
      state.missed += event.missed
      break
  }
}
