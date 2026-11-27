import { describe, expect, it } from 'vitest'

import { applyEvent, emptyState, KEEP_REQUESTS, type ServerEvent } from '../events'

const model = (id: string) => ({ id, object: 'model', created: 0, owned_by: 'teenygrad' })

describe('applyEvent', () => {
  it('test_follows_models_from_hello_through_loading', () => {
    const state = emptyState()
    const events: ServerEvent[] = [
      { type: 'hello', time: 1, version: '0.3.2', models: [model('teeny-echo')] },
      { type: 'model.loading', time: 2, model: 'functiongemma3:270m', detail: 'compiling' },
      { type: 'model.loading', time: 3, model: 'other:1b', detail: 'loading' },
    ]
    events.forEach((e) => applyEvent(state, e))
    expect(state.version).toBe('0.3.2')
    expect(state.models.map((m) => [m.id, m.status, m.detail])).toEqual([
      ['teeny-echo', 'served', ''],
      ['functiongemma3:270m', 'loading', 'compiling'],
      ['other:1b', 'loading', 'loading'],
    ])
    applyEvent(state, { type: 'model.added', time: 4, model: model('functiongemma3:270m') })
    applyEvent(state, { type: 'model.failed', time: 5, model: 'other:1b', error: 'no weights' })
    expect(state.models.map((m) => [m.id, m.status])).toEqual([
      ['teeny-echo', 'served'],
      ['other:1b', 'failed'],
      ['functiongemma3:270m', 'served'],
    ])
    expect(state.log[0]?.type).toBe('model.failed')
    expect(state.lastSeen).toBe(5)
  })

  it('test_tracks_requests_and_keeps_the_latest', () => {
    const state = emptyState()
    applyEvent(state, {
      type: 'request.started',
      time: 1,
      id: 'r1',
      method: 'GET',
      path: '/v1/models',
    })
    expect(state.requests[0]?.status).toBeNull()
    applyEvent(state, {
      type: 'request.finished',
      time: 2,
      id: 'r1',
      method: 'GET',
      path: '/v1/models',
      status: 200,
      duration_ms: 3,
    })
    expect(state.requests[0]).toMatchObject({ status: 200, duration_ms: 3, timings: null })
    const timings = {
      cache_n: 0,
      prompt_n: 4,
      prompt_ms: 2,
      prompt_per_token_ms: 0.5,
      prompt_per_second: 2000,
      predicted_n: 2,
      predicted_ms: 1,
      predicted_per_token_ms: 1,
      predicted_per_second: 1000,
    }
    applyEvent(state, { type: 'request.stats', time: 2, id: 'r1', model: 'teeny-echo', timings })
    applyEvent(state, { type: 'request.stats', time: 2, id: 'gone', model: 'x', timings })
    expect(state.requests[0]).toMatchObject({ model: 'teeny-echo', timings })
    for (let i = 0; i < KEEP_REQUESTS + 5; i++) {
      applyEvent(state, { type: 'request.started', time: 3, id: `x${i}`, method: 'GET', path: '/' })
    }
    expect(state.requests.length).toBe(KEEP_REQUESTS)
    applyEvent(state, { type: 'heartbeat', time: 9 })
    applyEvent(state, { type: 'lagged', time: 10, missed: 4 })
    expect(state.missed).toBe(4)
    expect(state.log.some((e) => e.type === 'heartbeat')).toBe(false)
  })
})
