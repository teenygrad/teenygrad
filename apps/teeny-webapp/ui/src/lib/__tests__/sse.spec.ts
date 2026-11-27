import { describe, expect, it } from 'vitest'

import { DONE, SseParser } from '../sse'

describe('SseParser', () => {
  it('test_splits_events_across_any_chunking', () => {
    const stream = 'data: {"a":1}\n\ndata: {"b":2}\r\n\r\n: comment\n\ndata: [DONE]\n\n'
    for (const size of [1, 3, 7, stream.length]) {
      const parser = new SseParser()
      const out: string[] = []
      for (let i = 0; i < stream.length; i += size) {
        out.push(...parser.feed(stream.slice(i, i + size)))
      }
      expect(out).toEqual(['{"a":1}', '{"b":2}', DONE])
    }
  })

  it('test_joins_multiline_data', () => {
    expect(new SseParser().feed('data: one\ndata:two\n\n')).toEqual(['one\ntwo'])
  })
})
