import { describe, expect, it } from 'vitest'

import { describeTimings, percentile, type Timings } from '../timings'

const timings: Timings = {
  cache_n: 16,
  prompt_n: 66,
  prompt_ms: 122.6,
  prompt_per_token_ms: 1.86,
  prompt_per_second: 538.1,
  predicted_n: 4,
  predicted_ms: 64.5,
  predicted_per_token_ms: 21.5,
  predicted_per_second: 46.5,
  queued_ms: 2,
  first_token_ms: 130,
  predicted_latencies_ms: [21.3, 20.9, 22.3],
}

describe('timings', () => {
  it('test_percentiles_are_nearest_rank', () => {
    expect(percentile([5, 1, 4, 2, 3], 50)).toBe(3)
    expect(percentile([5, 1, 4, 2, 3], 99)).toBe(5)
    expect(percentile([5, 1, 4, 2, 3], 0)).toBe(1)
    expect(percentile([], 50)).toBeNull()
  })

  it('test_describes_a_reply_in_one_line', () => {
    expect(describeTimings(timings)).toBe(
      '66 prompt tokens at 538.1 tokens/s (16 cached) · first token after 130.0 ms · ' +
        '4 tokens out at 46.5 tokens/s (p50 21.3 ms, p99 22.3 ms)',
    )
    expect(describeTimings({ ...timings, cache_n: 0, predicted_n: 1, first_token_ms: 1500 })).toBe(
      '66 prompt tokens at 538.1 tokens/s · first token after 1.50 s · 1 tokens out',
    )
  })
})
