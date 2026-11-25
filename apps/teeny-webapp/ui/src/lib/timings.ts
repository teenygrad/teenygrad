/**
 * How long a generation took, as teeny-api reports it: llama.cpp server's `timings`, plus
 * teenygrad's `queued_ms`, `first_token_ms` and `predicted_latencies_ms`.
 */
export interface Timings {
  cache_n: number
  prompt_n: number
  prompt_ms: number
  prompt_per_token_ms: number
  prompt_per_second: number
  predicted_n: number
  predicted_ms: number
  predicted_per_token_ms: number
  predicted_per_second: number
  queued_ms?: number
  first_token_ms?: number
  predicted_latencies_ms?: number[]
}

/** The `p`th percentile (0 to 100) of `values`, nearest-rank; null when empty. */
export function percentile(values: readonly number[], p: number): number | null {
  if (values.length === 0) {
    return null
  }
  const sorted = [...values].sort((a, b) => a - b)
  const rank = Math.ceil((Math.min(Math.max(p, 0), 100) / 100) * sorted.length)
  return sorted[Math.min(Math.max(rank, 1), sorted.length) - 1] ?? null
}

const ms = (x: number): string => (x >= 1000 ? `${(x / 1000).toFixed(2)} s` : `${x.toFixed(1)} ms`)
const rate = (x: number): string => `${x.toFixed(1)} tokens/s`

/** One line for a reply: the prompt phase, the first token, and decoding with its spread. */
export function describeTimings(t: Timings): string {
  const parts = [
    `${t.prompt_n} prompt tokens at ${rate(t.prompt_per_second)}` +
      (t.cache_n > 0 ? ` (${t.cache_n} cached)` : ''),
  ]
  if (t.first_token_ms !== undefined) {
    parts.push(`first token after ${ms(t.first_token_ms)}`)
  }
  let decode = `${t.predicted_n} tokens out`
  if (t.predicted_n > 1) {
    decode += ` at ${rate(t.predicted_per_second)}`
    const latencies = t.predicted_latencies_ms ?? []
    const p50 = percentile(latencies, 50)
    const p99 = percentile(latencies, 99)
    if (p50 !== null && p99 !== null) {
      decode += ` (p50 ${ms(p50)}, p99 ${ms(p99)})`
    }
  }
  parts.push(decode)
  return parts.join(' · ')
}
