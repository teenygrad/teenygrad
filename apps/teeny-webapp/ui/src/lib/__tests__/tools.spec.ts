import { describe, expect, it } from 'vitest'

import { parseTools } from '../tools'

describe('parseTools', () => {
  it('test_reads_openai_function_declarations', () => {
    const parsed = parseTools(
      '[{"type": "function", "function": {"name": "get_weather", "description": "d", "parameters": {"type": "object"}}}]',
    )
    expect(parsed).toEqual({
      tools: [
        {
          type: 'function',
          function: { name: 'get_weather', description: 'd', parameters: { type: 'object' } },
        },
      ],
    })
    expect(parseTools('  ')).toEqual({ tools: [] })
  })

  it('test_says_what_is_wrong', () => {
    const error = (text: string) => {
      const parsed = parseTools(text)
      return 'error' in parsed ? parsed.error : null
    }
    expect(error('{')).toMatch(/not JSON/)
    expect(error('{}')).toMatch(/JSON array/)
    expect(error('[{"type": "web_search"}]')).toMatch(/tools\[0\] is not a function tool/)
    expect(error('[{"type": "function", "function": {}}]')).toMatch(/name must be a string/)
  })
})
