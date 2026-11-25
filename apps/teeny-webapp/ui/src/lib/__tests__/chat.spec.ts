import { describe, expect, it } from 'vitest'

import { chatRequest, ReplyAccumulator } from '../chat'

describe('chatRequest', () => {
  it('test_sends_only_the_settings_given', () => {
    const settings = {
      model: 'm',
      system: '',
      maxTokens: null,
      temperature: null,
      tools: [],
      toolChoice: 'required' as const,
    }
    expect(chatRequest(settings, [{ role: 'user', content: 'hi' }])).toEqual({
      model: 'm',
      messages: [{ role: 'user', content: 'hi' }],
      stream: true,
      stream_options: { include_usage: true },
    })
    const full = chatRequest(
      {
        ...settings,
        system: 'be brief',
        maxTokens: 64,
        temperature: 0,
        tools: [{ type: 'function', function: { name: 'f' } }],
      },
      [],
    )
    expect(full).toHaveProperty('tool_choice', 'required')
    expect(
      chatRequest(
        { ...settings, tools: [{ type: 'function', function: { name: 'f' } }], toolChoice: 'auto' },
        [],
      ),
    ).not.toHaveProperty('tool_choice')
    expect(full).toMatchObject({
      messages: [{ role: 'system', content: 'be brief' }],
      max_completion_tokens: 64,
      temperature: 0,
      tools: [{ type: 'function', function: { name: 'f' } }],
    })
  })
})

describe('ReplyAccumulator', () => {
  it('test_puts_text_and_tool_calls_together', () => {
    const reply = new ReplyAccumulator()
    reply.apply({ choices: [{ index: 0, delta: { content: 'Let me ' } }] })
    reply.apply({ choices: [{ index: 0, delta: { content: 'check.' } }] })
    reply.apply({
      choices: [
        {
          index: 0,
          delta: {
            tool_calls: [{ index: 0, id: 'call_1', function: { name: 'get_', arguments: '{"ci' } }],
          },
        },
      ],
    })
    reply.apply({
      choices: [
        {
          index: 0,
          delta: { tool_calls: [{ index: 0, function: { name: 'weather', arguments: 'ty":1}' } }] },
          finish_reason: 'tool_calls',
        },
      ],
    })
    const timings = {
      cache_n: 0,
      prompt_n: 3,
      prompt_ms: 1,
      prompt_per_token_ms: 0.3,
      prompt_per_second: 3000,
      predicted_n: 4,
      predicted_ms: 3,
      predicted_per_token_ms: 1,
      predicted_per_second: 1000,
    }
    reply.apply({
      choices: [],
      usage: { prompt_tokens: 3, completion_tokens: 4, total_tokens: 7 },
      timings,
    })
    expect(reply.timings).toEqual(timings)
    expect(reply.finishReason).toBe('tool_calls')
    expect(reply.usage?.total_tokens).toBe(7)
    expect(reply.message()).toEqual({
      role: 'assistant',
      content: 'Let me check.',
      tool_calls: [
        {
          id: 'call_1',
          type: 'function',
          function: { name: 'get_weather', arguments: '{"city":1}' },
        },
      ],
    })
  })

  it('test_ignores_other_choices_and_throws_stream_errors', () => {
    const reply = new ReplyAccumulator()
    reply.apply({ choices: [{ index: 1, delta: { content: 'other' } }] })
    expect(reply.message()).toEqual({ role: 'assistant', content: '' })
    expect(() => reply.apply({ error: { message: 'model failed' } })).toThrow('model failed')
  })
})
