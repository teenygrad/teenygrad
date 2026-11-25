import type { FunctionTool } from './tools'

/** A tool call the model made: which function, with what arguments (JSON text). */
export interface ToolCall {
  id: string
  type: 'function'
  function: { name: string; arguments: string }
}

/** A message of the conversation, as the chat completions API takes it. */
export type ChatMessage =
  | { role: 'system'; content: string }
  | { role: 'user'; content: string }
  | { role: 'assistant'; content: string | null; tool_calls?: ToolCall[] }
  | { role: 'tool'; tool_call_id: string; content: string }

export interface Usage {
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
}

/** Whether the model may (`auto`), must (`required`) or may not (`none`) call a tool. */
export type ToolChoice = 'auto' | 'required' | 'none'

/** What a request is sent with, besides the conversation. */
export interface ChatSettings {
  model: string
  system: string
  maxTokens: number | null
  temperature: number | null
  tools: FunctionTool[]
  toolChoice: ToolChoice
}

/** The body of a streamed chat completion of `messages` with `settings`. */
export function chatRequest(settings: ChatSettings, messages: ChatMessage[]): object {
  const system: ChatMessage[] =
    settings.system.trim() === '' ? [] : [{ role: 'system', content: settings.system }]
  return {
    model: settings.model,
    messages: [...system, ...messages],
    stream: true,
    stream_options: { include_usage: true },
    ...(settings.maxTokens != null ? { max_completion_tokens: settings.maxTokens } : {}),
    ...(settings.temperature != null ? { temperature: settings.temperature } : {}),
    ...(settings.tools.length > 0 ? { tools: settings.tools } : {}),
    ...(settings.tools.length > 0 && settings.toolChoice !== 'auto'
      ? { tool_choice: settings.toolChoice }
      : {}),
  }
}

/** A tool call's pieces as they stream in, by `index`. */
interface ToolCallDelta {
  index: number
  id?: string
  type?: string
  function?: { name?: string; arguments?: string }
}

/** One chunk of a streamed chat completion (`chat.completion.chunk`), or an error. */
export interface ChatChunk {
  choices?: {
    index: number
    delta: { content?: string | null; tool_calls?: ToolCallDelta[] }
    finish_reason?: string | null
  }[]
  usage?: Usage | null
  error?: { message: string }
}

/**
 * The first choice of a streamed chat completion, put together from its chunks: the reply's
 * text, the tool calls it makes, why it finished, and the usage.
 */
export class ReplyAccumulator {
  content = ''
  toolCalls: ToolCall[] = []
  finishReason: string | null = null
  usage: Usage | null = null

  /** Adds `chunk`; throws the error a chunk carries. */
  apply(chunk: ChatChunk): void {
    if (chunk.error) {
      throw new Error(chunk.error.message)
    }
    if (chunk.usage) {
      this.usage = chunk.usage
    }
    for (const choice of chunk.choices ?? []) {
      if (choice.index !== 0) {
        continue
      }
      this.content += choice.delta.content ?? ''
      for (const d of choice.delta.tool_calls ?? []) {
        const call = (this.toolCalls[d.index] ??= {
          id: '',
          type: 'function',
          function: { name: '', arguments: '' },
        })
        call.id = d.id ?? call.id
        call.function.name += d.function?.name ?? ''
        call.function.arguments += d.function?.arguments ?? ''
      }
      if (choice.finish_reason) {
        this.finishReason = choice.finish_reason
      }
    }
  }

  /** The reply, as the assistant message to send back with the conversation. */
  message(): ChatMessage {
    const calls = this.toolCalls.filter(Boolean)
    return {
      role: 'assistant',
      content: this.content === '' && calls.length > 0 ? null : this.content,
      ...(calls.length > 0 ? { tool_calls: calls } : {}),
    }
  }
}
