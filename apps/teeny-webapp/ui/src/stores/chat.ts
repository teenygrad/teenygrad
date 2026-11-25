import { defineStore } from 'pinia'
import { computed, reactive, ref } from 'vue'

import { useServerStore } from './server'
import {
  chatRequest,
  ReplyAccumulator,
  type ChatMessage,
  type ToolChoice,
  type Usage,
} from '@/lib/chat'
import type { Timings } from '@/lib/timings'
import { parseTools } from '@/lib/tools'

/**
 * A conversation with one model: the settings it is sent with (model, system prompt, max
 * tokens, temperature, tools), its messages, and the reply streaming in. When the model calls
 * tools, the conversation waits for a result for each, then continues.
 */
export const useChatStore = defineStore('chat', () => {
  const server = useServerStore()

  const settings = reactive({
    model: '',
    system: '',
    maxTokens: null as number | null,
    temperature: null as number | null,
    toolsText: '',
    toolChoice: 'auto' as ToolChoice,
  })
  const messages = ref<ChatMessage[]>([])
  /** The reply streaming in, if one is. */
  const draft = ref<{ content: string; tools: string[] } | null>(null)
  const error = ref<string | null>(null)
  const usage = ref<Usage | null>(null)
  const timings = ref<Timings | null>(null)
  let abort: AbortController | null = null

  const tools = computed(() => parseTools(settings.toolsText))
  const toolsError = computed(() => ('error' in tools.value ? tools.value.error : null))
  const streaming = computed(() => draft.value !== null)

  /** The tool calls of the last reply still waiting for a result. */
  const pendingCalls = computed(() => {
    const last = messages.value[messages.value.length - 1]
    return last?.role === 'assistant' ? (last.tool_calls ?? []) : []
  })

  /** Sends the conversation, streaming the model's reply onto it. */
  async function complete(): Promise<void> {
    if ('error' in tools.value) {
      error.value = tools.value.error
      return
    }
    error.value = null
    const reply = new ReplyAccumulator()
    draft.value = { content: '', tools: [] }
    abort = new AbortController()
    const body = chatRequest(
      {
        model: settings.model,
        system: settings.system,
        maxTokens: settings.maxTokens,
        temperature: settings.temperature,
        tools: tools.value.tools,
        // The choice is for the model's answer to the user; continuing from tool results
        // is left to it, so a forced call doesn't repeat for ever.
        toolChoice:
          messages.value[messages.value.length - 1]?.role === 'user' ? settings.toolChoice : 'auto',
      },
      messages.value,
    )
    try {
      await server.api().streamChat(
        body,
        (chunk) => {
          reply.apply(chunk)
          draft.value = {
            content: reply.content,
            tools: reply.toolCalls.filter(Boolean).map((c) => c.function.name),
          }
        },
        abort.signal,
      )
    } catch (e) {
      error.value = (e as Error).message
    } finally {
      abort = null
      draft.value = null
    }
    if (reply.content !== '' || reply.toolCalls.length > 0) {
      messages.value.push(reply.message())
    }
    usage.value = reply.usage ?? usage.value
    timings.value = reply.timings ?? timings.value
  }

  /** Adds `text` as the user's message and asks the model for its reply. */
  async function send(text: string): Promise<void> {
    if (text.trim() === '' || streaming.value || pendingCalls.value.length > 0) {
      return
    }
    messages.value.push({ role: 'user', content: text })
    await complete()
  }

  /** Answers the pending tool calls with `results` (by call id), then continues. */
  async function submitToolResults(results: Record<string, string>): Promise<void> {
    const calls = pendingCalls.value
    if (calls.length === 0 || streaming.value) {
      return
    }
    for (const call of calls) {
      messages.value.push({ role: 'tool', tool_call_id: call.id, content: results[call.id] ?? '' })
    }
    await complete()
  }

  /** Stops the reply streaming in, keeping what has arrived. */
  function stop(): void {
    abort?.abort()
  }

  /** Forgets the conversation (the settings stay). */
  function clear(): void {
    stop()
    messages.value = []
    error.value = null
    usage.value = null
    timings.value = null
  }

  return {
    settings,
    messages,
    draft,
    error,
    usage,
    timings,
    toolsError,
    streaming,
    pendingCalls,
    send,
    submitToolResults,
    stop,
    clear,
  }
})
