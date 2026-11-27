<script setup lang="ts">
import { nextTick, reactive, ref, watch, watchEffect } from 'vue'

import { describeTimings } from '@/lib/timings'
import { useChatStore } from '@/stores/chat'
import { useServerStore } from '@/stores/server'

const chat = useChatStore()
const server = useServerStore()

const input = ref('')
const results = reactive<Record<string, string>>({})
const transcript = ref<HTMLElement | null>(null)

// Talk to a served model: the first, until one is picked (or the picked one goes).
watchEffect(() => {
  const served = server.servedModels
  if (!served.includes(chat.settings.model) && served.length > 0) {
    chat.settings.model = served[0] ?? ''
  }
})

// Keep the latest of the conversation in view.
watch(
  () => [chat.messages.length, chat.draft?.content],
  async () => {
    await nextTick()
    transcript.value?.scrollTo({ top: transcript.value.scrollHeight })
  },
)

async function send(): Promise<void> {
  const text = input.value
  input.value = ''
  await chat.send(text)
}

async function answerTools(): Promise<void> {
  await chat.submitToolResults({ ...results })
  for (const id of Object.keys(results)) {
    delete results[id]
  }
}

function onKey(event: KeyboardEvent): void {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    void send()
  }
}

/** A number setting from its input: empty is unset. */
function numberOrNull(value: string): number | null {
  const n = Number(value)
  return value.trim() === '' || Number.isNaN(n) ? null : n
}
</script>

<template>
  <div class="chat">
    <aside class="settings panel">
      <h2>Settings</h2>
      <label>
        Model
        <select v-model="chat.settings.model" :disabled="server.servedModels.length === 0">
          <option v-if="server.servedModels.length === 0" value="">no models served</option>
          <option v-for="id in server.servedModels" :key="id" :value="id">{{ id }}</option>
        </select>
      </label>
      <label>
        System prompt
        <textarea v-model="chat.settings.system" rows="4" placeholder="optional" />
      </label>
      <div class="row">
        <label>
          Max tokens
          <input
            type="number"
            min="1"
            placeholder="model default"
            :value="chat.settings.maxTokens ?? ''"
            @input="
              chat.settings.maxTokens = numberOrNull(($event.target as HTMLInputElement).value)
            "
          />
        </label>
        <label>
          Temperature
          <input
            type="number"
            min="0"
            max="2"
            step="0.1"
            placeholder="model default"
            :value="chat.settings.temperature ?? ''"
            @input="
              chat.settings.temperature = numberOrNull(($event.target as HTMLInputElement).value)
            "
          />
        </label>
      </div>
      <label>
        Tools
        <textarea
          v-model="chat.settings.toolsText"
          rows="10"
          spellcheck="false"
          class="mono"
          placeholder='[{"type": "function", "function": {"name": "get_weather", "parameters": {...}}}]'
        />
      </label>
      <p v-if="chat.toolsError" class="error">{{ chat.toolsError }}</p>
      <label>
        Tool choice
        <select v-model="chat.settings.toolChoice">
          <option value="auto">auto: the model decides</option>
          <option value="required">required: it must call one</option>
          <option value="none">none: it may not call any</option>
        </select>
      </label>
      <p class="hint">
        OpenAI-style function declarations. The model's calls are shown, and you type each result.
      </p>
    </aside>

    <section class="conversation panel">
      <div ref="transcript" class="transcript">
        <p v-if="chat.messages.length === 0 && !chat.draft" class="hint">
          Say something to {{ chat.settings.model || 'a model' }}.
        </p>
        <template v-for="(m, i) in chat.messages" :key="i">
          <div v-if="m.role === 'user'" class="message user">{{ m.content }}</div>
          <div v-else-if="m.role === 'assistant'" class="message assistant">
            <div v-if="m.content">{{ m.content }}</div>
            <div v-for="call in m.tool_calls ?? []" :key="call.id" class="tool-call mono">
              → {{ call.function.name }}({{ call.function.arguments }})
            </div>
          </div>
          <div v-else-if="m.role === 'tool'" class="message tool mono">← {{ m.content }}</div>
        </template>
        <div v-if="chat.draft" class="message assistant streaming">
          {{ chat.draft.content }}
          <span v-for="name in chat.draft.tools" :key="name" class="tool-call mono"
            >→ {{ name }}(…)</span
          >
          <span class="cursor">▍</span>
        </div>
      </div>

      <p v-if="chat.error" class="error">{{ chat.error }}</p>

      <form v-if="chat.pendingCalls.length > 0" class="tool-results" @submit.prevent="answerTools">
        <p class="hint">The model called tools; give each result to continue.</p>
        <label v-for="call in chat.pendingCalls" :key="call.id">
          <span class="mono">{{ call.function.name }}({{ call.function.arguments }})</span>
          <input v-model="results[call.id]" class="mono" placeholder='{"result": ...}' />
        </label>
        <button type="submit" :disabled="chat.streaming">Send results</button>
      </form>

      <form v-else class="composer" @submit.prevent="send">
        <textarea
          v-model="input"
          rows="2"
          placeholder="Message (Enter to send, Shift+Enter for a new line)"
          :disabled="!chat.settings.model"
          @keydown="onKey"
        />
        <button v-if="chat.streaming" type="button" @click="chat.stop()">Stop</button>
        <button v-else type="submit" :disabled="!chat.settings.model || input.trim() === ''">
          Send
        </button>
      </form>

      <footer class="status">
        <span>
          <span v-if="chat.usage">
            {{ chat.usage.prompt_tokens }} prompt + {{ chat.usage.completion_tokens }} completion
            tokens
          </span>
          <span v-if="chat.timings" class="timings" :title="'Last reply'">
            · {{ describeTimings(chat.timings) }}
          </span>
        </span>
        <button type="button" class="link" @click="chat.clear()">Clear conversation</button>
      </footer>
    </section>
  </div>
</template>

<style scoped>
.chat {
  display: grid;
  grid-template-columns: minmax(16rem, 22rem) 1fr;
  gap: 1rem;
  height: 100%;
  min-height: 0;
}
.settings {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  overflow-y: auto;
}
.settings h2 {
  margin: 0;
  font-size: 1rem;
}
label {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  font-size: 0.85rem;
  color: var(--muted);
}
.row {
  display: flex;
  gap: 0.5rem;
}
.row label {
  flex: 1;
}
.conversation {
  display: flex;
  flex-direction: column;
  min-height: 0;
}
.transcript {
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  padding-bottom: 0.5rem;
}
.message {
  max-width: 80%;
  padding: 0.5rem 0.75rem;
  border-radius: 0.5rem;
  white-space: pre-wrap;
  word-break: break-word;
}
.user {
  align-self: flex-end;
  background: var(--accent);
  color: var(--on-accent);
}
.assistant {
  align-self: flex-start;
  background: var(--surface-2);
}
.tool {
  align-self: flex-start;
  background: transparent;
  border: 1px dashed var(--border);
  font-size: 0.85rem;
}
.tool-call {
  display: block;
  font-size: 0.85rem;
  color: var(--muted);
}
.cursor {
  animation: blink 1s steps(2) infinite;
}
@keyframes blink {
  50% {
    opacity: 0;
  }
}
.composer,
.tool-results {
  display: flex;
  gap: 0.5rem;
  align-items: flex-end;
  border-top: 1px solid var(--border);
  padding-top: 0.75rem;
}
.tool-results {
  flex-direction: column;
  align-items: stretch;
}
.composer textarea {
  flex: 1;
}
.status {
  display: flex;
  justify-content: space-between;
  font-size: 0.8rem;
  color: var(--muted);
  padding-top: 0.5rem;
}
</style>
