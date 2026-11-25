import { defineStore } from 'pinia'
import { computed, reactive, ref, shallowRef } from 'vue'

import { ApiClient } from '@/api/client'
import { EventSocket, type SocketState } from '@/api/socket'
import { applyEvent, emptyState, type ServerEvent } from '@/lib/events'

/** The teeny-api the app talks to, and what its `/v1/events` has said. */
export const useServerStore = defineStore('server', () => {
  const client = shallowRef<ApiClient | null>(null)
  const state = reactive(emptyState())
  const socketState = ref<SocketState>('closed')
  let socket: EventSocket | null = null

  /** The models served now, by id: those a request can name. */
  const servedModels = computed(() =>
    state.models.filter((m) => m.status === 'served').map((m) => m.id),
  )

  /** Talks to the API at `apiUrl`, following its events. */
  function connect(apiUrl: string): void {
    socket?.close()
    client.value = new ApiClient(apiUrl)
    socket = new EventSocket(
      client.value.eventsUrl(),
      (event: ServerEvent) => applyEvent(state, event),
      (s) => (socketState.value = s),
    )
  }

  function api(): ApiClient {
    if (!client.value) {
      throw new Error('the app is not connected to an API')
    }
    return client.value
  }

  return { client, state, socketState, servedModels, connect, api }
})
