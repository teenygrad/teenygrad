<script setup lang="ts">
import { computed } from 'vue'

import { useServerStore } from '@/stores/server'

const server = useServerStore()

const time = (ms: number): string => new Date(ms).toLocaleTimeString()

const inFlight = computed(() => server.state.requests.filter((r) => r.status === null).length)

/** An event's fields, without its type and time. */
function fields(event: object): string {
  const rest = Object.fromEntries(
    Object.entries(event).filter(([key]) => key !== 'type' && key !== 'time'),
  )
  return JSON.stringify(rest)
}
</script>

<template>
  <div class="console">
    <section class="panel">
      <h2>Server</h2>
      <dl>
        <dt>API</dt>
        <dd class="mono">{{ server.client?.baseUrl }}</dd>
        <dt>Events</dt>
        <dd>
          <span :class="['dot', server.socketState]" />{{ server.socketState }}
          <span v-if="server.state.lastSeen" class="muted">
            · last heard {{ time(server.state.lastSeen) }}</span
          >
        </dd>
        <dt>Version</dt>
        <dd>{{ server.state.version ?? '—' }}</dd>
        <dt v-if="server.state.missed > 0">Missed</dt>
        <dd v-if="server.state.missed > 0" class="error">
          {{ server.state.missed }} events (the console fell behind)
        </dd>
      </dl>
    </section>

    <section class="panel">
      <h2>Models</h2>
      <p v-if="server.state.models.length === 0" class="hint">No models yet.</p>
      <table v-else>
        <thead>
          <tr>
            <th>Model</th>
            <th>State</th>
            <th>Owner</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="m in server.state.models" :key="m.id">
            <td class="mono">{{ m.id }}</td>
            <td>
              <span :class="['badge', m.status]">{{ m.status }}</span>
              <span v-if="m.detail" class="muted"> {{ m.detail }}</span>
            </td>
            <td>{{ m.owned_by }}</td>
          </tr>
        </tbody>
      </table>
    </section>

    <section class="panel">
      <h2>
        Requests <span class="muted">({{ inFlight }} in flight)</span>
      </h2>
      <p v-if="server.state.requests.length === 0" class="hint">No requests yet.</p>
      <table v-else>
        <thead>
          <tr>
            <th>Time</th>
            <th>Request</th>
            <th>Status</th>
            <th>Duration</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="r in server.state.requests" :key="r.id" :title="r.id">
            <td>{{ time(r.started) }}</td>
            <td class="mono">{{ r.method }} {{ r.path }}</td>
            <td :class="{ error: (r.status ?? 0) >= 400 }">{{ r.status ?? '…' }}</td>
            <td>{{ r.duration_ms === null ? '' : `${r.duration_ms} ms` }}</td>
          </tr>
        </tbody>
      </table>
    </section>

    <section class="panel log">
      <h2>Events</h2>
      <ol class="mono">
        <li v-for="(e, i) in server.state.log" :key="i">
          <span class="muted">{{ time(e.time) }}</span> <strong>{{ e.type }}</strong>
          {{ fields(e) }}
        </li>
      </ol>
    </section>
  </div>
</template>

<style scoped>
.console {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(24rem, 1fr));
  gap: 1rem;
  align-content: start;
}
h2 {
  margin: 0 0 0.75rem;
  font-size: 1rem;
}
dl {
  display: grid;
  grid-template-columns: max-content 1fr;
  gap: 0.35rem 1rem;
  margin: 0;
}
dt {
  color: var(--muted);
}
dd {
  margin: 0;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
}
th,
td {
  text-align: left;
  padding: 0.3rem 0.5rem;
  border-bottom: 1px solid var(--border);
}
th {
  color: var(--muted);
  font-weight: 500;
}
.badge {
  padding: 0.05rem 0.4rem;
  border-radius: 0.25rem;
  font-size: 0.75rem;
  background: var(--surface-2);
}
.badge.served {
  background: var(--ok-bg);
  color: var(--ok);
}
.badge.loading {
  background: var(--warn-bg);
  color: var(--warn);
}
.badge.failed {
  background: var(--error-bg);
  color: var(--error);
}
.log {
  grid-column: 1 / -1;
}
.log ol {
  list-style: none;
  margin: 0;
  padding: 0;
  max-height: 24rem;
  overflow-y: auto;
  font-size: 0.8rem;
}
.log li {
  padding: 0.15rem 0;
  border-bottom: 1px solid var(--border);
  word-break: break-all;
}
</style>
