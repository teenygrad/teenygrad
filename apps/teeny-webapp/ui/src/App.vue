<script setup lang="ts">
import { RouterLink, RouterView } from 'vue-router'

import { useServerStore } from '@/stores/server'

const server = useServerStore()
</script>

<template>
  <header>
    <strong class="brand">teenygrad</strong>
    <nav>
      <RouterLink to="/chat">Chat</RouterLink>
      <RouterLink to="/console">Console</RouterLink>
    </nav>
    <span class="connection" :title="server.client?.baseUrl">
      <span :class="['dot', server.socketState]" />
      {{ server.socketState === 'open' ? 'connected' : server.socketState }}
    </span>
  </header>
  <main>
    <RouterView />
  </main>
</template>

<style>
:root {
  --bg: #f6f7f9;
  --surface: #ffffff;
  --surface-2: #eef0f3;
  --border: #dde1e6;
  --text: #1c2026;
  --muted: #69717d;
  --accent: #2f6fde;
  --on-accent: #ffffff;
  --ok: #1d7a46;
  --ok-bg: #e2f4ea;
  --warn: #8a5a00;
  --warn-bg: #fdf1d6;
  --error: #b3261e;
  --error-bg: #fbe3e1;
  color-scheme: light dark;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14171c;
    --surface: #1c2027;
    --surface-2: #262b34;
    --border: #2f3540;
    --text: #e4e7ec;
    --muted: #9099a6;
    --accent: #4d86ea;
    --ok: #6fd39b;
    --ok-bg: #16352a;
    --warn: #f0c060;
    --warn-bg: #3a2f14;
    --error: #f28b82;
    --error-bg: #3c1e1c;
  }
}
* {
  box-sizing: border-box;
}
html,
body,
#app {
  height: 100%;
  margin: 0;
}
body {
  font-family: system-ui, sans-serif;
  background: var(--bg);
  color: var(--text);
}
#app {
  display: flex;
  flex-direction: column;
}
header {
  display: flex;
  align-items: center;
  gap: 1.5rem;
  padding: 0.6rem 1rem;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
}
header nav {
  display: flex;
  gap: 1rem;
}
header nav a {
  color: var(--muted);
  text-decoration: none;
}
header nav a.router-link-active {
  color: var(--text);
  font-weight: 600;
}
.connection {
  margin-left: auto;
  font-size: 0.85rem;
  color: var(--muted);
}
main {
  flex: 1;
  min-height: 0;
  padding: 1rem;
  overflow: auto;
}
.panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 0.5rem;
  padding: 1rem;
}
input,
select,
textarea,
button {
  font: inherit;
  color: inherit;
}
input,
select,
textarea {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 0.375rem;
  padding: 0.4rem 0.5rem;
}
textarea {
  resize: vertical;
}
button {
  background: var(--accent);
  color: var(--on-accent);
  border: none;
  border-radius: 0.375rem;
  padding: 0.45rem 0.9rem;
  cursor: pointer;
}
button:disabled {
  opacity: 0.5;
  cursor: default;
}
button.link {
  background: none;
  color: var(--accent);
  padding: 0;
}
.mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}
.muted,
.hint {
  color: var(--muted);
}
.hint {
  font-size: 0.85rem;
  margin: 0;
}
.error {
  color: var(--error);
  font-size: 0.85rem;
}
.dot {
  display: inline-block;
  width: 0.5rem;
  height: 0.5rem;
  border-radius: 50%;
  margin-right: 0.35rem;
  background: var(--error);
}
.dot.open {
  background: var(--ok);
}
.dot.connecting {
  background: var(--warn);
}
</style>
