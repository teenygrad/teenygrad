import { createRouter, createWebHistory } from 'vue-router'

import ChatView from '@/views/ChatView.vue'
import ConsoleView from '@/views/ConsoleView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    { path: '/', redirect: '/chat' },
    { path: '/chat', component: ChatView },
    { path: '/console', component: ConsoleView },
    { path: '/:rest(.*)*', redirect: '/chat' },
  ],
})

export default router
