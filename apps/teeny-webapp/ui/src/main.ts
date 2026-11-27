import { createApp } from 'vue'
import { createPinia } from 'pinia'

import App from './App.vue'
import { loadConfig } from './config'
import router from './router'
import { useServerStore } from './stores/server'

const app = createApp(App)

app.use(createPinia())
app.use(router)

loadConfig().then((config) => {
  useServerStore().connect(config.apiUrl)
  app.mount('#app')
})
