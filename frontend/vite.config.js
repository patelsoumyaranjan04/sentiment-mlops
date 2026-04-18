// frontend/vite.config.js
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3002,
    proxy: {
      "/predict": "http://localhost:8000",
      "/health":  "http://localhost:8000",
      "/ready":   "http://localhost:8000",
      "/metrics": "http://localhost:8000",
    },
  },
});
