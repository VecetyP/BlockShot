import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The backend runs on :8000 (override with BACKEND_URL); proxy it so the browser sees one origin.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: process.env.BACKEND_URL ?? "http://127.0.0.1:8000", rewrite: (p) => p.replace(/^\/api/, "") },
    },
  },
});
