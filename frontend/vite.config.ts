import react from "@vitejs/plugin-react";
import { Agent } from "node:http";
import { defineConfig } from "vite";

// The backend runs on :8000; proxy it so the browser sees one origin.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // keepAlive: reuse connections, or job polling leaves thousands of closed sockets on Windows.
    proxy: { "/api": { target: "http://127.0.0.1:8000", agent: new Agent({ keepAlive: true }), rewrite: (p) => p.replace(/^\/api/, "") } },
  },
});
