import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Nothing clever here on purpose. The API base URL comes from an env var so
// the same build runs against localhost during development and against
// Railway in production, without a code change.
export default defineConfig({
  plugins: [react()],
});
