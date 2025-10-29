import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  server: {
    host: true, // allows mobile devices to connect
    https: false
  },
  build: {
    outDir: "dist"
  }
});
