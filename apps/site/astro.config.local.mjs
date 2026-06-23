import { defineConfig } from "astro/config";
import node from "@astrojs/node";

// Local-dev build config: uses the Node adapter so `astro build` +
// `node ./dist/server/entry.mjs` runs the site without Vite's first-
// request compile. The default astro.config.mjs keeps the Vercel
// adapter for production deploys.
export default defineConfig({
  output: "server",
  adapter: node({ mode: "standalone" }),
  server: {
    port: 4321,
  },
});
