import { defineConfig } from "astro/config";
import vercel from "@astrojs/vercel";

export default defineConfig({
  output: "static",
  adapter: vercel(),
  trailingSlash: "never",
  server: {
    port: 4321,
  },
  vite: {
    server: {
      open: false,
    },
  },
});
