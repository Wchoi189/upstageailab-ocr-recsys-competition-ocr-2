import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    // Optimize worker bundle size
    rollupOptions: {
      output: {
        // Separate worker chunks for better caching
        manualChunks: (id) => {
          // Group worker dependencies separately
          if (id.includes("/workers/")) {
            return "workers";
          }
          // Group transform utilities together
          if (id.includes("/workers/transforms") || id.includes("/workers/types")) {
            return "worker-utils";
          }
          // Group vendor dependencies
          if (id.includes("node_modules")) {
            if (id.includes("react") || id.includes("react-dom")) {
              return "react-vendor";
            }
            return "vendor";
          }
        },
      },
    },
    // Optimize chunk size warnings
    chunkSizeWarningLimit: 1000,
  },
  worker: {
    // Use ES modules for workers (better tree-shaking)
    format: "es",
    // Rollup options for workers
    rollupOptions: {
      output: {
        // Minimize worker bundle size
        format: "es",
        // Separate worker chunks for better caching
        entryFileNames: "workers/[name].js",
        chunkFileNames: "workers/[name]-[hash].js",
      },
    },
  },
  optimizeDeps: {
    // Pre-bundle worker dependencies for faster startup
    include: ["react", "react-dom"],
    // Exclude worker files from pre-bundling (they're loaded dynamically)
    exclude: ["workers"],
  },
});
