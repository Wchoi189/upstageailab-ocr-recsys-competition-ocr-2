import { createSystem, defaultConfig, defineConfig } from "@chakra-ui/react";

const customConfig = defineConfig({
  theme: {
    tokens: {
      colors: {
        brand: {
          50: { value: "#e5f1ff" },
          100: { value: "#bfd8ff" },
          200: { value: "#99bfff" },
          300: { value: "#73a7ff" },
          400: { value: "#4d8eff" },
          500: { value: "#3375e6" },
          600: { value: "#265cb4" },
          700: { value: "#194281" },
          800: { value: "#0d294f" },
          900: { value: "#020f1f" },
        },
      },
      fonts: {
        heading: { value: "'Inter', system-ui, sans-serif" },
        body: { value: "'Inter', system-ui, sans-serif" },
      },
    },
    semanticTokens: {
      colors: {
        "surface.background": {
          value: { base: "{colors.gray.50}", _dark: "{colors.gray.900}" },
        },
        "surface.panel": {
          value: { base: "white", _dark: "{colors.gray.800}" },
        },
        "text.muted": {
          value: { base: "{colors.gray.600}", _dark: "{colors.gray.300}" },
        },
        "border.subtle": {
          value: { base: "{colors.gray.200}", _dark: "{colors.gray.700}" },
        },
      },
    },
  },
});

const theme = createSystem(defaultConfig, customConfig);

export default theme;
