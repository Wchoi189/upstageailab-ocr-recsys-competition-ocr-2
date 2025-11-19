import { extendTheme, type ThemeConfig } from "@chakra-ui/react";

const config: ThemeConfig = {
  initialColorMode: "light",
  useSystemColorMode: false,
};

const colors = {
  brand: {
    50: "#e5f1ff",
    100: "#bfd8ff",
    200: "#99bfff",
    300: "#73a7ff",
    400: "#4d8eff",
    500: "#3375e6",
    600: "#265cb4",
    700: "#194281",
    800: "#0d294f",
    900: "#020f1f",
  },
};

const semanticTokens = {
  colors: {
    "surface.background": {
      default: "gray.50",
      _dark: "gray.900",
    },
    "surface.panel": {
      default: "white",
      _dark: "gray.800",
    },
    "text.muted": {
      default: "gray.600",
      _dark: "gray.300",
    },
    "border.subtle": {
      default: "gray.200",
      _dark: "gray.700",
    },
  },
};

const styles = {
  global: {
    "html, body": {
      bg: "surface.background",
      color: "gray.900",
      _dark: {
        color: "gray.50",
      },
    },
  },
};

const fonts = {
  heading: "'Inter', system-ui, sans-serif",
  body: "'Inter', system-ui, sans-serif",
};

const theme = extendTheme({
  config,
  colors,
  semanticTokens,
  styles,
  fonts,
});

export default theme;
