
// Centralized Configuration Constants
// All hardcoded strings and environment variable fallbacks reside here.

export const APP_CONFIG = {
  // System Defaults
  DEFAULTS: {
    BRANCH_NAME: 'feature/agent-qms-update',
    TIMEZONE: 'KST',
    AUTHOR: '',
    MODEL: 'gemini-2.5-flash',
    API_TIMEOUT_MS: 30000,
  },

  // Local Storage Keys
  STORAGE: {
    SETTINGS: 'agentqms_settings',
    THEME: 'agentqms_theme',
    SIDEBAR_STATE: 'agentqms_sidebar',
  },

  // Framework Paths (Virtual Representation)
  PATHS: {
    ROOT: 'AgentQMS',
    TOOLS: 'AgentQMS/agent_tools',
    REGISTRY: 'AgentQMS/registry',
    MODULES: 'AgentQMS/modules',
  },

  // Feature Flags & Endpoints
  FEATURES: {
    ENABLE_REAL_FS: true,
    MOCK_LATENCY_MS: 800,
  },
  
  API: {
    BRIDGE_URL: '/api', // Relative path to use Vite proxy
  }
};
