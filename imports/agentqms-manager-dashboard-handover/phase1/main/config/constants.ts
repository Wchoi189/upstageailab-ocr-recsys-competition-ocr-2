
// Centralized Configuration Constants
// All hardcoded strings and environment variable fallbacks reside here.

export const APP_CONFIG = {
  // System Defaults
  DEFAULTS: {
    BRANCH_NAME: process.env.DEFAULT_BRANCH || 'feature/agent-qms-update',
    TIMEZONE: process.env.DEFAULT_TIMEZONE || 'KST',
    AUTHOR: process.env.DEFAULT_AUTHOR || '',
    MODEL: process.env.DEFAULT_MODEL || 'gemini-2.5-flash',
    API_TIMEOUT_MS: Number(process.env.API_TIMEOUT_MS) || 30000,
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
    ENABLE_REAL_FS: process.env.ENABLE_REAL_FS === 'true', // Set to true once Python Bridge is active
    MOCK_LATENCY_MS: 800,
  },
  
  API: {
    BRIDGE_URL: process.env.BRIDGE_API_URL || 'http://localhost:8000/api',
  }
};
