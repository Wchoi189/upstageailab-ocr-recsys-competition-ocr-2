
import { APP_CONFIG } from "../config/constants";

export interface BridgeStatus {
  status: string;
  version: string;
  cwd: string;
  agentqms_root: string;
  config?: any;
}

export interface FileItem {
  name: string;
  type: 'directory' | 'file';
  size: number;
  last_modified: string;
}

export interface FileListResponse {
  path: string;
  items: FileItem[];
}

export interface FileReadResponse {
  path: string;
  content: string;
  encoding: string;
}

export interface ToolExecutionResult {
  success: boolean;
  output: string;
  error?: string | null;
  return_code?: number;
}

// --- New Interfaces for API v1 ---

export interface Artifact {
  id: string;
  type: 'implementation_plan' | 'assessment' | 'audit' | 'bug_report';
  title: string;
  status: string;
  path: string;
  created_at: string;
  content?: string; // Content might be included in detail view
  frontmatter?: any;
}

export interface TrackingStatus {
  kind: string;
  status: string;
  success: boolean;
  error?: string;
}

export interface ArtifactListResponse {
  items: Artifact[];
  total: number;
}

export interface ArtifactCreate {
  type: 'implementation_plan' | 'assessment' | 'audit' | 'bug_report';
  title: string;
  content: string;
}

export interface ArtifactUpdate {
  content?: string;
  frontmatter_updates?: Record<string, any>;
}

export interface ComplianceResult {
  status: string;
  violations?: any[];
  // Add other fields as needed based on actual response
}

const API_URL = APP_CONFIG.API.BRIDGE_URL;

async function fetchJson<T>(endpoint: string, options?: RequestInit): Promise<T> {
  try {
    const res = await fetch(`${API_URL}${endpoint}`, options);
    if (!res.ok) {
      const errorBody = await res.text();
      throw new Error(`Bridge API Error (${res.status}): ${errorBody}`);
    }
    return await res.json() as T;
  } catch (err) {
    console.error(`Bridge Service Error [${endpoint}]:`, err);
    throw err;
  }
}

export const bridgeService = {
  /**
   * Check if the Python Bridge server is online.
   */
  getStatus: async (): Promise<BridgeStatus> => {
    return fetchJson<BridgeStatus>('/status');
  },

  /**
   * List files in a directory relative to project root.
   */
  listFiles: async (path: string = '.'): Promise<FileListResponse> => {
    const params = new URLSearchParams({ path });
    return fetchJson<FileListResponse>(`/fs/list?${params.toString()}`);
  },

  /**
   * Read file content.
   */
  readFile: async (path: string): Promise<FileReadResponse> => {
    const params = new URLSearchParams({ path });
    return fetchJson<FileReadResponse>(`/fs/read?${params.toString()}`);
  },

  /**
   * Write content to a file.
   */
  writeFile: async (path: string, content: string): Promise<{ success: boolean; bytes_written: number }> => {
    return fetchJson<{ success: boolean; bytes_written: number }>('/fs/write', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path, content }),
    });
  },

  /**
   * Execute an AgentQMS tool script.
   */
  executeTool: async (tool_id: string, args: Record<string, any>): Promise<ToolExecutionResult> => {
    return fetchJson<ToolExecutionResult>('/v1/tools/exec', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tool_id, args }),
    });
  },

  /**
   * Get the Registry Index.
   */
  getRegistry: async (): Promise<any> => {
      return fetchJson<any>('/registry/index');
  },

  // --- API v1 Methods ---

  /**
   * Get System Health
   */
  getHealth: async (): Promise<{ status: string }> => {
    return fetchJson<{ status: string }>('/v1/health');
  },

  /**
   * Get System Version
   */
  getVersion: async (): Promise<{ version: string }> => {
    return fetchJson<{ version: string }>('/v1/version');
  },

  /**
   * List Artifacts
   */
  listArtifacts: async (params?: { type?: string; status?: string; limit?: number }): Promise<ArtifactListResponse> => {
    const queryParams = new URLSearchParams();
    if (params?.type) queryParams.append('type', params.type);
    if (params?.status) queryParams.append('status', params.status);
    if (params?.limit) queryParams.append('limit', params.limit.toString());

    return fetchJson<ArtifactListResponse>(`/v1/artifacts?${queryParams.toString()}`);
  },

  /**
   * Get Artifact Details
   */
  getArtifact: async (id: string): Promise<Artifact> => {
    return fetchJson<Artifact>(`/v1/artifacts/${id}`);
  },

  /**
   * Create New Artifact
   */
  createArtifact: async (data: ArtifactCreate): Promise<Artifact> => {
    return fetchJson<Artifact>('/v1/artifacts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
  },

  /**
   * Update Artifact
   */
  updateArtifact: async (id: string, data: ArtifactUpdate): Promise<Artifact> => {
    return fetchJson<Artifact>(`/v1/artifacts/${id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
  },

  /**
   * Delete (Archive) Artifact
   */
  deleteArtifact: async (id: string): Promise<void> => {
    return fetchJson<void>(`/v1/artifacts/${id}`, {
      method: 'DELETE',
    });
  },

  /**
   * Run Compliance Validation
   */
  validateCompliance: async (target: string = 'all'): Promise<ComplianceResult> => {
    const params = new URLSearchParams({ target });
    return fetchJson<ComplianceResult>(`/v1/compliance/validate?${params.toString()}`);
  },

  /**
   * Get Tracking Database Status
   */
  getTrackingStatus: async (kind: string = 'all'): Promise<TrackingStatus> => {
    const params = new URLSearchParams({ kind });
    return fetchJson<TrackingStatus>(`/v1/tracking/status?${params.toString()}`);
  }
};
