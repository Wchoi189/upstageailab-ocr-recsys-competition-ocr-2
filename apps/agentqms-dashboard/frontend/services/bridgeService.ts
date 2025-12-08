
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
  tool_id: string;
  exit_code: number;
  stdout: string;
  stderr: string;
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
    return fetchJson<ToolExecutionResult>('/tools/exec', {
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
  }
};
