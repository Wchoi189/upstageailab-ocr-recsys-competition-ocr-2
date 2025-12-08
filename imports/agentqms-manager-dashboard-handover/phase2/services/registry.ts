import { DocEntry } from '../types';
import { bridgeService } from './bridgeService';

export interface SystemStats {
  totalDocs: number;
  docGrowth: number;
  referenceHealth: number;
  brokenLinks: number;
  pendingMigrations: number;
  distribution: { name: string; valid: number; issues: number }[];
}

/**
 * Registry Service
 * Adapts the backend Bridge API for UI consumption.
 */

export const getRegistry = async (): Promise<DocEntry[]> => {
  try {
    const rawData = await bridgeService.getRegistry();
    // Assuming backend returns { artifacts: [...] } or just an array
    return rawData.artifacts || [];
  } catch (e) {
    console.warn("Bridge unconnected. Returning empty registry.", e);
    return [];
  }
};

export const resolveUDI = async (udi: string): Promise<DocEntry | undefined> => {
  const registry = await getRegistry();
  return registry.find(doc => doc.udi === udi);
};

export const resolvePath = async (path: string): Promise<DocEntry | undefined> => {
  const registry = await getRegistry();
  // Simple fuzzy match
  return registry.find(doc => doc.path.includes(path) || path.includes(doc.path));
};

export const generateUDI = (): string => {
  return `udi://doc-${Math.floor(Math.random() * 100000).toString().padStart(6, '0')}`;
};

export const getSystemStats = async (): Promise<SystemStats> => {
  const docs = await getRegistry();
  
  return {
    totalDocs: docs.length,
    docGrowth: 0,
    referenceHealth: 100, // Placeholder for future logic
    brokenLinks: 0,
    pendingMigrations: 0,
    distribution: [
      { name: 'Docs', valid: docs.length, issues: 0 },
      { name: 'Links', valid: 0, issues: 0 },
    ]
  };
};

export const commitMigration = async (filePath: string, content: string): Promise<{ success: boolean; message: string }> => {
  try {
    const result = await bridgeService.writeFile(filePath, content);
    return {
      success: result.success,
      message: `Successfully wrote ${result.bytes_written} bytes to disk.`
    };
  } catch (error) {
    console.error("Migration failed:", error);
    return {
      success: false,
      message: error instanceof Error ? error.message : "Failed to write to disk via Bridge."
    };
  }
};