
import { DocEntry } from '../types';

// Mock Document Registry - In a real app, this would be fetched from the Backend Bridge via bridgeService
const MOCK_REGISTRY: DocEntry[] = [
  {
    path: 'docs/architecture/overview.md',
    udi: 'udi://arch-001',
    title: 'System Architecture Overview',
    lastModified: '2023-10-25',
    contentSnippet: 'The system follows a microservices architecture...'
  },
  {
    path: 'docs/api/endpoints.md',
    udi: 'udi://api-002',
    title: 'API Endpoint Reference',
    lastModified: '2023-10-26',
    contentSnippet: 'All endpoints are prefixed with /v1/...'
  },
  {
    path: 'docs/guides/getting-started.md',
    udi: 'udi://guide-003',
    title: 'Getting Started Guide',
    lastModified: '2023-10-27',
    contentSnippet: 'To begin, install the CLI tool...'
  },
  {
    path: 'docs/internal/protocols.md',
    udi: 'udi://proto-004',
    title: 'Internal Communication Protocols',
    lastModified: '2023-10-28',
    contentSnippet: 'Services communicate via gRPC...'
  }
];

export interface SystemStats {
  totalDocs: number;
  docGrowth: number;
  referenceHealth: number;
  brokenLinks: number;
  pendingMigrations: number;
  distribution: { name: string; valid: number; issues: number }[];
}

export const getRegistry = (): Promise<DocEntry[]> => {
  return new Promise((resolve) => {
    setTimeout(() => resolve(MOCK_REGISTRY), 500);
  });
};

export const resolveUDI = (udi: string): DocEntry | undefined => {
  return MOCK_REGISTRY.find(doc => doc.udi === udi);
};

export const resolvePath = (path: string): DocEntry | undefined => {
  // Simple mock fuzzy match
  return MOCK_REGISTRY.find(doc => doc.path.includes(path) || path.includes(doc.path));
};

export const generateUDI = (): string => {
  return `udi://doc-${Math.floor(Math.random() * 10000).toString().padStart(4, '0')}`;
};

// Mock function to simulate sending updated content back to the Python Bridge
export const commitMigration = (filePath: string, content: string): Promise<{ success: boolean; message: string }> => {
  return new Promise((resolve) => {
    setTimeout(() => {
      console.log(`[BRIDGE] Writing to ${filePath}:`, content.substring(0, 50) + '...');
      resolve({ 
        success: true, 
        message: `Successfully wrote ${content.length} bytes to disk.` 
      });
    }, 1500);
  });
};
