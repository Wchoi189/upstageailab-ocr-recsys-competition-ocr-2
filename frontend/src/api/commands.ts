/**
 * Command Builder API endpoints
 */

import { apiGet, apiPost } from "./client";

export interface CommandSchema {
  id: string;
  name: string;
  fields: Record<string, unknown>;
}

export interface BuildCommandRequest {
  script: string;
  overrides: string[];
}

export interface BuildCommandResponse {
  command: string;
  overrides: string[];
}

/**
 * Get available command schemas
 */
export async function getCommandSchemas(): Promise<CommandSchema[]> {
  return apiGet<CommandSchema[]>("/commands/schemas");
}

/**
 * Build command from overrides
 */
export async function buildCommand(request: BuildCommandRequest): Promise<BuildCommandResponse> {
  return apiPost<BuildCommandRequest, BuildCommandResponse>("/commands/build", request);
}

/**
 * Get command recommendations
 */
export async function getCommandRecommendations(): Promise<unknown[]> {
  return apiGet<unknown[]>("/commands/recommendations");
}
