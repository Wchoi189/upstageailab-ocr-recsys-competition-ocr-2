/**
 * Command Builder API endpoints
 */

import { apiGet, apiPost } from "./client";
import type {
  SchemaSummary,
  CommandSchema,
  BuildCommandRequest,
  BuildCommandResponse,
  SchemaId,
  Recommendation,
} from "@/types/schema";

/**
 * Get available command schema summaries
 */
export async function getCommandSchemas(): Promise<SchemaSummary[]> {
  return apiGet<SchemaSummary[]>("/commands/schemas");
}

/**
 * Get full schema definition with UI elements
 */
export async function getSchemaDetails(schemaId: SchemaId): Promise<CommandSchema> {
  return apiGet<CommandSchema>(`/commands/schemas/${schemaId}`);
}

/**
 * Build command from values
 */
export async function buildCommand(
  request: BuildCommandRequest
): Promise<BuildCommandResponse> {
  return apiPost<BuildCommandRequest, BuildCommandResponse>(
    "/commands/build",
    request
  );
}

/**
 * Get command recommendations, optionally filtered by architecture
 */
export async function getCommandRecommendations(
  architecture?: string
): Promise<Recommendation[]> {
  const query = architecture ? `?architecture=${architecture}` : "";
  return apiGet<Recommendation[]>(`/commands/recommendations${query}`);
}
