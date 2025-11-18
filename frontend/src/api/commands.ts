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
 * Get command recommendations
 */
export async function getCommandRecommendations(): Promise<unknown[]> {
  return apiGet<unknown[]>("/commands/recommendations");
}
