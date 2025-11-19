import { apiGet, apiPost } from "@/api/client";
import type {
  SchemaSummary,
  CommandSchema,
  BuildCommandRequest,
  BuildCommandResponse,
  SchemaId,
  Recommendation,
} from "@/types/schema";

export function getCommandSchemas(): Promise<SchemaSummary[]> {
  return apiGet<SchemaSummary[]>("/commands/schemas");
}

export function getSchemaDetails(schemaId: SchemaId): Promise<CommandSchema> {
  return apiGet<CommandSchema>(`/commands/schemas/${schemaId}`);
}

export function buildCommand(request: BuildCommandRequest): Promise<BuildCommandResponse> {
  return apiPost<BuildCommandRequest, BuildCommandResponse>("/commands/build", request);
}

export function getCommandRecommendations(architecture?: string): Promise<Recommendation[]> {
  const query = architecture ? `?architecture=${architecture}` : "";
  return apiGet<Recommendation[]>(`/commands/recommendations${query}`);
}
