/**
 * Type definitions for command builder schemas.
 * Mirrors the Python schema structure from YAML files.
 */

export type SchemaId = "train" | "test" | "predict";

export type UIElementType =
  | "text_input"
  | "number_input"
  | "checkbox"
  | "selectbox"
  | "multiselect"
  | "slider"
  | "info";

export interface ValidationRules {
  required?: boolean;
  required_if?: string;
  min_length?: number;
  max_length?: number;
  min?: number;
  max?: number;
  pattern?: string;
}

export interface UIElement {
  key: string;
  type: UIElementType;
  label: string;
  default?: unknown;
  help?: string;
  hydra_override?: string;
  visible_if?: string;
  validation?: ValidationRules;
  options_source?: string;
  filter_by_architecture_key?: string;
  metadata_default_key?: string;
  metadata_help_key?: string;
  metadata_info_key?: string;
  info_template?: string;
  architecture_fallback?: string;
  options?: string[];
  min?: number;
  max?: number;
  step?: number;
}

export interface CommandSchema {
  title: string;
  constant_overrides: string[];
  ui_elements: UIElement[];
}

export interface SchemaSummary {
  id: SchemaId;
  label: string;
  script: string;
  description?: string;
}

export interface BuildCommandRequest {
  schema_id: SchemaId;
  values: Record<string, unknown>;
  append_model_suffix?: boolean;
}

export interface BuildCommandResponse {
  command: string;
  overrides: string[];
  constant_overrides: string[];
  validation_error?: string;
}

export type FormValues = Record<string, unknown>;

export interface Recommendation {
  id: string;
  title: string;
  description: string;
  architecture?: string;
  parameters: Record<string, unknown>;
}
