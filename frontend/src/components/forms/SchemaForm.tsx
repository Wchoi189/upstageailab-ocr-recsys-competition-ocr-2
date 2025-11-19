/**
 * Schema-driven form generator.
 * Dynamically renders form fields based on UI element definitions.
 */

import { useState, useEffect, useMemo } from "react";
import type React from "react";
import type { UIElement, FormValues, CommandSchema } from "@/types/schema";
import { evaluateVisibility } from "@/utils/visibility";
import {
  TextInput,
  Checkbox,
  SelectBox,
  NumberInput,
  InfoDisplay,
  Slider,
} from "./FormPrimitives";

export interface SchemaFormProps {
  schema: CommandSchema;
  values: FormValues;
  onChange: (values: FormValues) => void;
  errors?: Record<string, string>;
}

export function SchemaForm(props: SchemaFormProps): React.JSX.Element {
  const { schema, values, onChange, errors = {} } = props;

  // Initialize form values with defaults
  useEffect(() => {
    const initialValues: FormValues = {};
    schema.ui_elements.forEach((element) => {
      if (element.default !== undefined && values[element.key] === undefined) {
        initialValues[element.key] = element.default;
      }
    });

    if (Object.keys(initialValues).length > 0) {
      onChange({ ...initialValues, ...values });
    }
  }, [schema, values, onChange]);

  const handleFieldChange = (key: string, value: unknown): void => {
    onChange({ ...values, [key]: value });
  };

  // Filter elements based on visibility
  const visibleElements = useMemo(() => {
    return schema.ui_elements.filter((element) =>
      evaluateVisibility(element.visible_if, values)
    );
  }, [schema.ui_elements, values]);

  return (
    <div style={{ maxWidth: "600px" }}>
      {visibleElements.map((element) => (
        <FormField
          key={element.key}
          element={element}
          value={values[element.key]}
          onChange={(val) => handleFieldChange(element.key, val)}
          error={errors[element.key]}
        />
      ))}
    </div>
  );
}

interface FormFieldProps {
  element: UIElement;
  value: unknown;
  onChange: (value: unknown) => void;
  error?: string;
}

function FormField(props: FormFieldProps): React.JSX.Element | null {
  const { element, value, onChange, error } = props;

  // Build tooltip from metadata_help_key if available
  const tooltip = element.metadata_help_key
    ? `Metadata: ${element.metadata_help_key}`
    : undefined;

  // Normalize type to handle any case issues
  const fieldType = String(element.type).toLowerCase();

  switch (fieldType) {
    case "text_input":
      return (
        <TextInput
          value={(value as string) || ""}
          onChange={onChange}
          label={element.label}
          help={element.help}
          tooltip={tooltip}
          error={error}
        />
      );

    case "checkbox":
      return (
        <Checkbox
          value={(value as boolean) || false}
          onChange={onChange}
          label={element.label}
          help={element.help}
          tooltip={tooltip}
          error={error}
        />
      );

    case "selectbox":
      return (
        <SelectBox
          value={(value as string) || element.default as string || ""}
          onChange={onChange}
          label={element.label}
          help={element.help}
          tooltip={tooltip}
          error={error}
          options={element.options || []}
        />
      );

    case "number_input":
      return (
        <NumberInput
          value={(value as number) || 0}
          onChange={onChange}
          label={element.label}
          help={element.help}
          tooltip={tooltip}
          error={error}
          min={element.min}
          max={element.max}
          step={element.step}
        />
      );

    case "slider":
      console.log("✅ Slider case matched!", element.key, element);
      return (
        <Slider
          value={(value as number) || (element.default as number) || element.min || 0}
          onChange={onChange}
          label={element.label}
          help={element.help}
          tooltip={tooltip}
          error={error}
          min={element.min}
          max={element.max}
          step={element.step}
        />
      );

    case "info":
      return (
        <InfoDisplay
          label={element.label}
          content={formatInfoContent(element, value)}
        />
      );

    default:
      console.warn(`❌ Unsupported field type: ${element.type} (normalized: ${fieldType})`, element);
      return null;
  }
}

/**
 * Format info field content using template if available.
 */
function formatInfoContent(element: UIElement, value: unknown): string {
  if (element.info_template && typeof value === "object" && value !== null) {
    let content = element.info_template;
    Object.entries(value as Record<string, unknown>).forEach(([k, v]) => {
      content = content.replace(`{${k}}`, String(v));
    });
    return content;
  }
  return String(value || "");
}
