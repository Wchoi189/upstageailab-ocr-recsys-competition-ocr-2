/**
 * Schema-driven form generator.
 * Dynamically renders form fields based on UI element definitions.
 */

import { useState, useEffect, useMemo } from "react";
import type { UIElement, FormValues, CommandSchema } from "@/types/schema";
import { evaluateVisibility } from "@/utils/visibility";
import {
  TextInput,
  Checkbox,
  SelectBox,
  NumberInput,
  InfoDisplay,
} from "./FormPrimitives";

export interface SchemaFormProps {
  schema: CommandSchema;
  values: FormValues;
  onChange: (values: FormValues) => void;
  errors?: Record<string, string>;
}

export function SchemaForm(props: SchemaFormProps): JSX.Element {
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

function FormField(props: FormFieldProps): JSX.Element | null {
  const { element, value, onChange, error } = props;

  switch (element.type) {
    case "text_input":
      return (
        <TextInput
          value={(value as string) || ""}
          onChange={onChange}
          label={element.label}
          help={element.help}
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
      console.warn(`Unsupported field type: ${element.type}`);
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
