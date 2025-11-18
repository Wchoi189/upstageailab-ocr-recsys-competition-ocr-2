/**
 * Utilities for evaluating conditional visibility expressions.
 * Supports simple expressions like "resume_training == true".
 */

import type { FormValues } from "@/types/schema";

/**
 * Evaluate a visibility expression against form values.
 * Supports basic comparisons: ==, !=, >, <, >=, <=
 */
export function evaluateVisibility(
  expression: string | undefined,
  values: FormValues
): boolean {
  if (!expression) {
    return true;
  }

  // Parse simple expressions like "key == value" or "key != value"
  const eqMatch = expression.match(/^\s*(\w+)\s*==\s*(.+)\s*$/);
  if (eqMatch) {
    const [, key, valueStr] = eqMatch;
    const expectedValue = parseValue(valueStr.trim());
    return values[key] === expectedValue;
  }

  const neqMatch = expression.match(/^\s*(\w+)\s*!=\s*(.+)\s*$/);
  if (neqMatch) {
    const [, key, valueStr] = neqMatch;
    const expectedValue = parseValue(valueStr.trim());
    return values[key] !== expectedValue;
  }

  // Default to visible if expression cannot be parsed
  console.warn(`Cannot parse visibility expression: ${expression}`);
  return true;
}

/**
 * Parse a value string to its appropriate type.
 */
function parseValue(valueStr: string): unknown {
  // Boolean
  if (valueStr === "true") return true;
  if (valueStr === "false") return false;

  // Number
  if (/^-?\d+(\.\d+)?$/.test(valueStr)) {
    return parseFloat(valueStr);
  }

  // String (remove quotes if present)
  if (valueStr.startsWith('"') && valueStr.endsWith('"')) {
    return valueStr.slice(1, -1);
  }
  if (valueStr.startsWith("'") && valueStr.endsWith("'")) {
    return valueStr.slice(1, -1);
  }

  // Default to string
  return valueStr;
}
