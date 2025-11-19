/**
 * Evaluate visibility condition for UI elements.
 *
 * Supports simple conditions like:
 * - "field == value"
 * - "field != value"
 *
 * @param condition - Visibility condition string (e.g., "field == true")
 * @param values - Current form values
 * @returns true if element should be visible, false otherwise
 */
export function evaluateVisibility(
  condition: string | undefined,
  values: Record<string, unknown>
): boolean {
  if (!condition) {
    return true;
  }

  try {
    const [left, operator, right] = condition.split(" ");
    const fieldValue = values[left];

    switch (operator) {
      case "==":
        return String(fieldValue) === right;
      case "!=":
        return String(fieldValue) !== right;
      default:
        return true;
    }
  } catch (error) {
    console.warn("Failed to evaluate visibility condition", condition, error);
    return true;
  }
}
