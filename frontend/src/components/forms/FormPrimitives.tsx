/**
 * Form primitive components for schema-driven forms.
 * Follows coding standards: 100 char line length, explicit types.
 */

import type { ChangeEvent } from "react";

export interface FormFieldProps {
  value: unknown;
  onChange: (value: unknown) => void;
  label: string;
  help?: string;
  disabled?: boolean;
  error?: string;
}

export interface TextInputProps extends FormFieldProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}

export function TextInput(props: TextInputProps): JSX.Element {
  const { value, onChange, label, help, disabled, error, placeholder } = props;

  const handleChange = (e: ChangeEvent<HTMLInputElement>): void => {
    onChange(e.target.value);
  };

  return (
    <div style={{ marginBottom: "1rem" }}>
      <label style={{ display: "block", marginBottom: "0.25rem", fontWeight: 500 }}>
        {label}
      </label>
      <input
        type="text"
        value={value}
        onChange={handleChange}
        disabled={disabled}
        placeholder={placeholder}
        style={{
          width: "100%",
          padding: "0.5rem",
          border: error ? "1px solid #ef4444" : "1px solid #d1d5db",
          borderRadius: "0.375rem",
          fontSize: "0.875rem",
        }}
      />
      {help && (
        <p style={{ fontSize: "0.75rem", color: "#6b7280", marginTop: "0.25rem" }}>
          {help}
        </p>
      )}
      {error && (
        <p style={{ fontSize: "0.75rem", color: "#ef4444", marginTop: "0.25rem" }}>
          {error}
        </p>
      )}
    </div>
  );
}

export interface CheckboxProps extends FormFieldProps {
  value: boolean;
  onChange: (value: boolean) => void;
}

export function Checkbox(props: CheckboxProps): JSX.Element {
  const { value, onChange, label, help, disabled, error } = props;

  const handleChange = (e: ChangeEvent<HTMLInputElement>): void => {
    onChange(e.target.checked);
  };

  return (
    <div style={{ marginBottom: "1rem" }}>
      <label style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
        <input
          type="checkbox"
          checked={value}
          onChange={handleChange}
          disabled={disabled}
          style={{ width: "1rem", height: "1rem" }}
        />
        <span style={{ fontWeight: 500 }}>{label}</span>
      </label>
      {help && (
        <p style={{ fontSize: "0.75rem", color: "#6b7280", marginTop: "0.25rem" }}>
          {help}
        </p>
      )}
      {error && (
        <p style={{ fontSize: "0.75rem", color: "#ef4444", marginTop: "0.25rem" }}>
          {error}
        </p>
      )}
    </div>
  );
}

export interface SelectBoxProps extends FormFieldProps {
  value: string;
  onChange: (value: string) => void;
  options: string[];
}

export function SelectBox(props: SelectBoxProps): JSX.Element {
  const { value, onChange, label, help, disabled, error, options } = props;

  const handleChange = (e: ChangeEvent<HTMLSelectElement>): void => {
    onChange(e.target.value);
  };

  return (
    <div style={{ marginBottom: "1rem" }}>
      <label style={{ display: "block", marginBottom: "0.25rem", fontWeight: 500 }}>
        {label}
      </label>
      <select
        value={value}
        onChange={handleChange}
        disabled={disabled}
        style={{
          width: "100%",
          padding: "0.5rem",
          border: error ? "1px solid #ef4444" : "1px solid #d1d5db",
          borderRadius: "0.375rem",
          fontSize: "0.875rem",
          backgroundColor: "white",
        }}
      >
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
      {help && (
        <p style={{ fontSize: "0.75rem", color: "#6b7280", marginTop: "0.25rem" }}>
          {help}
        </p>
      )}
      {error && (
        <p style={{ fontSize: "0.75rem", color: "#ef4444", marginTop: "0.25rem" }}>
          {error}
        </p>
      )}
    </div>
  );
}

export interface NumberInputProps extends FormFieldProps {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
}

export function NumberInput(props: NumberInputProps): JSX.Element {
  const { value, onChange, label, help, disabled, error, min, max, step } = props;

  const handleChange = (e: ChangeEvent<HTMLInputElement>): void => {
    onChange(parseFloat(e.target.value));
  };

  return (
    <div style={{ marginBottom: "1rem" }}>
      <label style={{ display: "block", marginBottom: "0.25rem", fontWeight: 500 }}>
        {label}
      </label>
      <input
        type="number"
        value={value}
        onChange={handleChange}
        disabled={disabled}
        min={min}
        max={max}
        step={step}
        style={{
          width: "100%",
          padding: "0.5rem",
          border: error ? "1px solid #ef4444" : "1px solid #d1d5db",
          borderRadius: "0.375rem",
          fontSize: "0.875rem",
        }}
      />
      {help && (
        <p style={{ fontSize: "0.75rem", color: "#6b7280", marginTop: "0.25rem" }}>
          {help}
        </p>
      )}
      {error && (
        <p style={{ fontSize: "0.75rem", color: "#ef4444", marginTop: "0.25rem" }}>
          {error}
        </p>
      )}
    </div>
  );
}

export interface InfoDisplayProps {
  label: string;
  content: string;
}

export function InfoDisplay(props: InfoDisplayProps): JSX.Element {
  const { label, content } = props;

  return (
    <div style={{ marginBottom: "1rem" }}>
      <label style={{ display: "block", marginBottom: "0.25rem", fontWeight: 500 }}>
        {label}
      </label>
      <div
        style={{
          padding: "0.75rem",
          backgroundColor: "#f3f4f6",
          borderRadius: "0.375rem",
          fontSize: "0.875rem",
          whiteSpace: "pre-wrap",
        }}
      >
        {content}
      </div>
    </div>
  );
}
