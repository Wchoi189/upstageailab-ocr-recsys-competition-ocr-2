/**
 * Form primitive components for schema-driven forms.
 * Follows coding standards: 100 char line length, explicit types.
 */

import type { ChangeEvent } from "react";
import type React from "react";
import { InfoIcon } from "@/components/ui/Tooltip";

export interface FormFieldProps {
  value: unknown;
  onChange: (value: unknown) => void;
  label: string;
  help?: string;
  tooltip?: string;
  disabled?: boolean;
  error?: string;
}

export interface TextInputProps extends FormFieldProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}

export function TextInput(props: TextInputProps): React.JSX.Element {
  const { value, onChange, label, help, disabled, error, placeholder, tooltip } = props;

  const handleChange = (e: ChangeEvent<HTMLInputElement>): void => {
    onChange(e.target.value);
  };

  return (
    <div style={{ marginBottom: "1rem" }}>
      <label
        style={{
          display: "flex",
          alignItems: "center",
          gap: "0.5rem",
          marginBottom: "0.25rem",
          fontWeight: 500,
        }}
      >
        <span>{label}</span>
        {tooltip && <InfoIcon tooltip={tooltip} />}
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
          fontFamily: "inherit",
          height: "2.5rem",
          lineHeight: "1.5",
          boxSizing: "border-box",
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

export function Checkbox(props: CheckboxProps): React.JSX.Element {
  const { value, onChange, label, help, disabled, error, tooltip } = props;

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
        {tooltip && <InfoIcon tooltip={tooltip} />}
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
  options: string[] | Array<{ label: string; value: string }>;
}

export function SelectBox(props: SelectBoxProps): React.JSX.Element {
  const { value, onChange, label, help, disabled, error, options, tooltip } = props;

  const handleChange = (e: ChangeEvent<HTMLSelectElement>): void => {
    onChange(e.target.value);
  };

  return (
    <div style={{ marginBottom: "1rem" }}>
      <label
        style={{
          display: "flex",
          alignItems: "center",
          gap: "0.5rem",
          marginBottom: "0.25rem",
          fontWeight: 500,
        }}
      >
        <span>{label}</span>
        {tooltip && <InfoIcon tooltip={tooltip} />}
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
          color: "#213547",
        }}
      >
        {options.map((option, index) => {
          // Handle both string and object options
          if (typeof option === "string") {
            return (
              <option
                key={`opt-${index}-${option}`}
                value={option}
                style={{ color: "#213547", backgroundColor: "white" }}
              >
                {option}
              </option>
            );
          } else if (option && typeof option === "object" && "value" in option && "label" in option) {
            return (
              <option
                key={`opt-${index}-${option.value}`}
                value={String(option.value)}
                style={{ color: "#213547", backgroundColor: "white" }}
              >
                {String(option.label)}
              </option>
            );
          } else {
            // Fallback for unexpected formats
            const optionStr = String(option);
            return (
              <option
                key={`opt-${index}-${optionStr}`}
                value={optionStr}
                style={{ color: "#213547", backgroundColor: "white" }}
              >
                {optionStr}
              </option>
            );
          }
        })}
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

export function NumberInput(props: NumberInputProps): React.JSX.Element {
  const { value, onChange, label, help, disabled, error, min, max, step, tooltip } = props;

  const handleChange = (e: ChangeEvent<HTMLInputElement>): void => {
    onChange(parseFloat(e.target.value));
  };

  return (
    <div style={{ marginBottom: "1rem" }}>
      <label
        style={{
          display: "flex",
          alignItems: "center",
          gap: "0.5rem",
          marginBottom: "0.25rem",
          fontWeight: 500,
        }}
      >
        <span>{label}</span>
        {tooltip && <InfoIcon tooltip={tooltip} />}
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

export function InfoDisplay(props: InfoDisplayProps): React.JSX.Element {
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

export interface SliderProps extends FormFieldProps {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
}

export function Slider(props: SliderProps): React.JSX.Element {
  const {
    value,
    onChange,
    label,
    help,
    disabled,
    error,
    min = 0,
    max = 100,
    step = 1,
    tooltip,
  } = props;

  const handleChange = (e: ChangeEvent<HTMLInputElement>): void => {
    onChange(parseFloat(e.target.value));
  };

  return (
    <div style={{ marginBottom: "1rem" }}>
      <label
        style={{
          display: "flex",
          alignItems: "center",
          gap: "0.5rem",
          marginBottom: "0.25rem",
          fontWeight: 500,
        }}
      >
        <span>
          {label}: {value}
        </span>
        {tooltip && <InfoIcon tooltip={tooltip} />}
      </label>
      <input
        type="range"
        value={value}
        onChange={handleChange}
        disabled={disabled}
        min={min}
        max={max}
        step={step}
        style={{
          width: "100%",
          padding: "0.5rem 0",
        }}
      />
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: "0.75rem",
          color: "#6b7280",
          marginTop: "0.25rem",
        }}
      >
        <span>{min}</span>
        <span>{max}</span>
      </div>
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
