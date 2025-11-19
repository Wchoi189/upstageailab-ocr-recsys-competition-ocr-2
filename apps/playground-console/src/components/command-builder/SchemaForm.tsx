"use client";

import {
  FormControl,
  FormLabel,
  FormHelperText,
  Input,
  NumberInput,
  NumberInputField,
  Select,
  Switch,
  Text,
  VStack,
  Box,
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
  Tooltip,
} from "@chakra-ui/react";
import { useEffect, useMemo } from "react";
import type React from "react";

import type { CommandSchema, FormValues, UIElement } from "@/types/schema";
import { evaluateVisibility } from "@/utils/visibility";

interface SchemaFormProps {
  schema: CommandSchema;
  values: FormValues;
  onChange: (values: FormValues) => void;
  errors?: Record<string, string>;
}

export function SchemaForm({ schema, values, onChange, errors = {} }: SchemaFormProps): React.JSX.Element {
  useEffect(() => {
    const withDefaults: FormValues = { ...values };
    schema.ui_elements.forEach((element) => {
      if (withDefaults[element.key] === undefined && element.default !== undefined) {
        withDefaults[element.key] = element.default;
      }
    });
    if (Object.keys(withDefaults).length !== Object.keys(values).length) {
      onChange(withDefaults);
    }
  }, [schema, values, onChange]);

  const visibleElements = useMemo(
    () => schema.ui_elements.filter((element) => evaluateVisibility(element.visible_if, values)),
    [schema.ui_elements, values]
  );

  return (
    <VStack align="stretch" spacing={5} maxW="640px">
      {visibleElements.map((element) => (
        <FieldRenderer
          key={element.key}
          element={element}
          value={values[element.key]}
          onChange={(value) => onChange({ ...values, [element.key]: value })}
          error={errors[element.key]}
        />
      ))}
    </VStack>
  );
}

interface FieldRendererProps {
  element: UIElement;
  value: unknown;
  onChange: (value: unknown) => void;
  error?: string;
}

function FieldRenderer({ element, value, onChange, error }: FieldRendererProps): React.JSX.Element | null {
  const tooltip = element.help;
  const fieldType = String(element.type).toLowerCase();

  const commonProps = {
    element,
    value,
    onChange,
    error,
    tooltip,
  };

  switch (fieldType) {
    case "text_input":
      return <TextInputField {...commonProps} />;
    case "number_input":
      return <NumberInputField {...commonProps} />;
    case "checkbox":
      return <CheckboxField {...commonProps} />;
    case "selectbox":
      return <SelectField {...commonProps} />;
    case "slider":
      return <SliderField {...commonProps} />;
    case "info":
      return <InfoField {...commonProps} />;
    default:
      return null;
  }
}

function Wrapper({
  element,
  children,
  error,
  tooltip,
}: {
  element: UIElement;
  children: React.ReactNode;
  error?: string;
  tooltip?: string;
}): React.JSX.Element {
  return (
    <FormControl>
      <Box display="flex" alignItems="center" gap={2}>
        <FormLabel m={0}>{element.label}</FormLabel>
        {tooltip && (
          <Text fontSize="xs" color="text.muted">
            {tooltip}
          </Text>
        )}
      </Box>
      {children}
      {element.help && (
        <FormHelperText color="text.muted">{element.help}</FormHelperText>
      )}
      {error && (
        <Text color="red.500" fontSize="sm" mt={1}>
          {error}
        </Text>
      )}
    </FormControl>
  );
}

function TextInputField(props: FieldRendererProps & { tooltip?: string }): React.JSX.Element {
  const { element, value, onChange, error, tooltip } = props;
  return (
    <Wrapper element={element} error={error} tooltip={tooltip}>
      <Input value={(value as string) ?? ""} onChange={(event) => onChange(event.target.value)} />
    </Wrapper>
  );
}

function NumberInputField(props: FieldRendererProps & { tooltip?: string }): React.JSX.Element {
  const { element, value, onChange, error, tooltip } = props;
  return (
    <Wrapper element={element} error={error} tooltip={tooltip}>
      <NumberInput
        value={Number(value ?? element.default ?? 0)}
        min={element.min}
        max={element.max}
        step={element.step}
        onChange={(val) => onChange(Number(val))}
      >
        <NumberInputField />
      </NumberInput>
    </Wrapper>
  );
}

function CheckboxField(props: FieldRendererProps & { tooltip?: string }): React.JSX.Element {
  const { element, value, onChange, error, tooltip } = props;
  return (
    <Wrapper element={element} error={error} tooltip={tooltip}>
      <Switch isChecked={Boolean(value ?? element.default)} onChange={(event) => onChange(event.target.checked)} />
    </Wrapper>
  );
}

function SelectField(props: FieldRendererProps & { tooltip?: string }): React.JSX.Element {
  const { element, value, onChange, error, tooltip } = props;
  return (
    <Wrapper element={element} error={error} tooltip={tooltip}>
      <Select
        value={(value as string) ?? (element.default as string) ?? ""}
        onChange={(event) => onChange(event.target.value)}
      >
        {(element.options ?? []).map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </Select>
    </Wrapper>
  );
}

function SliderField(props: FieldRendererProps & { tooltip?: string }): React.JSX.Element {
  const { element, value, onChange, error, tooltip } = props;
  const sliderValue = Number(
    value ?? element.default ?? element.min ?? 0
  );

  return (
    <Wrapper element={element} error={error} tooltip={tooltip}>
      <Tooltip label={sliderValue} placement="top" hasArrow>
        <Slider
          value={sliderValue}
          min={element.min}
          max={element.max}
          step={element.step}
          onChange={(val) => onChange(val)}
        >
          <SliderTrack>
            <SliderFilledTrack />
          </SliderTrack>
          <SliderThumb />
        </Slider>
      </Tooltip>
    </Wrapper>
  );
}

function InfoField({ element, value }: FieldRendererProps): React.JSX.Element {
  return (
    <Box border="1px solid" borderColor="border.subtle" borderRadius="md" p={4} bg="surface.panel">
      <Text fontWeight="semibold" mb={1}>
        {element.label}
      </Text>
      <Text fontSize="sm" color="text.muted">
        {typeof value === "string" ? value : element.help}
      </Text>
    </Box>
  );
}
