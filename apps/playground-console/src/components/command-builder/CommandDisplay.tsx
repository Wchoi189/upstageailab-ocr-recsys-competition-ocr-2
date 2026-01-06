"use client";

import { Button, Code, Separator, Flex, Stack, Text } from "@chakra-ui/react";
import type React from "react";

interface CommandDisplayProps {
  command: string;
  overrides: string[];
  constantOverrides: string[];
  validationError?: string;
  onCopy?: () => void;
}

export function CommandDisplay(props: CommandDisplayProps): React.JSX.Element {
  const { command, overrides, constantOverrides, validationError, onCopy } = props;

  return (
    <Stack gap={4} border="1px solid" borderColor="border.subtle" borderRadius="xl" p={6} bg="surface.panel">
      <Flex justify="space-between" align="center">
        <Text fontWeight="semibold">Generated command</Text>
        <Button size="sm" onClick={onCopy}>
          Copy
        </Button>
      </Flex>
      <Code whiteSpace="pre-wrap" fontSize="sm" p={4} bg="gray.900" color="green.200" borderRadius="md">
        {command}
      </Code>
      {validationError && (
        <Text color="red.500" fontSize="sm">
          {validationError}
        </Text>
      )}
      <Separator />
      <Stack gap={2}>
        <Text fontSize="sm" color="text.muted">
          Overrides
        </Text>
        {overrides.map((item) => (
          <Code key={item}>{item}</Code>
        ))}
        <Text fontSize="sm" color="text.muted" pt={2}>
          Constants
        </Text>
        {constantOverrides.map((item) => (
          <Code key={item} variant="subtle">
            {item}
          </Code>
        ))}
      </Stack>
    </Stack>
  );
}
