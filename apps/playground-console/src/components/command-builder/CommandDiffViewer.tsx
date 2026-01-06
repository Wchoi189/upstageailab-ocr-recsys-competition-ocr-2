"use client";

import { Box, Code, Stack, Text } from "@chakra-ui/react";
import type React from "react";

interface CommandDiffViewerProps {
  before: string;
  after: string;
}

export function CommandDiffViewer({ before, after }: CommandDiffViewerProps): React.JSX.Element {
  return (
    <Stack gap={3} border="1px solid" borderColor="border.subtle" borderRadius="lg" p={4} bg="surface.panel">
      <Text fontWeight="semibold">Command diff</Text>
      <Stack direction={{ base: "column", md: "row" }} gap={4} align="stretch">
        <DiffPanel label="Previous" value={before} />
        <DiffPanel label="Current" value={after} />
      </Stack>
    </Stack>
  );
}

function DiffPanel({ label, value }: { label: string; value: string }): React.JSX.Element {
  return (
    <Box flex={1}>
      <Text fontSize="sm" color="text.muted" mb={2}>
        {label}
      </Text>
      <Code whiteSpace="pre-wrap" fontSize="sm" p={3} display="block">
        {value}
      </Code>
    </Box>
  );
}
