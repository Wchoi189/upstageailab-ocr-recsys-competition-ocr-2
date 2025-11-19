"use client";

import { Alert, CloseButton, Flex } from "@chakra-ui/react";
import { useState } from "react";
import type React from "react";

export function ShellBanner(): React.JSX.Element | null {
  const [visible, setVisible] = useState(true);

  if (!visible) {
    return null;
  }

  return (
    <Alert.Root
      status="info"
      borderRadius="lg"
      mb={6}
      bg="brand.50"
      color="brand.700"
    >
      <Alert.Indicator color="brand.500" />
      <Flex direction="column" gap={1} flex={1} pr={4}>
        <Alert.Title>Solar Pro 2 is now live</Alert.Title>
        <Alert.Description>
          Try the upgraded document parsing pipeline with improved table understanding directly in this console.
        </Alert.Description>
      </Flex>
      <CloseButton
        position="absolute"
        top={2}
        right={2}
        onClick={() => setVisible(false)}
        color="brand.700"
        size="sm"
      />
    </Alert.Root>
  );
}
