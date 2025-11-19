"use client";

import { Alert, AlertDescription, AlertIcon, AlertTitle, CloseButton, Flex } from "@chakra-ui/react";
import { useState } from "react";
import type React from "react";

export function ShellBanner(): React.JSX.Element | null {
  const [visible, setVisible] = useState(true);

  if (!visible) {
    return null;
  }

  return (
    <Alert
      status="info"
      borderRadius="lg"
      mb={6}
      bg="brand.50"
      color="brand.700"
      alignItems="flex-start"
    >
      <AlertIcon color="brand.500" />
      <Flex direction="column" gap={1} flex={1} pr={4}>
        <AlertTitle>Solar Pro 2 is now live</AlertTitle>
        <AlertDescription>
          Try the upgraded document parsing pipeline with improved table understanding directly in this console.
        </AlertDescription>
      </Flex>
      <CloseButton
        alignSelf="flex-start"
        onClick={() => setVisible(false)}
        color="brand.700"
      />
    </Alert>
  );
}
