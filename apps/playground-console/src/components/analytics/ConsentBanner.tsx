"use client";

import { Box, Button, HStack, Text, Alert } from "@chakra-ui/react";
import { useConsent } from "@/hooks/useConsent";

/**
 * ConsentBanner - Cookie consent banner component
 *
 * Displays a banner at the bottom of the page asking for cookie consent.
 * Only shown if no consent has been given yet.
 */
export function ConsentBanner() {
  const { showBanner, acceptConsent, declineConsent } = useConsent();

  if (!showBanner) {
    return null;
  }

  return (
    <Box
      position="fixed"
      bottom={0}
      left={0}
      right={0}
      zIndex={9999}
      bg="gray.900"
      borderTop="1px solid"
      borderColor="gray.700"
      p={4}
    >
      <Alert.Root
        status="info"
        variant="subtle"
        flexDirection={{ base: "column", md: "row" }}
        alignItems="center"
        justifyContent="space-between"
      >
        <Box flex="1" mb={{ base: 3, md: 0 }}>
          <Text fontSize="sm" color="gray.200">
            We use cookies and analytics to improve your experience.
            By clicking &quot;Accept&quot;, you consent to our use of cookies and analytics tools.
          </Text>
        </Box>
        <HStack gap={2} flexShrink={0}>
          <Button
            size="sm"
            colorScheme="blue"
            onClick={acceptConsent}
          >
            Accept
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={declineConsent}
          >
            Decline
          </Button>
        </HStack>
      </Alert.Root>
    </Box>
  );
}
