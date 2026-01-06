"use client";

import { Button, Flex, HStack, Link as ChakraLink, Text } from "@chakra-ui/react";
import type React from "react";
import Link from "next/link";

const navLinks = [
  { label: "Documentation", href: "https://docs.upstage.ai" },
  { label: "Playground", href: "/" },
  { label: "Dashboard", href: "https://console.upstage.ai/api-keys" },
];

export function TopNav(): React.JSX.Element {
  return (
    <Flex
      as="header"
      height="64px"
      align="center"
      justify="space-between"
      px={8}
      borderBottom="1px solid"
      borderColor="border.subtle"
      bg="surface.panel"
      position="sticky"
      top={0}
      zIndex={10}
    >
      <Text fontWeight="semibold" fontSize="lg" color="brand.500">
        Upstage Console
      </Text>
      <HStack gap={6} display={{ base: "none", md: "flex" }}>
        {navLinks.map((link) => (
          <ChakraLink
            key={link.label}
            as={Link}
            href={link.href}
            color="text.muted"
            fontWeight="medium"
            _hover={{ color: "brand.500" }}
          >
            {link.label}
          </ChakraLink>
        ))}
      </HStack>
      <Button variant="solid" colorScheme="brand" size="sm">
        Log in
      </Button>
    </Flex>
  );
}
