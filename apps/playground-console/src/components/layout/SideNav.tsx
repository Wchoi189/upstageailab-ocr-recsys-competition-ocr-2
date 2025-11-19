"use client";

import {
  Box,
  Flex,
  Text,
  VStack,
  Button,
} from "@chakra-ui/react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import type React from "react";

const sections = [
  {
    label: "Overview",
    items: [{ label: "Console Home", href: "/" }],
  },
  {
    label: "Build",
    items: [{ label: "Command Builder", href: "/playground/command-builder" }],
  },
  {
    label: "Extract",
    items: [
      { label: "Universal Extraction", href: "/extract/universal" },
      { label: "Prebuilt Extraction", href: "/extract/prebuilt" },
    ],
  },
];

export function SideNav(): React.JSX.Element {
  const pathname = usePathname();

  return (
    <Box
      as="nav"
      width={{ base: "full", md: "260px" }}
      borderRight="1px solid"
      borderColor="border.subtle"
      bg="surface.panel"
      minH="100vh"
      px={4}
      py={6}
      display={{ base: "none", md: "block" }}
    >
      {sections.map((section) => (
        <Box key={section.label} mb={8}>
          <Text
            textTransform="uppercase"
            fontSize="xs"
            letterSpacing="0.08em"
            color="text.muted"
            mb={3}
          >
            {section.label}
          </Text>
          <VStack spacing={2} align="stretch">
            {section.items.map((item) => {
              const isActive = pathname === item.href;
              return (
                <Button
                  key={item.label}
                  as={Link}
                  href={item.href}
                  justifyContent="flex-start"
                  variant="ghost"
                  bg={isActive ? "brand.50" : undefined}
                  fontWeight={isActive ? "semibold" : "medium"}
                  _hover={{ bg: "gray.100" }}
                >
                  {item.label}
                </Button>
              );
            })}
          </VStack>
        </Box>
      ))}
      <Flex
        direction="column"
        gap={2}
        p={4}
        borderRadius="lg"
        bg="brand.50"
        border="1px solid"
        borderColor="brand.100"
      >
        <Text fontWeight="semibold" fontSize="sm" color="brand.600">
          New launch
        </Text>
        <Text fontSize="xs" color="text.muted">
          Solar Pro 2 is live. Explore enhanced document parsing accuracy.
        </Text>
      </Flex>
    </Box>
  );
}
