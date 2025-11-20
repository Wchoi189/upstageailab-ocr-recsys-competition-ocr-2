"use client";

import { Box, Heading, Stack, Tabs, Text } from "@chakra-ui/react";
import type React from "react";

interface PreviewPanelProps {
  imageSrc: string;
  json: Record<string, unknown>;
}

export function PreviewPanel({ imageSrc, json }: PreviewPanelProps): React.JSX.Element {
  return (
    <Tabs.Root variant="enclosed" colorScheme="brand" defaultValue="preview">
      <Tabs.List>
        <Tabs.Trigger value="preview">Preview</Tabs.Trigger>
        <Tabs.Trigger value="json">JSON</Tabs.Trigger>
      </Tabs.List>
      <Tabs.Content value="preview" px={0} pt={4}>
        <Box
          border="1px solid"
          borderColor="border.subtle"
          borderRadius="lg"
          overflow="hidden"
          bg="black"
          minH="360px"
          backgroundImage={`url(${imageSrc})`}
          backgroundSize="contain"
          backgroundRepeat="no-repeat"
          backgroundPosition="center"
        />
      </Tabs.Content>
      <Tabs.Content value="json">
        <Box
          as="pre"
          fontSize="sm"
          bg="gray.900"
          color="green.200"
          borderRadius="md"
          p={4}
          overflowX="auto"
          maxH="320px"
        >
          {JSON.stringify(json, null, 2)}
        </Box>
      </Tabs.Content>
    </Tabs.Root>
  );
}

interface CardGalleryProps {
  items: Array<{ id: string; title: string; subtitle: string; thumbnail: string }>;
  selectedId: string;
  onSelect: (id: string) => void;
}

export function CardGallery({ items, selectedId, onSelect }: CardGalleryProps): React.JSX.Element {
  return (
    <Stack direction={{ base: "column", md: "row" }} spacing={4} overflowX={{ base: "visible", md: "auto" }}>
      {items.map((item) => (
        <Box
          key={item.id}
          border="2px solid"
          borderColor={item.id === selectedId ? "brand.300" : "border.subtle"}
          borderRadius="lg"
          p={3}
          cursor="pointer"
          onClick={() => onSelect(item.id)}
          bg="surface.panel"
          minW={{ base: "100%", md: "220px" }}
        >
          <Box
            borderRadius="lg"
            height="140px"
            backgroundImage={`url(${item.thumbnail})`}
            backgroundSize="cover"
            backgroundPosition="center"
          />
          <Heading size="sm" mt={3}>
            {item.title}
          </Heading>
          <Text fontSize="sm" color="text.muted">
            {item.subtitle}
          </Text>
        </Box>
      ))}
    </Stack>
  );
}
