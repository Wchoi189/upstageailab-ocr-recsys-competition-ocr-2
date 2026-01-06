import { Box, Button, Heading, SimpleGrid, Stack, Text } from "@chakra-ui/react";
import type React from "react";

import { AppShell } from "@/components/layout/AppShell";

const cards = [
  {
    title: "Command Builder",
    description:
      "Generate train/test/predict commands with schema-driven forms and instant diffs.",
    actionLabel: "Open Builder",
  },
  {
    title: "Universal Extraction",
    description:
      "Parse arbitrary documents with adaptive OCR + layout models tuned for digitization.",
    actionLabel: "Configure Extraction",
  },
  {
    title: "Prebuilt Extraction",
    description:
      "Leverage curated templates (e.g., business cards) with preview + JSON panes.",
    actionLabel: "Launch Templates",
  },
];

export default function Home(): React.JSX.Element {
  return (
    <AppShell>
      <Stack gap={10}>
        <Stack gap={4}>
          <Text textTransform="uppercase" fontSize="sm" color="text.muted">
            Digitize / Command Builder
          </Text>
          <Heading size="lg">Build document pipelines faster</Heading>
          <Text maxW="3xl" color="text.muted">
            Switch between Command Builder, Universal Extraction, and Prebuilt Extraction tools
            within a unified Upstage-branded console that mirrors the production playground.
          </Text>
          <Stack direction={{ base: "column", sm: "row" }} gap={3}>
            <Button colorScheme="brand">Start Command Builder</Button>
            <Button variant="outline">View API Reference</Button>
          </Stack>
        </Stack>

        <SimpleGrid columns={{ base: 1, md: 3 }} gap={6}>
          {cards.map((card) => (
            <Box
              key={card.title}
              border="1px solid"
              borderColor="border.subtle"
              borderRadius="xl"
              p={6}
              bg="surface.panel"
              shadow="sm"
            >
              <Text fontSize="sm" color="text.muted" mb={2}>
                Tool
              </Text>
              <Heading size="md" mb={3}>
                {card.title}
              </Heading>
              <Text color="text.muted" mb={6}>
                {card.description}
              </Text>
              <Button variant="ghost" colorScheme="brand" size="sm">
                {card.actionLabel}
              </Button>
            </Box>
          ))}
        </SimpleGrid>
      </Stack>
    </AppShell>
  );
}
