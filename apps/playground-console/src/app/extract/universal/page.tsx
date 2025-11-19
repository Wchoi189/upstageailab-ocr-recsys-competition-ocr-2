import { Box, Button, Heading, HStack, Input, SimpleGrid, Stack, Text, Textarea } from "@chakra-ui/react";
import type { Metadata } from "next";
import type React from "react";

import { AppShell } from "@/components/layout/AppShell";
import { PreviewPanel } from "@/components/extract/PreviewPanel";

const sampleJson = {
  document_type: "invoice",
  vendor: "Upstage Labs",
  total: 1299.5,
  currency: "USD",
  items: [
    { description: "OCR API Credits", qty: 1, unit_price: 999 },
    { description: "Support plan", qty: 1, unit_price: 300.5 },
  ],
};

export const metadata: Metadata = {
  title: "Universal Extraction | Upstage Console",
};

export default function UniversalExtractionPage(): React.JSX.Element {
  return (
    <AppShell>
      <Stack spacing={8}>
        <Stack spacing={3}>
          <Text textTransform="uppercase" fontSize="sm" color="text.muted">
            Extract /
          </Text>
          <Heading size="lg">Universal Extraction</Heading>
          <Text color="text.muted" maxW="3xl">
            Upload any document and configure the OCR + layout pipeline. Use this space to iterate on pre-processing
            params before pushing to production.
          </Text>
          <HStack spacing={3}>
            <Button colorScheme="brand">Run extraction</Button>
            <Button variant="outline">Download JSON</Button>
          </HStack>
        </Stack>

        <SimpleGrid columns={{ base: 1, lg: 2 }} gap={8} alignItems="start">
          <Stack spacing={5}>
            <Box border="1px solid" borderColor="border.subtle" borderRadius="lg" p={5} bg="surface.panel">
              <Heading size="sm" mb={4}>
                Document sources
              </Heading>
              <Stack spacing={3}>
                <Input type="file" accept="image/*,.pdf" />
                <Input placeholder="or paste a URL" />
                <Textarea placeholder="Add notes for this run" rows={3} />
              </Stack>
            </Box>
            <Box border="1px solid" borderColor="border.subtle" borderRadius="lg" p={5} bg="surface.panel">
              <Heading size="sm" mb={4}>
                Pipeline settings
              </Heading>
              <SimpleGrid columns={{ base: 1, md: 2 }} gap={3}>
                <Input placeholder="Model version" defaultValue="solar-pro-2" />
                <Input placeholder="Language" defaultValue="Multi" />
                <Input placeholder="Denoise level" defaultValue="medium" />
                <Input placeholder="Table extraction" defaultValue="enabled" />
              </SimpleGrid>
            </Box>
          </Stack>
          <PreviewPanel
            imageSrc="https://images.unsplash.com/photo-1529429617124-aee711a70412?auto=format&fit=crop&w=800&q=80"
            json={sampleJson}
          />
        </SimpleGrid>
      </Stack>
    </AppShell>
  );
}
