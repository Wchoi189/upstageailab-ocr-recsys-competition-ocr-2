"use client";

import { Box, Button, Heading, SimpleGrid, Stack, Text } from "@chakra-ui/react";
import { useMemo, useState } from "react";
import type React from "react";

import { CardGallery, PreviewPanel } from "@/components/extract/PreviewPanel";

const cards = [
  {
    id: "card-1",
    title: "Business Card",
    subtitle: "Standard",
    thumbnail: "https://images.unsplash.com/photo-1507679799987-c73779587ccf?auto=format&fit=crop&w=600&q=80",
  },
  {
    id: "card-2",
    title: "Business Card",
    subtitle: "Vertical",
    thumbnail: "https://images.unsplash.com/photo-1529333166437-7750a6dd5a70?auto=format&fit=crop&w=600&q=80",
  },
  {
    id: "card-3",
    title: "Receipt",
    subtitle: "Thermal",
    thumbnail: "https://images.unsplash.com/photo-1489515217757-5fd1be406fef?auto=format&fit=crop&w=600&q=80",
  },
];

const mockJson = {
  name: "Dana Seo",
  company: "Upstage",
  role: "Product Lead",
  email: "dana@upstage.ai",
  phone: "+82-10-1234-5678",
};

export function PrebuiltExtractionClient(): React.JSX.Element {
  const [selectedCard, setSelectedCard] = useState(cards[0].id);

  const previewImage = useMemo(() => {
    return cards.find((card) => card.id === selectedCard)?.thumbnail ?? cards[0].thumbnail;
  }, [selectedCard]);

  return (
    <Stack spacing={8}>
      <Stack spacing={3}>
        <Text textTransform="uppercase" fontSize="sm" color="text.muted">
          Extract /
        </Text>
        <Heading size="lg">Prebuilt Extraction</Heading>
        <Text color="text.muted" maxW="3xl">
          Use curated extraction templates for business cards, receipts, and invoices. Thumbnails keep designers aligned,
          while Preview/JSON panes show the structured output immediately.
        </Text>
        <Button colorScheme="brand" alignSelf="flex-start">
          Deploy template
        </Button>
      </Stack>

      <Stack spacing={6}>
        <CardGallery items={cards} selectedId={selectedCard} onSelect={setSelectedCard} />
        <SimpleGrid columns={{ base: 1, lg: 2 }} gap={8} alignItems="start">
          <Box>
            <Heading size="sm" mb={4}>
              Template details
            </Heading>
            <Text fontSize="sm" color="text.muted">
              Selected template: {selectedCard}
            </Text>
          </Box>
          <PreviewPanel imageSrc={previewImage} json={mockJson} />
        </SimpleGrid>
      </Stack>
    </Stack>
  );
}
