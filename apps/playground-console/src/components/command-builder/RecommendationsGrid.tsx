"use client";

import { Badge, Box, Button, Stack, Text } from "@chakra-ui/react";
import type React from "react";

import type { Recommendation } from "@/types/schema";

interface RecommendationsGridProps {
  recommendations: Recommendation[];
  onSelect: (recommendation: Recommendation) => void;
  selectedId?: string;
}

export function RecommendationsGrid({
  recommendations,
  onSelect,
  selectedId,
}: RecommendationsGridProps): React.JSX.Element {
  if (!recommendations.length) {
    return <Text fontSize="sm">No recommendations available.</Text>;
  }

  return (
    <Stack spacing={4}>
      {recommendations.map((rec) => {
        const isSelected = rec.id === selectedId;
        return (
          <Box
            key={rec.id}
            border="1px solid"
            borderColor={isSelected ? "brand.200" : "border.subtle"}
            borderRadius="lg"
            p={4}
            bg="surface.panel"
          >
            <Stack direction="row" justify="space-between" align="center">
              <Stack spacing={0}>
                <Text fontWeight="semibold">{rec.title}</Text>
                <Text fontSize="sm" color="text.muted">
                  {rec.description}
                </Text>
              </Stack>
              {rec.architecture && (
                <Badge colorScheme="brand">{rec.architecture}</Badge>
              )}
            </Stack>
            <Button size="sm" mt={3} onClick={() => onSelect(rec)}>
              {isSelected ? "Selected" : "Apply"}
            </Button>
          </Box>
        );
      })}
    </Stack>
  );
}
