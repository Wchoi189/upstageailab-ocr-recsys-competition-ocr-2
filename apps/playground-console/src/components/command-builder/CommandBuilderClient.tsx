"use client";

import {
  Alert,
  Box,
  Button,
  Flex,
  Grid,
  GridItem,
  Heading,
  Spinner,
  Stack,
  Tabs,
  Text,
} from "@chakra-ui/react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import type React from "react";

import { buildCommand, getCommandRecommendations, getSchemaDetails } from "@/api/commands";
import { CommandDiffViewer } from "@/components/command-builder/CommandDiffViewer";
import { CommandDisplay } from "@/components/command-builder/CommandDisplay";
import { RecommendationsGrid } from "@/components/command-builder/RecommendationsGrid";
import { SchemaForm } from "@/components/command-builder/SchemaForm";
import type { BuildCommandResponse, FormValues, Recommendation, SchemaId } from "@/types/schema";

const tabOptions: Array<{ id: SchemaId; label: string }> = [
  { id: "train", label: "Training" },
  { id: "test", label: "Testing" },
  { id: "predict", label: "Prediction" },
];

export function CommandBuilderClient(): React.JSX.Element {
  const [activeTab, setActiveTab] = useState<SchemaId>("train");
  const [values, setValues] = useState<FormValues>({});
  const [commandResult, setCommandResult] = useState<BuildCommandResponse | null>(null);
  const [previousCommand, setPreviousCommand] = useState<string>("");
  const [selectedRecommendationId, setSelectedRecommendationId] = useState<string>();
  const [showRecommendations, setShowRecommendations] = useState(true);

  const schemaQuery = useQuery({
    queryKey: ["command-schema", activeTab],
    queryFn: () => getSchemaDetails(activeTab),
    staleTime: 5 * 60 * 1000,
  });

  const architectureValue = values.architecture as string | undefined;

  const recommendationsQuery = useQuery({
    queryKey: ["command-recommendations", architectureValue],
    queryFn: () => getCommandRecommendations(architectureValue),
    staleTime: 5 * 60 * 1000,
  });

  const buildCommandMutation = useMutation({
    mutationFn: buildCommand,
    onSuccess: (data) => {
      if (commandResult?.command) {
        setPreviousCommand(commandResult.command);
      }
      setCommandResult(data);
    },
  });

  useEffect(() => {
    if (!schemaQuery.data || Object.keys(values).length === 0) {
      return;
    }
    const timeout = setTimeout(() => {
      buildCommandMutation.mutate({
        schema_id: activeTab,
        values,
        append_model_suffix: true,
      });
    }, 250);
    return () => clearTimeout(timeout);
  }, [values, activeTab, schemaQuery.data, buildCommandMutation]);

  const isLoading = schemaQuery.isLoading;
  const hasError = schemaQuery.error instanceof Error;
  const schema = schemaQuery.data;

  const handleRecommendationSelect = (rec: Recommendation): void => {
    setSelectedRecommendationId(rec.id);
    setValues((prev) => ({ ...prev, ...rec.parameters }));
  };

  const commandSections = useMemo(() => {
    if (!schema) {
      return [];
    }
    return schema.constant_overrides;
  }, [schema]);

  const copyCommand = (): void => {
    if (commandResult?.command) {
      void navigator.clipboard.writeText(commandResult.command);
    }
  };

  const handleTabChange = (value: string): void => {
    setValues({});
    setCommandResult(null);
    setPreviousCommand("");
    setSelectedRecommendationId(undefined);
    setActiveTab(value as SchemaId);
  };

  return (
    <Stack gap={8}>
      <Box>
        <Text textTransform="uppercase" fontSize="sm" color="text.muted">
          Build /
        </Text>
        <Heading size="lg" mt={2}>
          Command Builder
        </Heading>
        <Text color="text.muted" maxW="3xl" mt={2}>
          Configure experiment metadata, model overrides, and dataset settings using live schemas pulled from the Admin API.
        </Text>
      </Box>

      <Tabs.Root
        colorScheme="brand"
        value={activeTab}
        onValueChange={(details) => handleTabChange(details.value)}
      >
        <Tabs.List>
          {tabOptions.map((tab) => (
            <Tabs.Trigger key={tab.id} value={tab.id}>
              {tab.label}
            </Tabs.Trigger>
          ))}
        </Tabs.List>
        {tabOptions.map((tab) => (
          <Tabs.Content key={tab.id} value={tab.id} px={0}>
            {isLoading && (
              <Flex align="center" justify="center" py={20}>
                <Spinner size="lg" />
              </Flex>
            )}
            {hasError && (
              <Alert.Root status="error" borderRadius="md">
                <Alert.Indicator />
                <Alert.Description>
                  Unable to load schema. Ensure the FastAPI server is running on the configured backend URL.
                </Alert.Description>
              </Alert.Root>
            )}
            {schema && (
              <Grid templateColumns={{ base: "1fr", xl: "2fr 1fr" }} gap={8} alignItems="start">
                <GridItem>
                  <SchemaForm schema={schema} values={values} onChange={setValues} />
                </GridItem>
                <GridItem>
                  <Stack gap={6}>
                    {commandResult ? (
                      <CommandDisplay
                        command={commandResult.command}
                        overrides={commandResult.overrides}
                        constantOverrides={commandSections}
                        validationError={commandResult.validation_error}
                        onCopy={copyCommand}
                      />
                    ) : (
                      <PlaceholderCard />
                    )}

                    {previousCommand && commandResult?.command && previousCommand !== commandResult.command && (
                      <CommandDiffViewer before={previousCommand} after={commandResult.command} />
                    )}

                    <Stack gap={3}>
                      <Flex justify="space-between" align="center">
                        <Text fontWeight="semibold">Recommended configs</Text>
                        <Button variant="ghost" size="sm" onClick={() => setShowRecommendations((val) => !val)}>
                          {showRecommendations ? "Hide" : "Show"}
                        </Button>
                      </Flex>
                      {showRecommendations &&
                        (recommendationsQuery.isLoading ? (
                          <Spinner size="sm" />
                        ) : (
                          <RecommendationsGrid
                            recommendations={recommendationsQuery.data ?? []}
                            onSelect={handleRecommendationSelect}
                            selectedId={selectedRecommendationId}
                          />
                        ))}
                    </Stack>
                  </Stack>
                </GridItem>
              </Grid>
            )}
          </Tabs.Content>
        ))}
      </Tabs.Root>
    </Stack>
  );
}

function PlaceholderCard(): React.JSX.Element {
  return (
    <Stack gap={4} border="1px dashed" borderColor="border.subtle" borderRadius="lg" p={5} bg="surface.panel">
      <Text fontWeight="semibold">Command output</Text>
      <Text fontSize="sm" color="text.muted">
        Fill out the form to generate a command preview with overrides and validation details.
      </Text>
    </Stack>
  );
}
