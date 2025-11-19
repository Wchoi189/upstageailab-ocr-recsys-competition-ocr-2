"use client";

import { Box, Flex } from "@chakra-ui/react";
import type React from "react";

import { ShellBanner } from "@/components/layout/ShellBanner";
import { SideNav } from "@/components/layout/SideNav";
import { TopNav } from "@/components/layout/TopNav";

interface AppShellProps {
  children: React.ReactNode;
}

export function AppShell({ children }: AppShellProps): React.JSX.Element {
  return (
    <Flex minH="100vh" bg="surface.background">
      <SideNav />
      <Flex flex="1" direction="column">
        <TopNav />
        <Box as="section" px={{ base: 4, md: 10 }} py={8} flex="1">
          <ShellBanner />
          {children}
        </Box>
      </Flex>
    </Flex>
  );
}
