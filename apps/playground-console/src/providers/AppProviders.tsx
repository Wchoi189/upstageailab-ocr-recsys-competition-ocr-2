"use client";

import { ChakraProvider } from "@chakra-ui/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";
import { type PropsWithChildren, useState } from "react";

import theme from "@/theme";
import { SessionProvider } from "@/contexts/SessionContext";

export function AppProviders({ children }: PropsWithChildren): React.JSX.Element {
  const [queryClient] = useState(() => new QueryClient());

  return (
    <ChakraProvider value={theme}>
      <SessionProvider>
        <QueryClientProvider client={queryClient}>
          {children}
          {process.env.NODE_ENV === "development" && (
            <ReactQueryDevtools initialIsOpen={false} position="bottom" buttonPosition="bottom-right" />
          )}
        </QueryClientProvider>
      </SessionProvider>
    </ChakraProvider>
  );
}
