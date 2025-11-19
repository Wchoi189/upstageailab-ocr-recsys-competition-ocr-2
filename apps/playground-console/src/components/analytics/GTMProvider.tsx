"use client";

import { GoogleTagManager } from "@next/third-parties/google";
import { useConsent } from "@/hooks/useConsent";
import { type ReactNode } from "react";

export interface GTMProviderProps {
  children: ReactNode;
}

/**
 * GTMProvider - Google Tag Manager integration component
 *
 * Only loads GTM if:
 * 1. NEXT_PUBLIC_GTM_ID is provided
 * 2. User has accepted cookie consent
 *
 * @param children - Child components to render
 */
export function GTMProvider({ children }: GTMProviderProps) {
  const gtmId = process.env.NEXT_PUBLIC_GTM_ID;
  const { consent } = useConsent();

  // Don't load GTM if no ID is provided
  if (!gtmId) {
    return <>{children}</>;
  }

  // Only load GTM if consent is accepted
  const shouldLoadGTM = consent === "accepted";

  return (
    <>
      {shouldLoadGTM && <GoogleTagManager gtmId={gtmId} />}
      {children}
    </>
  );
}
