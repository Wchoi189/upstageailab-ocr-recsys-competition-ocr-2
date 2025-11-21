"use client";

import { useState } from "react";

const CONSENT_STORAGE_KEY = "cookie-consent";

export type ConsentState = "pending" | "accepted" | "declined";

export interface ConsentHook {
  consent: ConsentState;
  showBanner: boolean;
  acceptConsent: () => void;
  declineConsent: () => void;
}

/**
 * useConsent - Hook to manage cookie/analytics consent
 *
 * Stores consent state in localStorage and provides methods to accept/decline
 *
 * @returns ConsentHook with consent state and methods
 */
export function useConsent(): ConsentHook {
  // Use lazy initializer to read from localStorage on mount
  const [consent, setConsent] = useState<ConsentState>(() => {
    try {
      const stored = localStorage.getItem(CONSENT_STORAGE_KEY);
      if (stored === "accepted" || stored === "declined") {
        return stored as ConsentState;
      }
    } catch (error) {
      console.error("Failed to read consent from localStorage:", error);
    }
    return "pending";
  });

  const [showBanner, setShowBanner] = useState(() => {
    try {
      const stored = localStorage.getItem(CONSENT_STORAGE_KEY);
      return stored !== "accepted" && stored !== "declined";
    } catch (error) {
      console.error("Failed to read consent from localStorage:", error);
      return true;
    }
  });

  const acceptConsent = () => {
    try {
      localStorage.setItem(CONSENT_STORAGE_KEY, "accepted");
      setConsent("accepted");
      setShowBanner(false);
    } catch (error) {
      console.error("Failed to save consent to localStorage:", error);
    }
  };

  const declineConsent = () => {
    try {
      localStorage.setItem(CONSENT_STORAGE_KEY, "declined");
      setConsent("declined");
      setShowBanner(false);
    } catch (error) {
      console.error("Failed to save consent to localStorage:", error);
    }
  };

  return {
    consent,
    showBanner,
    acceptConsent,
    declineConsent,
  };
}

