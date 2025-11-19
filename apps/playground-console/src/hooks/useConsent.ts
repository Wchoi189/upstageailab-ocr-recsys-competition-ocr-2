"use client";

import { useState, useEffect } from "react";

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
  const [consent, setConsent] = useState<ConsentState>("pending");
  const [showBanner, setShowBanner] = useState(false);

  useEffect(() => {
    // Check for existing consent in localStorage
    try {
      const stored = localStorage.getItem(CONSENT_STORAGE_KEY);
      if (stored === "accepted" || stored === "declined") {
        setConsent(stored as ConsentState);
        setShowBanner(false);
      } else {
        setShowBanner(true);
      }
    } catch (error) {
      console.error("Failed to read consent from localStorage:", error);
      setShowBanner(true);
    }
  }, []);

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
