"use client";

import { useSessionContext, type Session } from "@/contexts/SessionContext";

/**
 * useSession - Hook to access current session state
 *
 * @returns Session object with user info and authentication status
 *
 * @example
 * ```tsx
 * const { isAuthenticated, user } = useSession();
 *
 * if (isAuthenticated) {
 *   return <div>Welcome, {user?.name}</div>;
 * }
 * ```
 */
export function useSession(): Session {
  return useSessionContext();
}
