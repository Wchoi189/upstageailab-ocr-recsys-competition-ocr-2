"use client";

import { createContext, useContext, ReactNode } from "react";

export interface Session {
  user: { id?: string; email?: string; name?: string } | null;
  isAuthenticated: boolean;
}

const defaultSession: Session = {
  user: null,
  isAuthenticated: false,
};

const SessionContext = createContext<Session>(defaultSession);

export interface SessionProviderProps {
  children: ReactNode;
}

/**
 * SessionProvider - Provides session context to the application
 *
 * Currently returns a placeholder unauthenticated session.
 * This structure is ready for future auth integration (NextAuth, custom JWT, etc.)
 */
export function SessionProvider({ children }: SessionProviderProps) {
  // For now, always return unauthenticated session
  // TODO: Integrate with auth provider (NextAuth, custom JWT, etc.)
  const session: Session = {
    user: null,
    isAuthenticated: false,
  };

  return (
    <SessionContext.Provider value={session}>
      {children}
    </SessionContext.Provider>
  );
}

/**
 * useSessionContext - Internal hook to access session context
 * Use the exported useSession hook instead
 */
export function useSessionContext(): Session {
  const context = useContext(SessionContext);
  if (context === undefined) {
    throw new Error("useSessionContext must be used within SessionProvider");
  }
  return context;
}
