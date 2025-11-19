import type { Metadata } from "next";

import { AppShell } from "@/components/layout/AppShell";
import { CommandBuilderClient } from "@/components/command-builder/CommandBuilderClient";

export const metadata: Metadata = {
  title: "Command Builder | Upstage Console",
};

export default function CommandBuilderPage(): React.JSX.Element {
  return (
    <AppShell>
      <CommandBuilderClient />
    </AppShell>
  );
}
