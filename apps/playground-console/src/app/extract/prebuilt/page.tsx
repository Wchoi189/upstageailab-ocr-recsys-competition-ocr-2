import type { Metadata } from "next";
import type React from "react";

import { AppShell } from "@/components/layout/AppShell";
import { PrebuiltExtractionClient } from "@/components/extract/PrebuiltExtractionClient";

export const metadata: Metadata = {
  title: "Prebuilt Extraction | Upstage Console",
};

export default function PrebuiltExtractionPage(): React.JSX.Element {
  return (
    <AppShell>
      <PrebuiltExtractionClient />
    </AppShell>
  );
}
