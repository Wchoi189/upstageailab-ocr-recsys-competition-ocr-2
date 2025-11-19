import type { Metadata } from "next";
import { Inter } from "next/font/google";
import type { ReactNode } from "react";

import { AppProviders } from "@/providers/AppProviders";
import { GTMProvider } from "@/components/analytics/GTMProvider";
import { ConsentBanner } from "@/components/analytics/ConsentBanner";

import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "OCR Playground Console",
  description: "Next.js console for OCR command building and extraction",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <AppProviders>
          <GTMProvider>
            <main>{children}</main>
            <ConsentBanner />
          </GTMProvider>
        </AppProviders>
      </body>
    </html>
  );
}
