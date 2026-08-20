import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Clinical RAG Eval Harness",
  description: "Retrieval and extraction evaluation for clinical trial protocols",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}