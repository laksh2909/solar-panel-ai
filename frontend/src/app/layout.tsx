import type { Metadata } from "next";
import "./globals.css";
import { ToastProvider } from "@/components/common/Toast";

export const metadata: Metadata = {
  title: "SolarScan AI — Solar Panel AI Inspection System",
  description:
    "Enterprise AI-assisted visual assessment, optical RGB image classification, and maintenance recommendation system for photovoltaic solar assets.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="bg-surface font-body-md text-on-surface antialiased min-h-screen">
        <ToastProvider>{children}</ToastProvider>
      </body>
    </html>
  );
}
