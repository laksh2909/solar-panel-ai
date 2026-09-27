import React from "react";
import { Sidebar } from "./Sidebar";
import { Header } from "./Header";

interface AppShellProps {
  children: React.ReactNode;
  breadcrumbs?: { label: string; href?: string; active?: boolean }[];
}

export function AppShell({ children, breadcrumbs }: AppShellProps) {
  return (
    <div className="min-h-screen bg-surface flex flex-col md:flex-row">
      <Sidebar />
      <div className="flex min-h-screen w-full flex-col md:pl-64">
        <Header breadcrumbs={breadcrumbs} />
        <main className="relative w-full flex-1 bg-surface px-4 py-4 pt-16 max-w-7xl md:px-6 md:py-6">
          {children}
        </main>
      </div>
    </div>
  );
}
