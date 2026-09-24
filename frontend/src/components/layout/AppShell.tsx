import React from "react";
import { Sidebar } from "./Sidebar";
import { Header } from "./Header";

interface AppShellProps {
  children: React.ReactNode;
  breadcrumbs?: { label: string; href?: string; active?: boolean }[];
}

export function AppShell({ children, breadcrumbs }: AppShellProps) {
  return (
    <div className="min-h-screen bg-surface flex">
      <Sidebar />
      <div className="pl-64 flex flex-col min-h-screen w-full">
        <Header breadcrumbs={breadcrumbs} />
        <main className="relative pt-16 w-full flex-1 bg-surface px-6 py-6 max-w-7xl">
          {children}
        </main>
      </div>
    </div>
  );
}
