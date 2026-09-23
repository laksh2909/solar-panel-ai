"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Logo } from "./Logo";

interface NavItem {
  name: string;
  href: string;
  icon: string;
  pattern: RegExp;
}

const NAV_ITEMS: NavItem[] = [
  {
    name: "Dashboard",
    href: "/",
    icon: "dashboard",
    pattern: /^(\/|\/dashboard)$/,
  },
  {
    name: "New Inspection",
    href: "/new-inspection",
    icon: "add_a_photo",
    pattern: /^\/new-inspection/,
  },
  {
    name: "Inspection History",
    href: "/inspection-history",
    icon: "history",
    pattern: /^\/inspection-history/,
  },
  {
    name: "Inspection Result",
    href: "/inspection-result",
    icon: "analytics",
    pattern: /^\/inspection-result/,
  },
  {
    name: "Panel Details",
    href: "/panel-details",
    icon: "grid_view",
    pattern: /^\/panel-details/,
  },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed left-0 top-0 h-full w-72 bg-surface-container-lowest border-r border-outline-variant/30 z-50 flex flex-col justify-between">
      <div className="flex flex-col">
        {/* Brand Header */}
        <div className="h-16 px-space-md flex items-center justify-between border-b border-outline-variant/20">
          <Link href="/" className="flex items-center gap-space-sm">
            <Logo />
          </Link>
          <span className="px-space-xs py-0.5 bg-secondary-container text-on-secondary-container border border-secondary/20 rounded font-label-caps text-label-caps">
            ACTIVE
          </span>
        </div>

        {/* Facility Context Selector */}
        <div className="p-space-md border-b border-outline-variant/20">
          <label className="font-label-caps text-label-caps text-on-surface-variant block uppercase mb-space-xs">
            Facility Context
          </label>
          <div className="flex items-center justify-between px-space-sm py-1.5 bg-surface-container-low rounded border border-outline-variant/40 hover:border-outline-variant transition-colors cursor-pointer">
            <div className="flex items-center gap-1.5 truncate">
              <span className="material-symbols-outlined text-primary text-base">
                solar_power
              </span>
              <span className="font-code-id text-code-id text-on-surface truncate">
                Solar Farm Alpha - Sec 4
              </span>
            </div>
            <span className="material-symbols-outlined text-outline text-sm">
              unfold_more
            </span>
          </div>
        </div>

        {/* Navigation items */}
        <nav className="p-space-md flex flex-col gap-1">
          {NAV_ITEMS.map((item) => {
            const isActive = item.pattern.test(pathname);
            return (
              <Link
                key={item.name}
                href={item.href}
                className={`flex items-center gap-space-sm px-space-sm py-2 rounded transition-colors ${
                  isActive
                    ? "bg-primary text-on-primary font-semibold shadow-sm"
                    : "text-on-surface-variant hover:bg-surface-container hover:text-on-surface font-body-md text-body-md"
                }`}
              >
                <span className="material-symbols-outlined text-lg">
                  {item.icon}
                </span>
                <span>{item.name}</span>
              </Link>
            );
          })}
        </nav>
      </div>

      {/* AI Inference Engine status footer */}
      <div className="p-space-md border-t border-outline-variant/20 bg-surface-container-low">
        <div className="flex items-center justify-between mb-1.5">
          <span className="font-label-caps text-label-caps text-on-surface-variant uppercase">
            AI Inference Engine
          </span>
          <span className="inline-block w-2 h-2 rounded-full bg-secondary"></span>
        </div>
        <div className="p-space-xs bg-surface-container-lowest rounded border border-outline-variant/30">
          <div className="font-label-caps text-label-caps text-primary">
            MODEL: EfficientNet-B0 (RGB)
          </div>
          <div className="font-code-id text-code-id text-on-surface-variant mt-0.5">
            GET /api/health{" "}
            <span className="text-secondary font-semibold">READY</span>
          </div>
        </div>
      </div>
    </aside>
  );
}
