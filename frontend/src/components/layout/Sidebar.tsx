"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  Camera,
  History,
  LayoutGrid,
} from "lucide-react";
import { Logo } from "./Logo";
import { fetchHealth } from "@/lib/api";

interface NavItem {
  name: string;
  href: string;
  icon: React.ComponentType<{ className?: string }>;
  pattern: RegExp;
}

const NAV_ITEMS: NavItem[] = [
  {
    name: "Dashboard",
    href: "/",
    icon: LayoutDashboard,
    pattern: /^(\/|\/dashboard)$/,
  },
  {
    name: "New Inspection",
    href: "/new-inspection",
    icon: Camera,
    pattern: /^\/new-inspection/,
  },
  {
    name: "Inspection History",
    href: "/inspection-history",
    icon: History,
    pattern: /^\/inspection-history/,
  },
  {
    name: "Panel Details",
    href: "/panel-details",
    icon: LayoutGrid,
    pattern: /^\/panel-details/,
  },
];

export function Sidebar() {
  const pathname = usePathname();
  const [isOnline, setIsOnline] = useState(true);

  useEffect(() => {
    let isMounted = true;
    async function checkHealth() {
      try {
        const health = await fetchHealth();
        if (isMounted) {
          setIsOnline(health.connected === true && health.status === "ok");
        }
      } catch {
        if (isMounted) setIsOnline(false);
      }
    }
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <aside className="fixed left-0 top-0 h-full w-64 bg-surface-container-lowest border-r border-outline-variant/30 z-50 flex flex-col justify-between">
      <div className="flex flex-col">
        {/* Brand Header */}
        <div className="h-16 px-5 flex items-center border-b border-outline-variant/20">
          <Link href="/" className="flex items-center">
            <Logo />
          </Link>
        </div>

        {/* Navigation items */}
        <nav className="p-4 flex flex-col gap-1.5">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = item.pattern.test(pathname);
            return (
              <Link
                key={item.name}
                href={item.href}
                className={`flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-primary text-on-primary shadow-sm font-semibold"
                    : "text-on-surface-variant hover:bg-surface-container hover:text-on-surface"
                }`}
              >
                <Icon className={`w-5 h-5 ${isActive ? "text-on-primary" : "text-outline"}`} />
                <span>{item.name}</span>
              </Link>
            );
          })}
        </nav>
      </div>

      {/* AI Inspection System Status */}
      <div className="p-4 border-t border-outline-variant/20 bg-surface-container-low/50">
        <div className="flex items-center gap-2 mb-1">
          <span
            className={`w-2.5 h-2.5 rounded-full ${
              isOnline ? "bg-emerald-500 animate-pulse" : "bg-amber-500"
            }`}
          ></span>
          <span className="text-xs font-semibold text-on-surface">
            AI Inspection System
          </span>
        </div>
        <p className="text-xs text-on-surface-variant pl-4.5">
          {isOnline ? "Ready to analyze solar panel images" : "Service offline (demo mode)"}
        </p>
      </div>
    </aside>
  );
}
