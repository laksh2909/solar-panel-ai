"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { Bell, ChevronRight, User } from "lucide-react";
import { fetchHealth } from "@/lib/api";
import { useToast } from "@/components/common/Toast";

interface HeaderProps {
  breadcrumbs?: { label: string; href?: string; active?: boolean }[];
}

export function Header({ breadcrumbs }: HeaderProps) {
  const { showToast } = useToast();
  const [isOnline, setIsOnline] = useState(false);

  useEffect(() => {
    let isMounted = true;
    async function checkBackend() {
      try {
        const health = await fetchHealth();
        if (isMounted) {
          setIsOnline(health.connected === true && health.status === "ok");
        }
      } catch {
        if (isMounted) setIsOnline(false);
      }
    }
    checkBackend();
    const interval = setInterval(checkBackend, 20000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <header className="fixed top-0 left-64 right-0 h-16 bg-surface-container-lowest/95 backdrop-blur-sm border-b border-outline-variant/30 z-40 px-6 flex items-center justify-between">
      {/* Left: Title or Clean Breadcrumbs */}
      <div className="flex items-center gap-3">
        {breadcrumbs && breadcrumbs.length > 0 ? (
          <nav aria-label="Breadcrumb" className="flex items-center gap-1.5 text-xs text-on-surface-variant">
            {breadcrumbs.map((crumb, idx) => (
              <React.Fragment key={idx}>
                {idx > 0 && <ChevronRight className="w-3.5 h-3.5 text-outline" />}
                {crumb.href && !crumb.active ? (
                  <Link
                    href={crumb.href}
                    className="hover:text-primary transition-colors font-medium"
                  >
                    {crumb.label}
                  </Link>
                ) : (
                  <span
                    className={
                      crumb.active
                        ? "text-on-surface font-semibold"
                        : "text-on-surface-variant font-medium"
                    }
                  >
                    {crumb.label}
                  </span>
                )}
              </React.Fragment>
            ))}
          </nav>
        ) : (
          <div className="flex flex-col">
            <h1 className="text-sm font-semibold text-on-surface leading-tight">
              Solar Panel Inspection
            </h1>
            <span className="text-xs text-on-surface-variant leading-tight">
              AI-assisted visual inspection and maintenance support
            </span>
          </div>
        )}
      </div>

      {/* Right: Status, Notifications, User */}
      <div className="flex items-center gap-4">
        {/* System Online Badge */}
        <div className="flex items-center gap-2 px-2.5 py-1 bg-surface-container-low rounded-full border border-outline-variant/30 text-xs font-medium">
          <span
            className={`w-2 h-2 rounded-full ${
              isOnline ? "bg-emerald-500 animate-pulse" : "bg-amber-500"
            }`}
          ></span>
          <span className="text-on-surface-variant">
            {isOnline ? "System Online" : "Demo Mode"}
          </span>
        </div>

        {/* Notifications */}
        <button
          onClick={() => showToast("No new unread inspection alerts.")}
          className="relative p-2 text-on-surface-variant hover:text-on-surface hover:bg-surface-container rounded-lg transition-colors"
          title="Notifications"
          aria-label="View notifications"
        >
          <Bell className="w-4.5 h-4.5" />
          <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-primary rounded-full"></span>
        </button>

        <div className="h-5 w-px bg-outline-variant/30"></div>

        {/* User Profile */}
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-full bg-primary/10 text-primary flex items-center justify-center font-semibold text-xs border border-primary/20">
            <User className="w-4 h-4" />
          </div>
          <div className="hidden sm:flex flex-col text-left">
            <span className="text-xs font-semibold text-on-surface leading-tight">
              Field Technician
            </span>
            <span className="text-[11px] text-on-surface-variant leading-tight">
              Maintenance Team
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
