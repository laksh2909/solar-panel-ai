"use client";

import React, { useState, useEffect } from "react";
import { fetchHealth } from "@/lib/api";

interface HeaderProps {
  breadcrumbs?: { label: string; href?: string; active?: boolean }[];
}

export function Header({ breadcrumbs }: HeaderProps) {
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
    const interval = setInterval(checkBackend, 15000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);
  return (
    <header className="fixed top-0 left-72 right-0 h-16 bg-surface-container-lowest/90 backdrop-blur-md border-b border-outline-variant/30 z-40 px-space-lg flex items-center justify-between">
      <div className="flex items-center gap-space-md">
        {/* Breadcrumb path */}
        <div className="flex items-center gap-1.5 font-code-id text-code-id text-on-surface-variant">
          {breadcrumbs && breadcrumbs.length > 0 ? (
            breadcrumbs.map((crumb, idx) => (
              <React.Fragment key={idx}>
                {idx > 0 && <span className="text-outline">/</span>}
                <span
                  className={
                    crumb.active
                      ? "text-primary font-semibold"
                      : "text-on-surface font-medium"
                  }
                >
                  {crumb.label}
                </span>
              </React.Fragment>
            ))
          ) : (
            <>
              <span className="text-on-surface-variant">Asset</span>
              <span className="text-outline">/</span>
              <span className="text-on-surface font-semibold">Solar Farm Alpha</span>
              <span className="text-outline">/</span>
              <span className="text-primary font-semibold">SEC-4</span>
            </>
          )}
        </div>

        {/* Global Search Input */}
        <div className="relative ml-space-md hidden sm:block">
          <span className="material-symbols-outlined absolute left-2.5 top-2 text-outline text-lg">
            search
          </span>
          <input
            className="h-8 pl-8 pr-3 text-body-sm font-body-sm bg-surface border border-outline-variant/40 rounded focus:outline-none focus:border-primary w-64 text-on-surface placeholder:text-outline-variant transition-colors"
            placeholder="Search module ID, anomaly, string..."
            type="text"
          />
        </div>
      </div>

      <div className="flex items-center gap-space-md">
        {/* System Health Badge */}
        <div className="hidden xl:flex items-center gap-space-sm px-space-sm py-1 bg-surface rounded border border-outline-variant/30">
          <span
            className={`material-symbols-outlined text-sm ${
              isOnline ? "text-secondary" : "text-outline"
            }`}
          >
            {isOnline ? "check_circle" : "cloud_off"}
          </span>
          <span className="font-label-caps text-label-caps text-on-surface-variant font-medium">
            {isOnline ? "FASTAPI ONLINE" : "OFFLINE (DEMO)"}
          </span>
        </div>

        {/* Notification Bell */}
        <button
          className="relative p-1.5 text-on-surface-variant hover:text-on-surface hover:bg-surface-container rounded transition-colors"
          title="Notifications"
        >
          <span className="material-symbols-outlined text-xl">notifications</span>
          <span className="absolute top-1 right-1 w-2 h-2 bg-error rounded-full"></span>
        </button>

        <div className="h-6 w-px bg-outline-variant/30"></div>

        {/* User Tech Profile */}
        <div className="flex items-center gap-space-sm">
          <div className="flex flex-col text-right">
            <span className="font-body-md text-body-md text-on-surface font-semibold leading-tight">
              Eng. Marcus Vance
            </span>
            <span className="font-label-sm text-label-sm text-on-surface-variant leading-tight">
              Senior Field Tech
            </span>
          </div>
          <div className="w-8 h-8 rounded-full bg-primary-container text-on-primary font-bold text-xs flex items-center justify-center border border-outline-variant/40 shadow-sm">
            MV
          </div>
        </div>
      </div>
    </header>
  );
}
