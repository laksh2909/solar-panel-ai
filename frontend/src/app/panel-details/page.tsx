"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { LayoutGrid, MapPin, ChevronRight, Camera, Search } from "lucide-react";
import { AppShell } from "@/components/layout/AppShell";
import { Disclaimer } from "@/components/common/Disclaimer";
import { fetchPanels } from "@/lib/api";
import { PanelResponse } from "@/types/inspection";
import { MOCK_PANELS } from "@/lib/mock-data";

export default function PanelListPage() {
  const [panels, setPanels] = useState<PanelResponse[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");

  useEffect(() => {
    let isMounted = true;
    async function loadPanels() {
      try {
        const res = await fetchPanels();
        if (isMounted) {
          if (res?.items && res.items.length > 0) {
            setPanels(res.items);
          } else {
            setPanels(MOCK_PANELS);
          }
        }
      } catch (err) {
        console.error("Failed to load panels:", err);
        if (isMounted) {
          setPanels(MOCK_PANELS);
        }
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    }
    loadPanels();
    return () => {
      isMounted = false;
    };
  }, []);

  const filteredPanels = panels.filter((p) => {
    const q = searchQuery.toLowerCase().trim();
    return (
      q === "" ||
      p.panel_id.toLowerCase().includes(q) ||
      p.location.toLowerCase().includes(q)
    );
  });

  return (
    <AppShell
      breadcrumbs={[
        { label: "Dashboard", href: "/" },
        { label: "Panel Details", active: true },
      ]}
    >
      <div className="flex flex-col gap-6 max-w-5xl mx-auto w-full">
        {/* Page Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-on-surface tracking-tight">
              Panel Registry
            </h1>
            <p className="text-sm text-on-surface-variant mt-1">
              Select a solar panel to view its operational history and inspection records.
            </p>
          </div>
          <Link
            href="/new-inspection"
            className="inline-flex items-center gap-2 px-4 py-2 bg-primary hover:bg-primary-container text-on-primary text-xs font-semibold rounded-lg shadow-sm transition-colors self-start sm:self-auto"
          >
            <Camera className="w-4 h-4" />
            <span>New Inspection</span>
          </Link>
        </div>

        {/* Search */}
        <div className="bg-surface-container-lowest p-4 rounded-xl border border-outline-variant/30 shadow-sm flex items-center gap-3">
          <Search className="w-4 h-4 text-outline shrink-0 ml-1" />
          <input
            type="text"
            placeholder="Search panels by ID or location..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full text-xs bg-transparent text-on-surface placeholder:text-outline focus:outline-none"
          />
        </div>

        {/* Panels Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {isLoading ? (
            <div className="col-span-full py-12 text-center text-sm text-on-surface-variant">
              Loading registered panels...
            </div>
          ) : filteredPanels.length === 0 ? (
            <div className="col-span-full py-12 text-center text-sm text-on-surface-variant">
              No panels found matching &quot;{searchQuery}&quot;.
            </div>
          ) : (
            filteredPanels.map((panel) => (
              <Link
                key={panel.panel_id}
                href={`/panel-details/${panel.panel_id}`}
                className="bg-surface-container-lowest p-5 rounded-xl border border-outline-variant/30 hover:border-primary/50 shadow-sm hover:shadow transition-all flex flex-col justify-between group"
              >
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-xs font-semibold text-primary px-2 py-0.5 rounded bg-primary/10">
                      {panel.panel_id}
                    </span>
                    <ChevronRight className="w-4 h-4 text-outline group-hover:text-primary group-hover:translate-x-0.5 transition-all" />
                  </div>
                  <h2 className="text-base font-semibold text-on-surface">
                    {panel.panel_id}
                  </h2>
                  <p className="text-xs text-on-surface-variant mt-1.5 flex items-center gap-1.5">
                    <MapPin className="w-3.5 h-3.5 text-outline shrink-0" />
                    <span className="truncate">{panel.location}</span>
                  </p>
                </div>
                <div className="mt-4 pt-3 border-t border-outline-variant/20 flex items-center justify-between text-xs text-on-surface-variant">
                  <span>View Details</span>
                  <span className="text-primary font-medium group-hover:underline">
                    Inspect records &rarr;
                  </span>
                </div>
              </Link>
            ))
          )}
        </div>

        {/* Disclaimer */}
        <Disclaimer variant="compact" />
      </div>
    </AppShell>
  );
}
