"use client";

import React, { useState, useEffect, useMemo } from "react";
import Link from "next/link";
import { AppShell } from "@/components/layout/AppShell";
import { SeverityBadge, UrgencyBadge } from "@/components/common/StatusBadge";
import { Disclaimer } from "@/components/common/Disclaimer";
import { useToast } from "@/components/common/Toast";
import { MOCK_INSPECTIONS } from "@/lib/mock-data";
import { fetchInspections } from "@/lib/api";
import { InspectionResponse } from "@/types/inspection";

export default function InspectionHistoryPage() {
  const { showToast } = useToast();

  const [allInspections, setAllInspections] = useState<InspectionResponse[]>(MOCK_INSPECTIONS);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedFault, setSelectedFault] = useState("ALL");
  const [selectedSeverity, setSelectedSeverity] = useState("ALL");
  const [selectedUrgency, setSelectedUrgency] = useState("ALL");

  useEffect(() => {
    let isMounted = true;
    async function loadData() {
      try {
        const res = await fetchInspections(1, 100);
        if (isMounted && res?.items && res.items.length > 0) {
          setAllInspections(res.items);
        }
      } catch {
        // Fallback to MOCK_INSPECTIONS
      }
    }
    loadData();
    return () => {
      isMounted = false;
    };
  }, []);

  const filteredInspections = useMemo(() => {
    return allInspections.filter((item) => {
      const matchesSearch =
        searchQuery === "" ||
        item.panel_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
        item.location.toLowerCase().includes(searchQuery.toLowerCase()) ||
        String(item.inspection_id).toLowerCase().includes(searchQuery.toLowerCase());

      const matchesFault =
        selectedFault === "ALL" || item.predicted_class === selectedFault;

      const matchesSeverity =
        selectedSeverity === "ALL" || item.severity === selectedSeverity;

      const matchesUrgency =
        selectedUrgency === "ALL" || item.urgency === selectedUrgency;

      return matchesSearch && matchesFault && matchesSeverity && matchesUrgency;
    });
  }, [allInspections, searchQuery, selectedFault, selectedSeverity, selectedUrgency]);

  const handleResetFilters = () => {
    setSearchQuery("");
    setSelectedFault("ALL");
    setSelectedSeverity("ALL");
    setSelectedUrgency("ALL");
    showToast("Filters reset to default.");
  };

  return (
    <AppShell
      breadcrumbs={[
        { label: "Asset", href: "/" },
        { label: "Solar Farm Alpha", href: "/" },
        { label: "SEC-4", href: "/" },
        { label: "Inspection History", active: true },
      ]}
    >
      <div className="flex flex-col w-full gap-space-md">
        {/* Top Summary & KPI Header */}
        <div className="grid grid-cols-1 xl:grid-cols-12 gap-space-md items-stretch">
          <div className="xl:col-span-5 bg-surface-container-lowest p-space-md rounded-lg shadow-sm border border-outline-variant/30 flex flex-col justify-between">
            <div>
              <div className="flex items-center gap-space-xs font-label-caps text-label-caps text-primary uppercase tracking-widest mb-1">
                <span className="w-2 h-2 rounded-full bg-secondary"></span>
                Inspection History &amp; Audit Trail
              </div>
              <h1 className="font-headline-lg text-headline-lg text-on-surface tracking-tight">
                Historical Telemetry Ledger
              </h1>
              <p className="font-body-sm text-body-sm text-on-surface-variant mt-1">
                Comprehensive archive of high-resolution optical RGB panel inspections,
                EfficientNet-B0 anomaly diagnostics, and technician triages.
              </p>
            </div>
            <div className="flex items-center gap-2 mt-3 flex-wrap">
              <div className="flex items-center gap-2 px-space-sm py-1 rounded bg-surface-container-low text-on-surface font-body-sm text-body-sm border border-outline-variant/20">
                <span className="material-symbols-outlined text-primary text-base">
                  model_training
                </span>
                <span>EfficientNet-B0 Baseline</span>
              </div>
              <div className="px-space-sm py-1 rounded bg-surface-container-low text-on-surface-variant font-code-id text-code-id border border-outline-variant/20">
                GET /api/inspections
              </div>
            </div>
          </div>

          <div className="xl:col-span-7 grid grid-cols-1 sm:grid-cols-3 gap-space-sm">
            {/* 30-Day Criticals */}
            <div className="bg-surface-container-lowest p-space-md rounded-lg shadow-sm border border-outline-variant/30 flex flex-col justify-between">
              <div className="flex items-center justify-between">
                <span className="font-label-caps text-label-caps text-on-surface-variant uppercase">
                  30-Day Criticals
                </span>
                <span className="p-1 rounded bg-error-container text-on-error-container material-symbols-outlined text-sm">
                  warning
                </span>
              </div>
              <div className="my-space-xs">
                <div className="font-headline-lg text-headline-lg text-error font-semibold">
                  142
                </div>
                <div className="font-body-sm text-body-sm text-on-surface-variant flex items-center gap-1">
                  <span className="text-error font-semibold">+8.4%</span> vs prior cycle
                </div>
              </div>
              {/* Sparkline mini chart */}
              <svg
                className="w-full h-8 text-error"
                preserveAspectRatio="none"
                viewBox="0 0 100 25"
              >
                <path
                  d="M0,20 L15,18 L30,12 L45,19 L60,10 L75,14 L90,4 L100,8"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  vectorEffect="non-scaling-stroke"
                ></path>
                <path
                  d="M0,20 L15,18 L30,12 L45,19 L60,10 L75,14 L90,4 L100,8 L100,25 L0,25 Z"
                  fill="currentColor"
                  fillOpacity="0.12"
                ></path>
              </svg>
            </div>

            {/* Mean AI Confidence */}
            <div className="bg-surface-container-lowest p-space-md rounded-lg shadow-sm border border-outline-variant/30 flex flex-col justify-between">
              <div className="flex items-center justify-between">
                <span className="font-label-caps text-label-caps text-on-surface-variant uppercase">
                  Mean AI Confidence
                </span>
                <span className="p-1 rounded bg-primary-fixed text-on-primary-fixed-variant material-symbols-outlined text-sm">
                  auto_graph
                </span>
              </div>
              <div className="my-space-xs">
                <div className="font-headline-lg text-headline-lg text-primary font-semibold">
                  94.3%
                </div>
                <div className="font-body-sm text-body-sm text-on-surface-variant flex items-center gap-1">
                  <span className="text-secondary font-semibold">+1.2%</span>{" "}
                  post-calibration
                </div>
              </div>
              <div className="w-full bg-surface-container h-2 rounded-full overflow-hidden flex">
                <div className="bg-primary h-full" style={{ width: "78%" }}></div>
                <div className="bg-secondary h-full" style={{ width: "16%" }}></div>
                <div className="bg-outline h-full" style={{ width: "6%" }}></div>
              </div>
            </div>

            {/* Human Triaged */}
            <div className="bg-surface-container-lowest p-space-md rounded-lg shadow-sm border border-outline-variant/30 flex flex-col justify-between">
              <div className="flex items-center justify-between">
                <span className="font-label-caps text-label-caps text-on-surface-variant uppercase">
                  Human Triaged
                </span>
                <span className="p-1 rounded bg-secondary-container text-on-secondary-container material-symbols-outlined text-sm">
                  assignment_turned_in
                </span>
              </div>
              <div className="my-space-xs">
                <div className="font-headline-lg text-headline-lg text-secondary font-semibold">
                  88.9%
                </div>
                <div className="font-body-sm text-body-sm text-on-surface-variant flex items-center gap-1">
                  <span className="text-on-surface-variant">3,418 resolved</span>
                </div>
              </div>
              <div className="flex items-center justify-between text-label-caps font-label-caps text-on-surface-variant">
                <span>PENDING: 427</span>
                <span className="text-secondary font-semibold">SLA: 4.2h</span>
              </div>
            </div>
          </div>
        </div>

        {/* Filter & Search Controller Hub */}
        <div className="bg-surface-container-lowest rounded-lg p-space-md shadow-sm border border-outline-variant/30 flex flex-col gap-space-sm">
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-12 gap-space-sm items-center">
            {/* Search Input */}
            <div className="xl:col-span-4 relative">
              <span className="material-symbols-outlined absolute left-3 top-2.5 text-outline text-lg">
                search
              </span>
              <input
                className="h-9 w-full pl-9 pr-8 bg-surface border border-outline-variant/40 rounded text-body-sm font-body-sm text-on-surface placeholder:text-outline-variant focus:outline-none focus:border-primary transition-all"
                id="search-input"
                placeholder="Search by Panel ID (e.g. SP-HYD-001) or Location..."
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
              {searchQuery && (
                <button
                  className="absolute right-2.5 top-2.5 text-outline hover:text-on-surface text-sm material-symbols-outlined"
                  onClick={() => setSearchQuery("")}
                >
                  close
                </button>
              )}
            </div>

            {/* Filter: Fault Type */}
            <div className="xl:col-span-2">
              <div className="relative">
                <select
                  className="h-9 w-full appearance-none pl-3 pr-8 bg-surface border border-outline-variant/40 rounded text-body-sm font-body-sm text-on-surface focus:outline-none focus:border-primary cursor-pointer"
                  value={selectedFault}
                  onChange={(e) => setSelectedFault(e.target.value)}
                >
                  <option value="ALL">All Faults</option>
                  <option value="Bird-drop">Bird-drop</option>
                  <option value="Clean">Clean</option>
                  <option value="Dusty">Dusty</option>
                  <option value="Electrical-damage">Electrical-damage</option>
                  <option value="Physical-damage">Physical-damage</option>
                  <option value="Snow-Covered">Snow-Covered</option>
                </select>
                <span className="material-symbols-outlined absolute right-2.5 top-2.5 pointer-events-none text-outline text-base">
                  expand_more
                </span>
              </div>
            </div>

            {/* Filter: Visual Severity */}
            <div className="xl:col-span-2">
              <div className="relative">
                <select
                  className="h-9 w-full appearance-none pl-3 pr-8 bg-surface border border-outline-variant/40 rounded text-body-sm font-body-sm text-on-surface focus:outline-none focus:border-primary cursor-pointer"
                  value={selectedSeverity}
                  onChange={(e) => setSelectedSeverity(e.target.value)}
                >
                  <option value="ALL">All Severities</option>
                  <option value="LOW">LOW</option>
                  <option value="MEDIUM">MEDIUM</option>
                  <option value="HIGH">HIGH</option>
                </select>
                <span className="material-symbols-outlined absolute right-2.5 top-2.5 pointer-events-none text-outline text-base">
                  tune
                </span>
              </div>
            </div>

            {/* Filter: Urgency */}
            <div className="xl:col-span-2">
              <div className="relative">
                <select
                  className="h-9 w-full appearance-none pl-3 pr-8 bg-surface border border-outline-variant/40 rounded text-body-sm font-body-sm text-on-surface focus:outline-none focus:border-primary cursor-pointer"
                  value={selectedUrgency}
                  onChange={(e) => setSelectedUrgency(e.target.value)}
                >
                  <option value="ALL">All Urgencies</option>
                  <option value="ROUTINE">ROUTINE</option>
                  <option value="SCHEDULED">SCHEDULED</option>
                  <option value="PRIORITY">PRIORITY</option>
                  <option value="IMMEDIATE REVIEW">IMMEDIATE REVIEW</option>
                </select>
                <span className="material-symbols-outlined absolute right-2.5 top-2.5 pointer-events-none text-outline text-base">
                  emergency_home
                </span>
              </div>
            </div>

            {/* Date Range Selector */}
            <div className="xl:col-span-2">
              <button
                onClick={() => showToast("Date filter: Current cycle active.")}
                className="h-9 w-full flex items-center justify-between px-space-sm bg-surface border border-outline-variant/40 rounded text-body-sm font-body-sm text-on-surface hover:bg-surface-container-low transition-colors"
              >
                <div className="flex items-center gap-1.5 truncate">
                  <span className="material-symbols-outlined text-primary text-base">
                    date_range
                  </span>
                  <span className="truncate font-code-id text-code-id">
                    Oct 1 - Oct 31, 2023
                  </span>
                </div>
                <span className="material-symbols-outlined text-outline text-sm">
                  keyboard_arrow_down
                </span>
              </button>
            </div>
          </div>

          {/* Active Tags & Export Buttons */}
          <div className="flex flex-wrap items-center justify-between gap-space-sm pt-2 border-t border-outline-variant/20">
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="font-label-caps text-label-caps text-on-surface-variant uppercase mr-1">
                Active Scope:
              </span>
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-surface-container font-code-id text-code-id text-on-surface">
                Date: Last 30 Days
              </span>
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-surface-container font-code-id text-code-id text-on-surface">
                Modality: High-Res Optical RGB
              </span>
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-surface-container-high font-code-id text-code-id text-primary font-semibold">
                Showing {filteredInspections.length} of {allInspections.length}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={handleResetFilters}
                className="px-2.5 py-1 text-on-surface-variant hover:text-on-surface font-label-sm text-label-sm"
              >
                Reset Filters
              </button>
              <button
                onClick={() =>
                  showToast(
                    `Exported ${filteredInspections.length} inspection records to CSV.`
                  )
                }
                className="px-3 py-1 bg-surface-container-low hover:bg-surface-container text-on-surface rounded font-label-sm text-label-sm flex items-center gap-1 border border-outline-variant/30"
              >
                <span className="material-symbols-outlined text-sm">download</span>
                Export Ledger CSV
              </button>
            </div>
          </div>
        </div>

        {/* Data Table Card */}
        <div className="bg-surface-container-lowest rounded-lg shadow-sm border border-outline-variant/30 overflow-hidden flex flex-col">
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-surface-container-low text-on-surface-variant font-label-caps text-label-caps uppercase border-b border-outline-variant/20">
                  <th className="py-2.5 px-3">Inspection ID</th>
                  <th className="py-2.5 px-3">Panel ID</th>
                  <th className="py-2.5 px-3">Location</th>
                  <th className="py-2.5 px-3">Anomaly Type</th>
                  <th className="py-2.5 px-3 text-right">Confidence</th>
                  <th className="py-2.5 px-3 text-center">Severity</th>
                  <th className="py-2.5 px-3 text-center">Urgency</th>
                  <th className="py-2.5 px-3 text-center">Manual Review</th>
                  <th className="py-2.5 px-3">Timestamp</th>
                  <th className="py-2.5 px-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="font-body-md text-body-md text-on-surface divide-y divide-outline-variant/20">
                {filteredInspections.length === 0 ? (
                  <tr>
                    <td
                      colSpan={10}
                      className="py-8 text-center text-on-surface-variant"
                    >
                      No inspection records match the current filter criteria.
                    </td>
                  </tr>
                ) : (
                  filteredInspections.map((row) => (
                    <tr
                      key={row.inspection_id}
                      className="hover:bg-surface-container-low transition-colors group cursor-pointer"
                      onClick={() =>
                        showToast(`Selected inspection ${row.inspection_id}`)
                      }
                    >
                      <td className="py-3 px-3 font-code-id text-code-id font-semibold text-primary">
                        <Link
                          href={`/inspection-result/${row.inspection_id}`}
                          className="hover:underline"
                          onClick={(e) => e.stopPropagation()}
                        >
                          #{row.inspection_id}
                        </Link>
                      </td>
                      <td className="py-3 px-3 font-code-id text-code-id font-semibold text-on-surface">
                        <Link
                          href={`/panel-details/${row.panel_id}`}
                          className="hover:text-primary transition-colors"
                          onClick={(e) => e.stopPropagation()}
                        >
                          {row.panel_id}
                        </Link>
                      </td>
                      <td className="py-3 px-3 font-body-sm text-body-sm text-on-surface-variant">
                        {row.location}
                      </td>
                      <td className="py-3 px-3">
                        <span className="inline-flex items-center gap-1.5 font-medium">
                          <span
                            className={`w-2 h-2 rounded-full ${
                              row.predicted_class === "Electrical-damage"
                                ? "bg-error"
                                : row.predicted_class === "Clean"
                                ? "bg-secondary"
                                : "bg-outline"
                            }`}
                          ></span>
                          <span
                            className={
                              row.predicted_class === "Electrical-damage"
                                ? "text-error font-semibold"
                                : row.predicted_class === "Clean"
                                ? "text-secondary font-semibold"
                                : "text-on-surface"
                            }
                          >
                            {row.predicted_class}
                          </span>
                        </span>
                      </td>
                      <td className="py-3 px-3 text-right font-code-metric text-code-id text-on-surface font-semibold">
                        {(row.confidence * 100).toFixed(1)}%
                      </td>
                      <td className="py-3 px-3 text-center">
                        <SeverityBadge severity={row.severity} />
                      </td>
                      <td className="py-3 px-3 text-center">
                        <UrgencyBadge urgency={row.urgency} />
                      </td>
                      <td className="py-3 px-3 text-center">
                        {row.manual_inspection_recommended ? (
                          <span className="px-1.5 py-0.5 rounded font-label-caps text-label-caps font-bold bg-error-container text-on-error-container">
                            FLAGGED
                          </span>
                        ) : (
                          <span className="px-1.5 py-0.5 rounded font-label-caps text-label-caps text-on-surface-variant bg-surface-container">
                            CLEAR
                          </span>
                        )}
                      </td>
                      <td className="py-3 px-3 font-code-id text-code-id text-on-surface-variant">
                        {row.inspection_timestamp.includes("T")
                          ? new Date(row.inspection_timestamp).toLocaleDateString(
                              "en-US",
                              {
                                month: "short",
                                day: "numeric",
                                hour: "2-digit",
                                minute: "2-digit",
                              }
                            )
                          : row.inspection_timestamp}
                      </td>
                      <td className="py-3 px-3 text-right">
                        <Link
                          href={`/inspection-result/${row.inspection_id}`}
                          onClick={(e) => e.stopPropagation()}
                          className="inline-block px-2.5 py-1 bg-surface-container hover:bg-primary hover:text-on-primary text-on-surface rounded font-label-sm text-label-sm transition-colors shadow-sm"
                        >
                          View Result
                        </Link>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>

          {/* Table Footer Pagination */}
          <div className="p-space-sm bg-surface-container-low flex items-center justify-between font-label-sm text-label-sm text-on-surface-variant border-t border-outline-variant/20">
            <span>
              Displaying {filteredInspections.length} of {MOCK_INSPECTIONS.length}{" "}
              records
            </span>
            <div className="flex items-center gap-1">
              <button
                disabled
                className="px-2 py-0.5 bg-surface-container-lowest rounded text-on-surface opacity-50 border border-outline-variant/20"
              >
                Prev
              </button>
              <span className="px-2 font-code-id text-on-surface font-semibold">
                Page 1 / 1
              </span>
              <button
                disabled
                className="px-2 py-0.5 bg-surface-container-lowest rounded text-on-surface opacity-50 border border-outline-variant/20"
              >
                Next
              </button>
            </div>
          </div>
        </div>

        {/* Statutory Disclaimer */}
        <Disclaimer />
      </div>
    </AppShell>
  );
}
