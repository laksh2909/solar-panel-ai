"use client";

import React, { useState, useEffect, useMemo } from "react";
import Link from "next/link";
import {
  LayoutGrid,
  ClipboardCheck,
  AlertTriangle,
  UserCheck,
  Camera,
  ArrowRight,
  Search,
} from "lucide-react";
import { AppShell } from "@/components/layout/AppShell";
import { SeverityBadge } from "@/components/common/StatusBadge";
import { Disclaimer } from "@/components/common/Disclaimer";
import { fetchHealth, fetchPanels, fetchInspections } from "@/lib/api";
import { InspectionResponse, FaultType } from "@/types/inspection";
import { formatFaultName, formatInspectionDate } from "@/lib/formatters";

const FAULT_CLASSES: { key: FaultType; label: string; color: string }[] = [
  { key: "Clean", label: "Clean", color: "bg-emerald-500" },
  { key: "Dusty", label: "Dust", color: "bg-amber-400" },
  { key: "Bird-drop", label: "Bird Droppings", color: "bg-slate-400" },
  { key: "Electrical-damage", label: "Electrical Damage", color: "bg-rose-500" },
  { key: "Physical-damage", label: "Physical Damage", color: "bg-blue-500" },
  { key: "Snow-Covered", label: "Snow Covered", color: "bg-sky-300" },
];

export default function DashboardPage() {
  const [inspections, setInspections] = useState<InspectionResponse[]>([]);
  const [totalPanels, setTotalPanels] = useState<number | null>(null);
  const [totalInspections, setTotalInspections] = useState<number | null>(null);
  const [isBackendOnline, setIsBackendOnline] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [searchFilter, setSearchFilter] = useState<string>("");

  useEffect(() => {
    let isMounted = true;
    async function loadDashboardData() {
      try {
        const [health, panelsData, inspectionsData] = await Promise.all([
          fetchHealth(),
          fetchPanels(),
          fetchInspections(1, 50),
        ]);

        if (isMounted) {
          const online = health.connected === true && health.status === "ok";
          setIsBackendOnline(online);

          if (panelsData?.items) {
            setTotalPanels(panelsData.total || panelsData.items.length);
          }

          if (inspectionsData?.items) {
            setInspections(inspectionsData.items);
            setTotalInspections(inspectionsData.total || inspectionsData.items.length);
          }
        }
      } catch (err) {
        console.error("Failed to load dashboard data:", err);
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    }

    loadDashboardData();
    return () => {
      isMounted = false;
    };
  }, []);

  // Compute metrics from actual inspection records
  const highSeverityCount = useMemo(() => {
    return inspections.filter((i) => (i.severity || "").toUpperCase() === "HIGH").length;
  }, [inspections]);

  const manualReviewCount = useMemo(() => {
    return inspections.filter((i) => i.manual_inspection_recommended).length;
  }, [inspections]);

  // Compute actual fault distribution percentages from inspection records
  const faultDistribution = useMemo(() => {
    if (!inspections || inspections.length === 0) return [];
    const total = inspections.length;
    return FAULT_CLASSES.map((fc) => {
      const count = inspections.filter((i) => i.predicted_class === fc.key).length;
      const percentage = Math.round((count / total) * 100);
      return {
        ...fc,
        count,
        percentage,
      };
    });
  }, [inspections]);

  // Filter recent inspections for table
  const displayedInspections = useMemo(() => {
    if (!searchFilter.trim()) return inspections.slice(0, 10);
    const query = searchFilter.toLowerCase();
    return inspections.filter(
      (i) =>
        i.panel_id.toLowerCase().includes(query) ||
        i.location.toLowerCase().includes(query) ||
        formatFaultName(i.predicted_class).toLowerCase().includes(query)
    ).slice(0, 10);
  }, [inspections, searchFilter]);

  return (
    <AppShell>
      <div className="flex flex-col gap-6 w-full">
        {/* Page Header */}
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-on-surface-variant">
              Inspection Dashboard
            </p>
            <h1 className="mt-1 text-2xl font-bold tracking-tight text-on-surface">
              Solar Panel Inspection Overview
            </h1>
            <p className="mt-1 text-sm text-on-surface-variant">
              Review recent inspection activity, spot priority cases, and start a new visual assessment.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <Link
              href="/new-inspection"
              className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2.5 text-sm font-semibold text-on-primary shadow-sm transition-colors hover:bg-primary-container"
            >
              <Camera className="h-4 w-4" />
              <span>New Inspection</span>
            </Link>
          </div>
        </div>

        {/* KPI cards */}
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <div className="flex min-h-[150px] flex-col justify-between rounded-2xl border border-outline-variant/30 bg-surface-container-lowest p-5 shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-semibold uppercase tracking-[0.12em] text-on-surface-variant">
                Registered Panels
              </span>
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-surface-container text-primary">
                <LayoutGrid className="h-4 w-4" />
              </div>
            </div>
            <div className="mt-4">
              <div className="text-3xl font-bold tracking-tight text-on-surface">
                {isLoading ? "—" : totalPanels !== null ? totalPanels : "No data"}
              </div>
              <p className="mt-1.5 text-xs text-on-surface-variant">
                {totalPanels !== null ? "Currently on record" : "No panels registered"}
              </p>
            </div>
          </div>

          <div className="flex min-h-[150px] flex-col justify-between rounded-2xl border border-outline-variant/30 bg-surface-container-lowest p-5 shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-semibold uppercase tracking-[0.12em] text-on-surface-variant">
                Total Inspections
              </span>
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-surface-container text-primary">
                <ClipboardCheck className="h-4 w-4" />
              </div>
            </div>
            <div className="mt-4">
              <div className="text-3xl font-bold tracking-tight text-on-surface">
                {isLoading ? "—" : totalInspections !== null ? totalInspections : "No data"}
              </div>
              <p className="mt-1.5 text-xs text-on-surface-variant">
                {totalInspections !== null ? "Completed inspections" : "No inspections recorded"}
              </p>
            </div>
          </div>

          <div className="flex min-h-[150px] flex-col justify-between rounded-2xl border border-rose-200 bg-rose-50/40 p-5 shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-semibold uppercase tracking-[0.12em] text-rose-700">
                High Severity
              </span>
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-rose-100 text-rose-600">
                <AlertTriangle className="h-4 w-4" />
              </div>
            </div>
            <div className="mt-4">
              <div className="text-3xl font-bold tracking-tight text-rose-600">
                {isLoading ? "—" : highSeverityCount}
              </div>
              <p className="mt-1.5 text-xs text-on-surface-variant">
                {highSeverityCount > 0 ? "Need priority review" : "No high-severity cases"}
              </p>
            </div>
          </div>

          <div className="flex min-h-[150px] flex-col justify-between rounded-2xl border border-outline-variant/30 bg-surface-container-lowest p-5 shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-semibold uppercase tracking-[0.12em] text-on-surface-variant">
                Manual Review
              </span>
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-surface-container text-on-surface-variant">
                <UserCheck className="h-4 w-4" />
              </div>
            </div>
            <div className="mt-4">
              <div className="text-3xl font-bold tracking-tight text-on-surface">
                {isLoading ? "—" : manualReviewCount}
              </div>
              <p className="mt-1.5 text-xs text-on-surface-variant">
                {manualReviewCount > 0 ? "Require review" : "No manual review required"}
              </p>
            </div>
          </div>
        </div>

        {/* Primary Action Area */}
        <div className="flex flex-col gap-4 rounded-2xl border border-outline-variant/30 bg-surface-container-lowest p-5 shadow-sm md:flex-row md:items-center md:justify-between">
          <div className="flex items-start gap-4">
            <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary">
              <Camera className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-base font-semibold text-on-surface">New Inspection</h2>
              <p className="mt-0.5 max-w-xl text-sm text-on-surface-variant">
                Upload a solar panel image to begin a fresh visual assessment and review the result.
              </p>
            </div>
          </div>

          <Link
            href="/new-inspection"
            className="inline-flex w-full items-center justify-center gap-2 rounded-lg bg-primary px-5 py-2.5 text-sm font-semibold text-on-primary shadow-sm transition-colors hover:bg-primary-container md:w-auto"
          >
            <Camera className="h-4 w-4" />
            <span>New Inspection</span>
          </Link>
        </div>

        {/* Main Grid: Recent Inspections Table + Side Cards */}
        <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 items-start">
          {/* Recent Inspections Table (8 cols on xl) */}
          <div className="xl:col-span-8 bg-surface-container-lowest rounded-xl border border-outline-variant/30 shadow-sm overflow-hidden flex flex-col">
            <div className="flex flex-col gap-3 border-b border-outline-variant/20 bg-surface-container-low/40 p-4 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <h2 className="text-base font-semibold text-on-surface">Recent Inspections</h2>
                <p className="mt-0.5 text-xs text-on-surface-variant">
                  Latest visual inspection records
                </p>
              </div>

              <div className="relative w-full sm:w-60">
                <Search className="absolute left-3 top-2.5 h-4 w-4 text-outline" />
                <input
                  type="text"
                  placeholder="Filter by panel or fault..."
                  value={searchFilter}
                  onChange={(e) => setSearchFilter(e.target.value)}
                  className="w-full rounded-lg border border-outline-variant/40 bg-surface py-1.5 pl-9 pr-3 text-xs text-on-surface placeholder:text-outline focus:border-primary focus:outline-none"
                />
              </div>
            </div>

            <div className="hidden md:block overflow-x-auto">
              <table className="w-full border-collapse text-left text-sm">
                <thead>
                  <tr className="border-b border-outline-variant/20 bg-surface-container-low/50 text-[11px] font-semibold uppercase tracking-[0.08em] text-on-surface-variant">
                    <th className="px-4 py-3">Panel ID</th>
                    <th className="px-4 py-3">Location</th>
                    <th className="px-4 py-3">Fault</th>
                    <th className="px-4 py-3 text-right">Confidence</th>
                    <th className="px-4 py-3 text-center">Severity</th>
                    <th className="px-4 py-3">Action</th>
                    <th className="px-4 py-3">Date</th>
                    <th className="px-4 py-3 text-right">View</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-outline-variant/20">
                  {displayedInspections.length === 0 ? (
                    <tr>
                      <td colSpan={8} className="py-8 text-center text-sm text-on-surface-variant">
                        {isLoading ? "Loading inspections..." : "No inspection data yet"}
                      </td>
                    </tr>
                  ) : (
                    displayedInspections.map((row) => (
                      <tr key={row.inspection_id} className="transition-colors hover:bg-surface-container-low/40">
                        <td className="px-4 py-3 font-medium text-primary">
                          <Link href={`/panel-details/${row.panel_id}`} className="hover:underline">
                            {row.panel_id}
                          </Link>
                        </td>
                        <td className="px-4 py-3 text-xs text-on-surface-variant">{row.location}</td>
                        <td className="px-4 py-3">
                          <span className="font-medium text-on-surface">{formatFaultName(row.predicted_class)}</span>
                        </td>
                        <td className="px-4 py-3 text-right text-xs font-medium text-on-surface">
                          {(row.confidence * 100).toFixed(1)}%
                        </td>
                        <td className="px-4 py-3 text-center">
                          <SeverityBadge severity={row.severity} showEstimateHint />
                        </td>
                        <td
                          className="max-w-[180px] px-4 py-3 text-xs text-on-surface-variant"
                          title={row.maintenance_action}
                        >
                          {row.maintenance_action || "Routine monitoring"}
                        </td>
                        <td className="whitespace-nowrap px-4 py-3 text-xs text-on-surface-variant">
                          {formatInspectionDate(row.inspection_timestamp)}
                        </td>
                        <td className="px-4 py-3 text-right">
                          <Link
                            href={`/inspection-result/${row.inspection_id}`}
                            className="inline-flex items-center rounded-md px-2.5 py-1 text-xs font-semibold text-primary transition-colors hover:bg-primary/10"
                          >
                            View
                          </Link>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>

            <div className="space-y-3 p-3 md:hidden">
              {displayedInspections.length === 0 ? (
                <div className="rounded-xl border border-dashed border-outline-variant/40 bg-surface-container-low/40 px-4 py-8 text-center text-sm text-on-surface-variant">
                  {isLoading ? "Loading inspections..." : "No inspection data yet"}
                </div>
              ) : (
                displayedInspections.map((row) => (
                  <div
                    key={row.inspection_id}
                    className="rounded-2xl border border-outline-variant/30 bg-surface-container-lowest p-3.5 shadow-sm"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-on-surface-variant">
                          {row.inspection_id}
                        </p>
                        <p className="mt-1 text-xs text-on-surface-variant">
                          {formatInspectionDate(row.inspection_timestamp)}
                        </p>
                      </div>
                      <Link
                        href={`/inspection-result/${row.inspection_id}`}
                        className="inline-flex items-center justify-center rounded-md bg-primary px-2.5 py-1.5 text-[11px] font-semibold text-on-primary"
                      >
                        View
                      </Link>
                    </div>

                    <div className="mt-3 space-y-2 text-sm">
                      <div className="flex items-start justify-between gap-3">
                        <span className="text-on-surface-variant">Fault</span>
                        <span className="max-w-[60%] text-right font-medium text-on-surface">
                          {formatFaultName(row.predicted_class)}
                        </span>
                      </div>

                      <div className="flex items-start justify-between gap-3">
                        <span className="text-on-surface-variant">Severity</span>
                        <div className="max-w-[60%]">
                          <SeverityBadge severity={row.severity} showEstimateHint />
                        </div>
                      </div>

                      <div className="flex items-start justify-between gap-3">
                        <span className="text-on-surface-variant">Confidence</span>
                        <span className="font-medium text-on-surface">
                          {(row.confidence * 100).toFixed(1)}%
                        </span>
                      </div>

                      {row.location && (
                        <div className="flex items-start justify-between gap-3">
                          <span className="text-on-surface-variant">Location</span>
                          <span className="max-w-[60%] text-right text-on-surface">{row.location}</span>
                        </div>
                      )}
                    </div>
                  </div>
                ))
              )}
            </div>

            <div className="flex items-center justify-between border-t border-outline-variant/20 bg-surface-container-low/40 px-3.5 py-3 text-xs text-on-surface-variant">
              <span>
                Showing {displayedInspections.length} of {inspections.length} inspections
              </span>
              <Link href="/inspection-history" className="inline-flex items-center gap-1 font-medium text-primary hover:underline">
                <span>View all history</span>
                <ArrowRight className="h-3.5 w-3.5" />
              </Link>
            </div>
          </div>

          {/* Right Auxiliary Column (4 cols on xl) */}
          <div className="xl:col-span-4 flex flex-col gap-6">
            {/* Fault Distribution */}
            <div className="flex flex-col rounded-2xl border border-outline-variant/30 bg-surface-container-lowest p-5 shadow-sm">
              <h2 className="text-base font-semibold text-on-surface">Fault Categories</h2>
              <p className="mt-0.5 text-xs text-on-surface-variant">Recorded fault types from recent inspections</p>

              {faultDistribution.length === 0 ? (
                <p className="py-4 text-center text-xs text-on-surface-variant">
                  No inspection distribution available yet.
                </p>
              ) : (
                <>
                  <div className="mt-4 flex h-2.5 overflow-hidden rounded-full bg-surface-container">
                    {faultDistribution.map((item) => (
                      <div
                        key={item.key}
                        className={`${item.color} h-full`}
                        style={{ width: `${item.percentage}%` }}
                        title={`${item.label}: ${item.count} (${item.percentage}%)`}
                      />
                    ))}
                  </div>

                  <div className="mt-4 space-y-2.5">
                    {faultDistribution.map((item) => (
                      <div key={item.key} className="flex items-center justify-between gap-3 text-xs">
                        <div className="flex min-w-0 items-center gap-2">
                          <span className={`h-2.5 w-2.5 rounded-sm ${item.color}`}></span>
                          <span className="truncate text-on-surface">{item.label}</span>
                        </div>
                        <span className="font-medium text-on-surface">
                          {item.count} ({item.percentage}%)
                        </span>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>

            <div className="flex flex-col rounded-2xl border border-outline-variant/30 bg-surface-container-lowest p-5 shadow-sm">
              <div className="mb-3 flex items-center gap-2">
                <span
                  className={`h-2.5 w-2.5 rounded-full ${
                    isBackendOnline ? "bg-emerald-500" : "bg-amber-500"
                  }`}
                ></span>
                <h2 className="text-base font-semibold text-on-surface">System Status</h2>
              </div>
              <p className="text-xs text-on-surface-variant">
                {isBackendOnline ? "Ready to analyze new images" : "Offline demo mode"}
              </p>

              <div className="mt-4 space-y-2 rounded-xl border border-outline-variant/20 bg-surface-container-low/60 p-3 text-xs text-on-surface-variant">
                <div className="flex items-center justify-between gap-3">
                  <span>Analysis model</span>
                  <span className="font-semibold text-on-surface">EfficientNet-B0</span>
                </div>
                <div className="flex items-center justify-between gap-3">
                  <span>Input type</span>
                  <span className="font-semibold text-on-surface">RGB image</span>
                </div>
                <div className="flex items-center justify-between gap-3">
                  <span>Fault classes</span>
                  <span className="font-semibold text-on-surface">6 categories</span>
                </div>
                <div className="flex items-center justify-between gap-3">
                  <span>Attention</span>
                  <span className="font-semibold text-on-surface">Grad-CAM</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Disclaimer Footer */}
        <Disclaimer />
      </div>
    </AppShell>
  );
}
