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
  Filter,
  Search,
  Cpu,
} from "lucide-react";
import { AppShell } from "@/components/layout/AppShell";
import { SeverityBadge, UrgencyBadge } from "@/components/common/StatusBadge";
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
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-on-surface tracking-tight">
              Solar Panel Visual Inspection
            </h1>
            <p className="text-sm text-on-surface-variant mt-1">
              Upload a solar panel image to identify visible faults, estimate visual severity, and receive maintenance guidance.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <Link
              href="/new-inspection"
              className="inline-flex items-center gap-2 px-4 py-2.5 bg-primary hover:bg-primary-container text-on-primary text-sm font-semibold rounded-lg shadow-sm transition-colors"
            >
              <Camera className="w-4 h-4" />
              <span>Start New Inspection</span>
            </Link>
          </div>
        </div>

        {/* 4 Simple KPI Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Card 1: Total Panels */}
          <div className="bg-surface-container-lowest p-5 rounded-xl border border-outline-variant/30 shadow-sm flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-on-surface-variant uppercase tracking-wider">
                Total Panels
              </span>
              <div className="w-8 h-8 rounded-lg bg-surface-container flex items-center justify-center text-primary">
                <LayoutGrid className="w-4 h-4" />
              </div>
            </div>
            <div className="mt-4">
              <div className="text-3xl font-bold text-on-surface tracking-tight">
                {isLoading ? "—" : totalPanels !== null ? totalPanels : "No data"}
              </div>
              <p className="text-xs text-on-surface-variant mt-1.5">
                {totalPanels !== null ? `${totalPanels} panels registered` : "No panels registered"}
              </p>
            </div>
          </div>

          {/* Card 2: Total Inspections */}
          <div className="bg-surface-container-lowest p-5 rounded-xl border border-outline-variant/30 shadow-sm flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-on-surface-variant uppercase tracking-wider">
                Total Inspections
              </span>
              <div className="w-8 h-8 rounded-lg bg-surface-container flex items-center justify-center text-primary">
                <ClipboardCheck className="w-4 h-4" />
              </div>
            </div>
            <div className="mt-4">
              <div className="text-3xl font-bold text-on-surface tracking-tight">
                {isLoading ? "—" : totalInspections !== null ? totalInspections : "No data"}
              </div>
              <p className="text-xs text-on-surface-variant mt-1.5">
                {totalInspections !== null
                  ? `${totalInspections} inspections completed`
                  : "No inspections recorded"}
              </p>
            </div>
          </div>

          {/* Card 3: High-Severity Cases */}
          <div className="bg-surface-container-lowest p-5 rounded-xl border border-outline-variant/30 shadow-sm flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-rose-700 uppercase tracking-wider">
                High-Severity Cases
              </span>
              <div className="w-8 h-8 rounded-lg bg-rose-50 flex items-center justify-center text-rose-600">
                <AlertTriangle className="w-4 h-4" />
              </div>
            </div>
            <div className="mt-4">
              <div className="text-3xl font-bold text-rose-600 tracking-tight">
                {isLoading ? "—" : highSeverityCount}
              </div>
              <p className="text-xs text-on-surface-variant mt-1.5">
                {highSeverityCount > 0
                  ? `${highSeverityCount} inspections require priority attention`
                  : "No high-severity inspections"}
              </p>
            </div>
          </div>

          {/* Card 4: Manual Review Required */}
          <div className="bg-surface-container-lowest p-5 rounded-xl border border-outline-variant/30 shadow-sm flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-on-surface-variant uppercase tracking-wider">
                Manual Review Required
              </span>
              <div className="w-8 h-8 rounded-lg bg-surface-container flex items-center justify-center text-on-surface-variant">
                <UserCheck className="w-4 h-4" />
              </div>
            </div>
            <div className="mt-4">
              <div className="text-3xl font-bold text-on-surface tracking-tight">
                {isLoading ? "—" : manualReviewCount}
              </div>
              <p className="text-xs text-on-surface-variant mt-1.5">
                {manualReviewCount > 0
                  ? `${manualReviewCount} inspections require manual review`
                  : "All inspections verified"}
              </p>
            </div>
          </div>
        </div>

        {/* Primary Action Area */}
        <div className="bg-surface-container-lowest p-6 rounded-xl border border-outline-variant/30 shadow-sm flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div className="flex items-start gap-4">
            <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center text-primary shrink-0">
              <Camera className="w-6 h-6" />
            </div>
            <div>
              <h2 className="text-base font-semibold text-on-surface">
                Start New Inspection
              </h2>
              <p className="text-sm text-on-surface-variant mt-0.5 max-w-xl">
                Upload a solar panel image to identify visible faults, estimate visual severity, and receive an inspection summary with maintenance guidance.
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3 w-full md:w-auto shrink-0">
            <Link
              href="/new-inspection"
              className="w-full md:w-auto inline-flex items-center justify-center gap-2 px-5 py-2.5 bg-primary hover:bg-primary-container text-on-primary text-sm font-semibold rounded-lg shadow-sm transition-colors"
            >
              <Camera className="w-4 h-4" />
              <span>Start New Inspection</span>
            </Link>
          </div>
        </div>

        {/* Main Grid: Recent Inspections Table + Side Cards */}
        <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 items-start">
          {/* Recent Inspections Table (8 cols on xl) */}
          <div className="xl:col-span-8 bg-surface-container-lowest rounded-xl border border-outline-variant/30 shadow-sm overflow-hidden flex flex-col">
            <div className="p-4 sm:p-5 bg-surface-container-low/40 border-b border-outline-variant/20 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <h2 className="text-base font-semibold text-on-surface">
                  Recent Inspections
                </h2>
                <p className="text-xs text-on-surface-variant mt-0.5">
                  Latest visual diagnostic records
                </p>
              </div>

              {/* Search filter input */}
              <div className="relative">
                <Search className="w-4 h-4 absolute left-3 top-2.5 text-outline" />
                <input
                  type="text"
                  placeholder="Filter by panel or fault..."
                  value={searchFilter}
                  onChange={(e) => setSearchFilter(e.target.value)}
                  className="pl-9 pr-3 py-1.5 text-xs bg-surface border border-outline-variant/40 rounded-lg text-on-surface placeholder:text-outline focus:outline-none focus:border-primary w-full sm:w-56 transition-colors"
                />
              </div>
            </div>

            {/* Table */}
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-sm">
                <thead>
                  <tr className="bg-surface-container-low/60 text-on-surface-variant text-xs font-semibold uppercase tracking-wider border-b border-outline-variant/20">
                    <th className="py-3 px-4">Panel ID</th>
                    <th className="py-3 px-4">Location</th>
                    <th className="py-3 px-4">Detected Fault</th>
                    <th className="py-3 px-4 text-right">Confidence</th>
                    <th className="py-3 px-4 text-center">Severity</th>
                    <th className="py-3 px-4">Recommended Action</th>
                    <th className="py-3 px-4">Date</th>
                    <th className="py-3 px-4 text-right">View</th>
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
                      <tr
                        key={row.inspection_id}
                        className="hover:bg-surface-container-low/50 transition-colors"
                      >
                        <td className="py-3.5 px-4 font-medium text-primary">
                          <Link
                            href={`/panel-details/${row.panel_id}`}
                            className="hover:underline"
                          >
                            {row.panel_id}
                          </Link>
                        </td>
                        <td className="py-3.5 px-4 text-xs text-on-surface-variant">
                          {row.location}
                        </td>
                        <td className="py-3.5 px-4">
                          <span className="font-medium text-on-surface">
                            {formatFaultName(row.predicted_class)}
                          </span>
                        </td>
                        <td className="py-3.5 px-4 text-right font-medium text-on-surface text-xs">
                          {(row.confidence * 100).toFixed(1)}%
                        </td>
                        <td className="py-3.5 px-4 text-center">
                          <SeverityBadge severity={row.severity} showEstimateHint />
                        </td>
                        <td className="py-3.5 px-4 text-xs text-on-surface-variant max-w-xs truncate" title={row.maintenance_action}>
                          {row.maintenance_action || "Routine monitoring"}
                        </td>
                        <td className="py-3.5 px-4 text-xs text-on-surface-variant whitespace-nowrap">
                          {formatInspectionDate(row.inspection_timestamp)}
                        </td>
                        <td className="py-3.5 px-4 text-right">
                          <Link
                            href={`/inspection-result/${row.inspection_id}`}
                            className="inline-flex items-center px-2.5 py-1 text-xs font-semibold text-primary hover:bg-primary/10 rounded-md transition-colors"
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

            {/* Table Footer */}
            <div className="p-3.5 bg-surface-container-low/40 border-t border-outline-variant/20 flex items-center justify-between text-xs text-on-surface-variant">
              <span>
                Showing {displayedInspections.length} of {inspections.length} inspections
              </span>
              <Link
                href="/inspection-history"
                className="font-medium text-primary hover:underline inline-flex items-center gap-1"
              >
                <span>View all history</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Link>
            </div>
          </div>

          {/* Right Auxiliary Column (4 cols on xl) */}
          <div className="xl:col-span-4 flex flex-col gap-6">
            {/* Fault Distribution */}
            <div className="bg-surface-container-lowest p-5 rounded-xl border border-outline-variant/30 shadow-sm flex flex-col">
              <h2 className="text-base font-semibold text-on-surface">
                Fault Distribution
              </h2>
              <p className="text-xs text-on-surface-variant mt-0.5 mb-4">
                Breakdown across the 6 supported fault categories
              </p>

              {faultDistribution.length === 0 ? (
                <p className="text-xs text-on-surface-variant py-4 text-center">
                  No inspection distribution available yet.
                </p>
              ) : (
                <>
                  {/* Segmented bar */}
                  <div className="w-full h-2.5 bg-surface-container rounded-full overflow-hidden flex mb-4">
                    {faultDistribution.map((item) => (
                      <div
                        key={item.key}
                        className={`${item.color} h-full transition-all`}
                        style={{ width: `${item.percentage}%` }}
                        title={`${item.label}: ${item.count} (${item.percentage}%)`}
                      />
                    ))}
                  </div>

                  {/* Class list with actual counts */}
                  <div className="flex flex-col gap-2.5">
                    {faultDistribution.map((item) => (
                      <div
                        key={item.key}
                        className="flex items-center justify-between text-xs"
                      >
                        <div className="flex items-center gap-2">
                          <span className={`w-2.5 h-2.5 rounded-sm ${item.color}`}></span>
                          <span className="text-on-surface">{item.label}</span>
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

            {/* AI Inspection System Status */}
            <div className="bg-surface-container-lowest p-5 rounded-xl border border-outline-variant/30 shadow-sm flex flex-col">
              <div className="flex items-center gap-2 mb-2">
                <span
                  className={`w-2.5 h-2.5 rounded-full ${
                    isBackendOnline ? "bg-emerald-500 animate-pulse" : "bg-amber-500"
                  }`}
                ></span>
                <h2 className="text-base font-semibold text-on-surface">
                  AI Inspection System
                </h2>
              </div>
              <p className="text-xs text-on-surface-variant mb-4">
                {isBackendOnline
                  ? "Ready to analyze solar panel images"
                  : "Running in offline demo mode"}
              </p>

              {/* Clean technical info */}
              <div className="p-3 bg-surface-container-low/60 rounded-lg border border-outline-variant/20 flex flex-col gap-1.5 text-xs text-on-surface-variant">
                <div className="flex items-center justify-between">
                  <span>Model:</span>
                  <span className="font-semibold text-on-surface">EfficientNet-B0</span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Input:</span>
                  <span className="font-semibold text-on-surface">RGB image</span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Classes:</span>
                  <span className="font-semibold text-on-surface">6 Fault Categories</span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Explainability:</span>
                  <span className="font-semibold text-on-surface">Grad-CAM Attention</span>
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
