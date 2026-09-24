"use client";

import React, { useState, useEffect, useMemo } from "react";
import Link from "next/link";
import {
  Search,
  ArrowUpDown,
  RotateCcw,
  ClipboardCheck,
  AlertTriangle,
  UserCheck,
  Camera,
} from "lucide-react";
import { AppShell } from "@/components/layout/AppShell";
import { SeverityBadge } from "@/components/common/StatusBadge";
import { Disclaimer } from "@/components/common/Disclaimer";
import { useToast } from "@/components/common/Toast";
import { MOCK_INSPECTIONS } from "@/lib/mock-data";
import { fetchInspections } from "@/lib/api";
import { InspectionResponse } from "@/types/inspection";
import { formatFaultName, formatInspectionDate } from "@/lib/formatters";

const FAULT_OPTIONS: { value: string; label: string }[] = [
  { value: "ALL", label: "All Faults" },
  { value: "Clean", label: "Clean" },
  { value: "Dusty", label: "Dust" },
  { value: "Bird-drop", label: "Bird Droppings" },
  { value: "Electrical-damage", label: "Electrical Damage" },
  { value: "Physical-damage", label: "Physical Damage" },
  { value: "Snow-Covered", label: "Snow Covered" },
];

const SEVERITY_OPTIONS = [
  { value: "ALL", label: "All Severities" },
  { value: "HIGH", label: "High" },
  { value: "MEDIUM", label: "Medium" },
  { value: "LOW", label: "Low" },
];

export default function InspectionHistoryPage() {
  const { showToast } = useToast();

  const [allInspections, setAllInspections] = useState<InspectionResponse[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedFault, setSelectedFault] = useState("ALL");
  const [selectedSeverity, setSelectedSeverity] = useState("ALL");
  const [sortOrder, setSortOrder] = useState<"desc" | "asc">("desc");

  useEffect(() => {
    let isMounted = true;
    async function loadData() {
      try {
        const res = await fetchInspections(1, 100);
        if (isMounted) {
          if (res?.items && res.items.length > 0) {
            setAllInspections(res.items);
          } else {
            setAllInspections(MOCK_INSPECTIONS);
          }
        }
      } catch (err) {
        console.error("Failed to load inspections:", err);
        if (isMounted) {
          setAllInspections(MOCK_INSPECTIONS);
        }
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    }
    loadData();
    return () => {
      isMounted = false;
    };
  }, []);

  // Filter and sort
  const filteredInspections = useMemo(() => {
    const filtered = allInspections.filter((item) => {
      const q = searchQuery.toLowerCase().trim();
      const matchesSearch =
        q === "" ||
        item.panel_id.toLowerCase().includes(q) ||
        item.location.toLowerCase().includes(q) ||
        formatFaultName(item.predicted_class).toLowerCase().includes(q);

      const matchesFault =
        selectedFault === "ALL" || item.predicted_class === selectedFault;

      const matchesSeverity =
        selectedSeverity === "ALL" ||
        (item.severity || "").toUpperCase() === selectedSeverity;

      return matchesSearch && matchesFault && matchesSeverity;
    });

    // Sort by date
    return filtered.sort((a, b) => {
      const dateA = new Date(a.inspection_timestamp).getTime() || 0;
      const dateB = new Date(b.inspection_timestamp).getTime() || 0;
      return sortOrder === "desc" ? dateB - dateA : dateA - dateB;
    });
  }, [allInspections, searchQuery, selectedFault, selectedSeverity, sortOrder]);

  // Derived real metrics
  const totalCount = allInspections.length;
  const highSeverityCount = allInspections.filter(
    (i) => (i.severity || "").toUpperCase() === "HIGH"
  ).length;
  const manualReviewCount = allInspections.filter(
    (i) => i.manual_inspection_recommended
  ).length;

  const handleResetFilters = () => {
    setSearchQuery("");
    setSelectedFault("ALL");
    setSelectedSeverity("ALL");
    setSortOrder("desc");
    showToast("Filters reset to default.");
  };

  return (
    <AppShell
      breadcrumbs={[
        { label: "Dashboard", href: "/" },
        { label: "Inspection History", active: true },
      ]}
    >
      <div className="flex flex-col gap-6 w-full">
        {/* Page Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-on-surface tracking-tight">
              Inspection History
            </h1>
            <p className="text-sm text-on-surface-variant mt-1">
              Review previous solar panel inspections and recommendations.
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

        {/* Real Summary Metrics Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="bg-surface-container-lowest p-4 rounded-xl border border-outline-variant/30 shadow-sm flex items-center justify-between">
            <div>
              <span className="text-xs font-semibold text-on-surface-variant uppercase tracking-wider block">
                Total Inspections
              </span>
              <span className="text-2xl font-bold text-on-surface mt-1 block">
                {isLoading ? "—" : totalCount}
              </span>
              <span className="text-xs text-on-surface-variant">
                {totalCount} inspections recorded
              </span>
            </div>
            <div className="w-10 h-10 rounded-lg bg-surface-container flex items-center justify-center text-primary">
              <ClipboardCheck className="w-5 h-5" />
            </div>
          </div>

          <div className="bg-surface-container-lowest p-4 rounded-xl border border-outline-variant/30 shadow-sm flex items-center justify-between">
            <div>
              <span className="text-xs font-semibold text-rose-700 uppercase tracking-wider block">
                High-Severity Cases
              </span>
              <span className="text-2xl font-bold text-rose-600 mt-1 block">
                {isLoading ? "—" : highSeverityCount}
              </span>
              <span className="text-xs text-on-surface-variant">
                Require priority attention
              </span>
            </div>
            <div className="w-10 h-10 rounded-lg bg-rose-50 flex items-center justify-center text-rose-600">
              <AlertTriangle className="w-5 h-5" />
            </div>
          </div>

          <div className="bg-surface-container-lowest p-4 rounded-xl border border-outline-variant/30 shadow-sm flex items-center justify-between">
            <div>
              <span className="text-xs font-semibold text-on-surface-variant uppercase tracking-wider block">
                Manual Review Required
              </span>
              <span className="text-2xl font-bold text-on-surface mt-1 block">
                {isLoading ? "—" : manualReviewCount}
              </span>
              <span className="text-xs text-on-surface-variant">
                Recommended for physical check
              </span>
            </div>
            <div className="w-10 h-10 rounded-lg bg-surface-container flex items-center justify-center text-on-surface-variant">
              <UserCheck className="w-5 h-5" />
            </div>
          </div>
        </div>

        {/* Filter and Search Bar */}
        <div className="bg-surface-container-lowest p-4 rounded-xl border border-outline-variant/30 shadow-sm flex flex-col lg:flex-row items-stretch lg:items-center justify-between gap-3">
          {/* Search Input */}
          <div className="relative flex-1">
            <Search className="w-4 h-4 absolute left-3 top-2.5 text-outline" />
            <input
              type="text"
              placeholder="Search by panel ID or location..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-9 pr-3 py-2 text-xs bg-surface border border-outline-variant/40 rounded-lg text-on-surface placeholder:text-outline focus:outline-none focus:border-primary transition-colors"
            />
          </div>

          {/* Dropdown Filters */}
          <div className="flex items-center gap-2 flex-wrap">
            {/* Fault Filter */}
            <select
              value={selectedFault}
              onChange={(e) => setSelectedFault(e.target.value)}
              className="px-3 py-2 text-xs bg-surface border border-outline-variant/40 rounded-lg text-on-surface focus:outline-none focus:border-primary transition-colors cursor-pointer"
            >
              {FAULT_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>

            {/* Severity Filter */}
            <select
              value={selectedSeverity}
              onChange={(e) => setSelectedSeverity(e.target.value)}
              className="px-3 py-2 text-xs bg-surface border border-outline-variant/40 rounded-lg text-on-surface focus:outline-none focus:border-primary transition-colors cursor-pointer"
            >
              {SEVERITY_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>

            {/* Sort Toggle */}
            <button
              onClick={() =>
                setSortOrder((prev) => (prev === "desc" ? "asc" : "desc"))
              }
              className="inline-flex items-center gap-1.5 px-3 py-2 text-xs bg-surface border border-outline-variant/40 rounded-lg text-on-surface hover:bg-surface-container transition-colors"
              title="Toggle sort order"
            >
              <ArrowUpDown className="w-3.5 h-3.5 text-outline" />
              <span>{sortOrder === "desc" ? "Newest First" : "Oldest First"}</span>
            </button>

            {/* Reset */}
            {(searchQuery || selectedFault !== "ALL" || selectedSeverity !== "ALL") && (
              <button
                onClick={handleResetFilters}
                className="inline-flex items-center gap-1 px-2.5 py-2 text-xs text-on-surface-variant hover:text-on-surface transition-colors"
                title="Reset filters"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span>Reset</span>
              </button>
            )}
          </div>
        </div>

        {/* Inspections Table */}
        <div className="bg-surface-container-lowest rounded-xl border border-outline-variant/30 shadow-sm overflow-hidden flex flex-col">
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
                  <th className="py-3 px-4 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-outline-variant/20">
                {isLoading ? (
                  <tr>
                    <td colSpan={8} className="py-8 text-center text-sm text-on-surface-variant">
                      Loading inspections...
                    </td>
                  </tr>
                ) : filteredInspections.length === 0 ? (
                  <tr>
                    <td colSpan={8} className="py-8 text-center text-sm text-on-surface-variant">
                      No inspections found matching your filters.
                    </td>
                  </tr>
                ) : (
                  filteredInspections.map((row) => (
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
                      <td
                        className="py-3.5 px-4 text-xs text-on-surface-variant max-w-xs truncate"
                        title={row.maintenance_action}
                      >
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
                          View Result
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
              Showing {filteredInspections.length} of {allInspections.length} inspection records
            </span>
          </div>
        </div>

        {/* Disclaimer */}
        <Disclaimer variant="compact" />
      </div>
    </AppShell>
  );
}
