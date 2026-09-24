"use client";

import React, { useState, useEffect, use } from "react";
import Link from "next/link";
import {
  ArrowLeft,
  Camera,
  ClipboardCheck,
  AlertTriangle,
  MapPin,
  Calendar,
  Layers,
  ChevronRight,
} from "lucide-react";
import { AppShell } from "@/components/layout/AppShell";
import { SeverityBadge, UrgencyBadge } from "@/components/common/StatusBadge";
import { Disclaimer } from "@/components/common/Disclaimer";
import { MOCK_PANELS, MOCK_PANEL_001_INSPECTIONS } from "@/lib/mock-data";
import { fetchPanel, fetchPanelInspections } from "@/lib/api";
import { PanelResponse, InspectionResponse } from "@/types/inspection";
import { formatFaultName, formatInspectionDate } from "@/lib/formatters";

export default function PanelDetailsPage({
  params,
}: {
  params?: Promise<{ panelId?: string }>;
}) {
  const resolvedParams = params ? use(params) : undefined;
  const panelId = resolvedParams?.panelId || "SP-HYD-001";

  const defaultPanel =
    MOCK_PANELS.find((p) => p.panel_id.toLowerCase() === panelId.toLowerCase()) || {
      id: 1,
      panel_id: panelId,
      location: "Block A - Rooftop 1",
    };

  const [panel, setPanel] = useState<PanelResponse>(defaultPanel);
  const [inspections, setInspections] = useState<InspectionResponse[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let isMounted = true;
    async function loadData() {
      try {
        const [pData, iData] = await Promise.all([
          fetchPanel(panelId),
          fetchPanelInspections(panelId),
        ]);
        if (isMounted) {
          if (pData) {
            setPanel(pData);
          }
          if (iData?.items && iData.items.length > 0) {
            setInspections(iData.items);
          } else if (panelId === "SP-HYD-001") {
            setInspections(MOCK_PANEL_001_INSPECTIONS);
          }
        }
      } catch (err) {
        console.error("Failed to load panel details:", err);
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
  }, [panelId]);

  const latestInspection = inspections[0];

  return (
    <AppShell
      breadcrumbs={[
        { label: "Dashboard", href: "/" },
        { label: "Panel Details", href: "/panel-details" },
        { label: panel.panel_id, active: true },
      ]}
    >
      <div className="flex flex-col gap-6 max-w-5xl mx-auto w-full">
        {/* Page Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs font-semibold text-primary uppercase tracking-wider">
                Panel Details
              </span>
            </div>
            <h1 className="text-2xl font-bold text-on-surface tracking-tight">
              Panel {panel.panel_id}
            </h1>
            <p className="text-sm text-on-surface-variant mt-1 flex items-center gap-1.5">
              <MapPin className="w-3.5 h-3.5 text-outline" />
              <span>{panel.location}</span>
            </p>
          </div>

          <div className="flex items-center gap-3">
            <Link
              href="/panel-details"
              className="inline-flex items-center gap-1.5 px-3.5 py-2 bg-surface border border-outline-variant/30 hover:bg-surface-container text-on-surface text-xs font-semibold rounded-lg shadow-sm transition-colors"
            >
              <Layers className="w-3.5 h-3.5" />
              <span>All Panels</span>
            </Link>
            <Link
              href="/new-inspection"
              className="inline-flex items-center gap-1.5 px-3.5 py-2 bg-primary hover:bg-primary-container text-on-primary text-xs font-semibold rounded-lg shadow-sm transition-colors"
            >
              <Camera className="w-3.5 h-3.5" />
              <span>New Inspection</span>
            </Link>
          </div>
        </div>

        {/* Panel Overview Cards Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Total Inspections */}
          <div className="bg-surface-container-lowest p-5 rounded-xl border border-outline-variant/30 shadow-sm flex flex-col justify-between">
            <span className="text-xs font-semibold text-on-surface-variant uppercase tracking-wider">
              Total Inspections
            </span>
            <div className="mt-3">
              <div className="text-2xl font-bold text-on-surface">
                {isLoading ? "—" : inspections.length}
              </div>
              <p className="text-xs text-on-surface-variant mt-1">
                Completed visual assessments
              </p>
            </div>
          </div>

          {/* Last Inspection Date */}
          <div className="bg-surface-container-lowest p-5 rounded-xl border border-outline-variant/30 shadow-sm flex flex-col justify-between">
            <span className="text-xs font-semibold text-on-surface-variant uppercase tracking-wider">
              Last Inspection
            </span>
            <div className="mt-3">
              <div className="text-sm font-bold text-on-surface">
                {latestInspection
                  ? formatInspectionDate(latestInspection.inspection_timestamp)
                  : "No inspections yet"}
              </div>
              <p className="text-xs text-on-surface-variant mt-1">
                Most recent assessment
              </p>
            </div>
          </div>

          {/* Latest Detected Fault */}
          <div className="bg-surface-container-lowest p-5 rounded-xl border border-outline-variant/30 shadow-sm flex flex-col justify-between">
            <span className="text-xs font-semibold text-on-surface-variant uppercase tracking-wider">
              Latest Detected Fault
            </span>
            <div className="mt-3">
              <div className="text-base font-bold text-on-surface">
                {latestInspection
                  ? formatFaultName(latestInspection.predicted_class)
                  : "Clean"}
              </div>
              <p className="text-xs text-on-surface-variant mt-1">
                {latestInspection ? (
                  <span>
                    {(latestInspection.confidence * 100).toFixed(1)}% confidence
                  </span>
                ) : (
                  "No faults recorded"
                )}
              </p>
            </div>
          </div>

          {/* Latest Severity */}
          <div className="bg-surface-container-lowest p-5 rounded-xl border border-outline-variant/30 shadow-sm flex flex-col justify-between">
            <span className="text-xs font-semibold text-on-surface-variant uppercase tracking-wider">
              Latest Severity
            </span>
            <div className="mt-3">
              {latestInspection ? (
                <SeverityBadge severity={latestInspection.severity} showEstimateHint />
              ) : (
                <span className="text-xs text-on-surface-variant">None</span>
              )}
              <p className="text-xs text-on-surface-variant mt-2">
                Visual severity estimate
              </p>
            </div>
          </div>
        </div>

        {/* Latest Recommendation Banner */}
        {latestInspection && (
          <div className="bg-surface-container-lowest p-5 rounded-xl border border-outline-variant/30 shadow-sm flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <span className="text-xs font-semibold text-on-surface-variant uppercase tracking-wider block">
                Latest Recommended Action
              </span>
              <p className="text-base font-semibold text-on-surface mt-1">
                {latestInspection.maintenance_action || "Routine monitoring"}
              </p>
            </div>
            <Link
              href={`/inspection-result/${latestInspection.inspection_id}`}
              className="inline-flex items-center gap-1.5 px-4 py-2 bg-surface hover:bg-surface-container text-primary font-semibold text-xs rounded-lg border border-outline-variant/30 transition-colors shrink-0"
            >
              <span>View Latest Result</span>
              <ChevronRight className="w-3.5 h-3.5" />
            </Link>
          </div>
        )}

        {/* Inspection History for This Panel */}
        <div className="bg-surface-container-lowest rounded-xl border border-outline-variant/30 shadow-sm overflow-hidden flex flex-col">
          <div className="p-4 sm:p-5 bg-surface-container-low/40 border-b border-outline-variant/20 flex items-center justify-between">
            <div>
              <h2 className="text-base font-semibold text-on-surface">
                Inspection History
              </h2>
              <p className="text-xs text-on-surface-variant mt-0.5">
                Past diagnostic records for panel {panel.panel_id}
              </p>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-sm">
              <thead>
                <tr className="bg-surface-container-low/60 text-on-surface-variant text-xs font-semibold uppercase tracking-wider border-b border-outline-variant/20">
                  <th className="py-3 px-4">Date</th>
                  <th className="py-3 px-4">Detected Fault</th>
                  <th className="py-3 px-4 text-right">Confidence</th>
                  <th className="py-3 px-4 text-center">Severity</th>
                  <th className="py-3 px-4">Recommended Action</th>
                  <th className="py-3 px-4 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-outline-variant/20">
                {inspections.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="py-8 text-center text-sm text-on-surface-variant">
                      {isLoading
                        ? "Loading inspection records..."
                        : "No inspections recorded for this panel yet."}
                    </td>
                  </tr>
                ) : (
                  inspections.map((row) => (
                    <tr
                      key={row.inspection_id}
                      className="hover:bg-surface-container-low/50 transition-colors"
                    >
                      <td className="py-3.5 px-4 text-xs text-on-surface font-medium whitespace-nowrap">
                        {formatInspectionDate(row.inspection_timestamp)}
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
        </div>

        {/* Disclaimer */}
        <Disclaimer variant="compact" />
      </div>
    </AppShell>
  );
}
