"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { AppShell } from "@/components/layout/AppShell";
import { SeverityBadge, UrgencyBadge } from "@/components/common/StatusBadge";
import { Disclaimer } from "@/components/common/Disclaimer";
import { useToast } from "@/components/common/Toast";
import { MOCK_INSPECTIONS } from "@/lib/mock-data";
import { fetchHealth, fetchPanels, fetchInspections } from "@/lib/api";
import { InspectionResponse } from "@/types/inspection";

export default function DashboardPage() {
  const { showToast } = useToast();
  const [inspections, setInspections] = useState<InspectionResponse[]>(MOCK_INSPECTIONS);
  const [totalPanels, setTotalPanels] = useState<number>(1420);
  const [totalInspections, setTotalInspections] = useState<number>(3845);
  const [isBackendOnline, setIsBackendOnline] = useState<boolean>(false);

  useEffect(() => {
    let isMounted = true;
    async function loadDashboardData() {
      try {
        const [health, panelsData, inspectionsData] = await Promise.all([
          fetchHealth(),
          fetchPanels(),
          fetchInspections(1, 20),
        ]);

        if (isMounted) {
          const online = health.connected === true && health.status === "ok";
          setIsBackendOnline(online);

          if (panelsData?.items && panelsData.items.length > 0) {
            setTotalPanels(panelsData.total || panelsData.items.length);
          }

          if (inspectionsData?.items && inspectionsData.items.length > 0) {
            setInspections(inspectionsData.items);
            setTotalInspections(inspectionsData.total || inspectionsData.items.length);
          }
        }
      } catch {
        // Fallback to initial mock states
      }
    }

    loadDashboardData();
    return () => {
      isMounted = false;
    };
  }, []);

  const highSeverityCount = inspections.filter((i) => i.severity === "HIGH").length;
  const manualReviewCount = inspections.filter(
    (i) => i.manual_inspection_recommended
  ).length;

  return (
    <AppShell>
      <div className="flex flex-col w-full gap-space-md">
        {/* Top Control & Telemetry Bar */}
        <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-space-sm bg-surface-container-lowest p-space-md rounded shadow-sm border border-outline-variant/30">
          <div className="flex items-center gap-space-md flex-wrap">
            <div className="flex items-center gap-space-xs">
              <span
                className={`inline-block w-2.5 h-2.5 rounded-full ${
                  isBackendOnline ? "bg-secondary animate-pulse" : "bg-outline"
                }`}
              ></span>
              <span className="font-code-id text-code-id text-on-surface font-semibold">
                FACILITY ID // SEC4-ALPHA-PV
              </span>
            </div>
            <span className="text-outline-variant">|</span>
            <div className="flex items-center gap-1.5 font-label-sm text-label-sm text-on-surface">
              <span className="material-symbols-outlined text-primary text-sm">
                memory
              </span>
              <span>
                Model: <span className="font-code-id font-semibold">EfficientNet-B0 (RGB)</span>
              </span>
            </div>
            <span className="text-outline-variant">|</span>
            <div className="flex items-center gap-1.5 font-label-sm text-label-sm text-on-surface">
              <span className="material-symbols-outlined text-secondary text-sm">
                category
              </span>
              <span>
                Classes: <span className="font-code-id font-semibold">6 Fault Categories</span>
              </span>
            </div>
            <span className="text-outline-variant">|</span>
            <div className="flex items-center gap-1.5 font-label-sm text-label-sm text-on-surface">
              <span className="material-symbols-outlined text-primary text-sm">
                visibility
              </span>
              <span>
                Explainability: <span className="font-code-id font-semibold">Grad-CAM + Region Extractor</span>
              </span>
            </div>
          </div>
          <div className="flex items-center gap-space-xs text-on-surface-variant font-code-id text-code-id">
            <span
              className={`px-2 py-0.5 rounded font-label-caps text-label-caps font-semibold ${
                isBackendOnline
                  ? "bg-secondary-container text-on-secondary-container"
                  : "bg-surface-container text-on-surface-variant"
              }`}
            >
              {isBackendOnline ? "FASTAPI CONNECTED (LIVE)" : "SYNTHETIC DEMO TELEMETRY"}
            </span>
          </div>
        </div>

        {/* 4 Key KPI Blocks */}
        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-space-md">
          {/* KPI 1: Total Panels */}
          <div className="bg-surface-container-lowest p-space-md rounded shadow-sm border border-outline-variant/30 flex flex-col justify-between group hover:shadow-md transition-shadow">
            <div className="flex items-start justify-between">
              <div className="flex flex-col">
                <span className="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-wider">
                  Asset Registry
                </span>
                <span className="font-headline-sm text-headline-sm text-on-surface mt-0.5">
                  Total Panels
                </span>
              </div>
              <div className="p-2 bg-surface-container rounded text-primary">
                <span className="material-symbols-outlined text-lg">grid_view</span>
              </div>
            </div>
            <div className="mt-space-md flex flex-col">
              <div className="flex items-baseline gap-2">
                <span className="font-headline-xl text-headline-xl text-on-surface tracking-tight">
                  {totalPanels.toLocaleString()}
                </span>
                <span className="font-code-id text-code-id text-secondary font-semibold">
                  {isBackendOnline ? "LIVE" : "+1.3%"}
                </span>
              </div>
              <div className="mt-1 flex items-center gap-1 text-on-surface-variant font-label-sm text-label-sm">
                <span className="material-symbols-outlined text-secondary text-sm">
                  trending_up
                </span>
                <span>
                  {isBackendOnline
                    ? `${totalPanels} registered in database`
                    : "+18 registered this quarter"}
                </span>
              </div>
            </div>
          </div>

          {/* KPI 2: Total Inspections */}
          <div className="bg-surface-container-lowest p-space-md rounded shadow-sm border border-outline-variant/30 flex flex-col justify-between group hover:shadow-md transition-shadow">
            <div className="flex items-start justify-between">
              <div className="flex flex-col">
                <span className="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-wider">
                  Inspection Throughput
                </span>
                <span className="font-headline-sm text-headline-sm text-on-surface mt-0.5">
                  Total Inspections
                </span>
              </div>
              <div className="p-2 bg-surface-container rounded text-primary">
                <span className="material-symbols-outlined text-lg">fact_check</span>
              </div>
            </div>
            <div className="mt-space-md flex flex-col">
              <div className="flex items-baseline gap-2">
                <span className="font-headline-xl text-headline-xl text-on-surface tracking-tight">
                  {totalInspections.toLocaleString()}
                </span>
                <span className="font-code-id text-code-id text-secondary font-semibold">
                  {isBackendOnline ? "LIVE" : "98.4%"}
                </span>
              </div>
              <div className="mt-1 flex items-center gap-1 text-on-surface-variant font-label-sm text-label-sm">
                <span className="material-symbols-outlined text-secondary text-sm">
                  verified
                </span>
                <span>Processed via EfficientNet-B0</span>
              </div>
            </div>
          </div>

          {/* KPI 3: High Visual Severity Cases */}
          <div className="bg-surface-container-lowest p-space-md rounded shadow-sm border border-outline-variant/30 flex flex-col justify-between group hover:shadow-md transition-shadow">
            <div className="flex items-start justify-between">
              <div className="flex flex-col">
                <span className="font-label-caps text-label-caps text-error uppercase tracking-wider">
                  Visual Anomaly Triage
                </span>
                <span className="font-headline-sm text-headline-sm text-on-surface mt-0.5">
                  High Visual Severity
                </span>
              </div>
              <div className="p-2 bg-error-container rounded text-on-error-container">
                <span className="material-symbols-outlined text-lg">warning</span>
              </div>
            </div>
            <div className="mt-space-md flex flex-col">
              <div className="flex items-baseline gap-2">
                <span className="font-headline-xl text-headline-xl text-error tracking-tight">
                  {highSeverityCount}
                </span>
                <span className="px-1.5 py-0.5 bg-error-container text-on-error-container rounded font-label-caps text-label-caps font-semibold">
                  AI ESTIMATE
                </span>
              </div>
              <div className="mt-1 flex items-center gap-1 text-on-error-container font-label-sm text-label-sm">
                <span className="material-symbols-outlined text-error text-sm">
                  emergency
                </span>
                <span className="font-semibold text-error">
                  {highSeverityCount} flagged for physical inspection
                </span>
              </div>
            </div>
          </div>

          {/* KPI 4: Manual Review Needed */}
          <div className="bg-surface-container-lowest p-space-md rounded shadow-sm border border-outline-variant/30 flex flex-col justify-between group hover:shadow-md transition-shadow">
            <div className="flex items-start justify-between">
              <div className="flex flex-col">
                <span className="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-wider">
                  Confidence Threshold
                </span>
                <span className="font-headline-sm text-headline-sm text-on-surface mt-0.5">
                  Manual Review Needed
                </span>
              </div>
              <div className="p-2 bg-surface-container-high rounded text-on-surface">
                <span className="material-symbols-outlined text-lg">engineering</span>
              </div>
            </div>
            <div className="mt-space-md flex flex-col">
              <div className="flex items-baseline gap-2">
                <span className="font-headline-xl text-headline-xl text-on-surface tracking-tight">
                  {manualReviewCount}
                </span>
                <span className="px-1.5 py-0.5 bg-surface-container-high text-on-surface-variant rounded font-label-caps text-label-caps font-semibold">
                  FLAGGED
                </span>
              </div>
              <div className="mt-1 flex items-center gap-1 text-on-surface-variant font-label-sm text-label-sm">
                <span className="material-symbols-outlined text-outline text-sm">tune</span>
                <span>AI confidence threshold &lt; 70%</span>
              </div>
            </div>
          </div>
        </div>

        {/* Quick Action Banner */}
        <div className="bg-surface-container-lowest p-space-md rounded shadow-sm border border-outline-variant/30 flex flex-col md:flex-row items-start md:items-center justify-between gap-space-md">
          <div className="flex items-center gap-space-md">
            <div className="h-10 w-10 rounded bg-primary/10 flex items-center justify-center text-primary shrink-0">
              <span className="material-symbols-outlined">photo_camera</span>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-headline-sm text-headline-sm text-on-surface font-semibold">
                  Optical RGB Solar Defect Inspection
                </span>
                <span className="font-label-caps text-label-caps px-1.5 py-0.5 rounded bg-surface-container-high text-primary font-semibold">
                  EFFICIENTNET-B0 + GRAD-CAM
                </span>
              </div>
              <p className="font-body-sm text-body-sm text-on-surface-variant mt-0.5">
                Upload RGB solar panel imagery to classify faults across 6 target classes,
                calculate visual region area %, and receive actionable maintenance recommendations.
              </p>
            </div>
          </div>
          <div className="flex items-center gap-space-sm w-full md:w-auto justify-end">
            <button
              onClick={() =>
                showToast("Exporting inspection records to CSV summary...")
              }
              className="flex items-center gap-1.5 px-3 py-1.5 bg-surface-container-low hover:bg-surface-container text-on-surface font-body-md text-body-md rounded transition-colors shadow-sm border border-outline-variant/30"
            >
              <span className="material-symbols-outlined text-base text-outline">
                file_download
              </span>
              <span>Export Summary CSV</span>
            </button>
            <Link
              href="/new-inspection"
              className="flex items-center gap-1.5 px-3.5 py-1.5 bg-primary hover:bg-primary-container text-on-primary font-body-md text-body-md font-semibold rounded transition-colors shadow-sm"
            >
              <span className="material-symbols-outlined text-base">play_arrow</span>
              <span>Run Inspection</span>
            </Link>
          </div>
        </div>

        {/* Main Grid: Data Table + Auxiliary Widgets */}
        <div className="grid grid-cols-1 xl:grid-cols-12 gap-space-md items-start">
          {/* Recent Inspections Ledger (9 cols) */}
          <div className="xl:col-span-9 bg-surface-container-lowest rounded shadow-sm border border-outline-variant/30 overflow-hidden flex flex-col">
            {/* Table Header Bar */}
            <div className="p-space-md bg-surface-container-low flex flex-col sm:flex-row items-start sm:items-center justify-between gap-space-sm border-b border-outline-variant/20">
              <div className="flex items-center gap-space-sm">
                <span className="font-headline-sm text-headline-sm text-on-surface font-semibold">
                  Recent Inspections Ledger
                </span>
                <span className="px-2 py-0.5 bg-surface-container rounded text-on-surface-variant font-code-id text-code-id">
                  {inspections.length} items {isBackendOnline ? "(FastAPI Live)" : "(Synthetic Demo)"}
                </span>
              </div>
              <div className="flex items-center gap-space-xs">
                <button
                  onClick={() => showToast("Filter drawer opened")}
                  className="px-2.5 py-1 bg-surface-container-lowest text-on-surface hover:bg-surface-container rounded font-label-sm text-label-sm shadow-sm flex items-center gap-1 border border-outline-variant/30"
                >
                  <span className="material-symbols-outlined text-sm">
                    filter_alt
                  </span>{" "}
                  Filter
                </button>
                <button
                  onClick={() => showToast("Sorted by timestamp descending")}
                  className="px-2.5 py-1 bg-surface-container-lowest text-on-surface hover:bg-surface-container rounded font-label-sm text-label-sm shadow-sm flex items-center gap-1 border border-outline-variant/30"
                >
                  <span className="material-symbols-outlined text-sm">
                    swap_vert
                  </span>{" "}
                  Sort
                </button>
              </div>
            </div>

            {/* Scrollable Data Table Container */}
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="bg-surface-container-low/70 text-on-surface-variant font-label-caps text-label-caps uppercase border-b border-outline-variant/20">
                    <th className="py-2.5 px-space-md">Panel ID</th>
                    <th className="py-2.5 px-space-md">Location</th>
                    <th className="py-2.5 px-space-md">Predicted Fault</th>
                    <th className="py-2.5 px-space-md text-right">Confidence</th>
                    <th className="py-2.5 px-space-md text-center">Visual Severity (Est.)</th>
                    <th className="py-2.5 px-space-md text-center">Urgency</th>
                    <th className="py-2.5 px-space-md">Timestamp</th>
                    <th className="py-2.5 px-space-md text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="font-body-md text-body-md text-on-surface divide-y divide-outline-variant/20">
                  {inspections.map((row) => (
                    <tr
                      key={row.inspection_id}
                      className="hover:bg-surface-container-low transition-colors group cursor-pointer"
                      onClick={() =>
                        showToast(`Dossier loaded for: ${row.panel_id}`)
                      }
                    >
                      <td className="py-3 px-space-md font-code-id text-code-id font-semibold text-primary">
                        <Link
                          href={`/panel-details/${row.panel_id}`}
                          className="hover:underline"
                          onClick={(e) => e.stopPropagation()}
                        >
                          {row.panel_id}
                        </Link>
                      </td>
                      <td className="py-3 px-space-md font-body-sm text-body-sm text-on-surface-variant">
                        {row.location}
                      </td>
                      <td className="py-3 px-space-md">
                        <span className="inline-flex items-center gap-1.5 font-medium">
                          <span
                            className={`w-2 h-2 rounded-full ${
                              row.predicted_class === "Electrical-damage"
                                ? "bg-error"
                                : row.predicted_class === "Physical-damage"
                                ? "bg-tertiary"
                                : row.predicted_class === "Bird-drop"
                                ? "bg-outline"
                                : row.predicted_class === "Clean"
                                ? "bg-secondary"
                                : "bg-outline-variant"
                            }`}
                          ></span>
                          <span
                            className={
                              row.predicted_class === "Electrical-damage"
                                ? "text-error"
                                : row.predicted_class === "Clean"
                                ? "text-secondary"
                                : "text-on-surface"
                            }
                          >
                            {row.predicted_class}
                          </span>
                        </span>
                      </td>
                      <td className="py-3 px-space-md text-right font-code-metric text-code-id text-on-surface font-semibold">
                        {(row.confidence * 100).toFixed(1)}%
                      </td>
                      <td className="py-3 px-space-md text-center">
                        <SeverityBadge severity={row.severity} />
                      </td>
                      <td className="py-3 px-space-md text-center">
                        <UrgencyBadge urgency={row.urgency} />
                      </td>
                      <td className="py-3 px-space-md font-code-id text-code-id text-on-surface-variant">
                        {row.inspection_timestamp.includes("T")
                          ? new Date(row.inspection_timestamp).toLocaleDateString(
                              "en-US",
                              { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }
                            )
                          : row.inspection_timestamp}
                      </td>
                      <td className="py-3 px-space-md text-right">
                        <Link
                          href={`/inspection-result/${row.inspection_id}`}
                          onClick={(e) => e.stopPropagation()}
                          className="inline-block px-2.5 py-1 bg-surface-container hover:bg-primary hover:text-on-primary text-on-surface rounded font-label-sm text-label-sm transition-colors shadow-sm"
                        >
                          View Result
                        </Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Table Footer */}
            <div className="p-space-sm bg-surface-container-low flex items-center justify-between font-label-sm text-label-sm text-on-surface-variant border-t border-outline-variant/20">
              <span>Displaying {inspections.length} records {isBackendOnline ? "(Live Database)" : "(Synthetic Demo Dataset)"}</span>
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

          {/* Right Auxiliary Side Widgets (3 cols) */}
          <div className="xl:col-span-3 flex flex-col gap-space-md">
            {/* Fault Distribution Breakdown */}
            <div className="bg-surface-container-lowest p-space-md rounded shadow-sm border border-outline-variant/30 flex flex-col">
              <div className="flex items-center justify-between mb-space-sm">
                <span className="font-headline-sm text-headline-sm text-on-surface font-semibold">
                  Fault Distribution
                </span>
                <span className="font-label-caps text-label-caps text-on-surface-variant uppercase">
                  DEMO N=1,420
                </span>
              </div>
              <p className="font-body-sm text-body-sm text-on-surface-variant mb-space-md">
                Distribution breakdown across the 6 supported fault classification categories.
              </p>

              {/* Segmented Progress Bar */}
              <div className="w-full h-3 bg-surface-container rounded-full overflow-hidden flex mb-space-md">
                <div className="bg-secondary" style={{ width: "48%" }} title="Clean: 48%"></div>
                <div className="bg-surface-variant" style={{ width: "24%" }} title="Dusty: 24%"></div>
                <div className="bg-outline" style={{ width: "12%" }} title="Bird-drop: 12%"></div>
                <div className="bg-error" style={{ width: "9%" }} title="Electrical-damage: 9%"></div>
                <div className="bg-tertiary-container" style={{ width: "5%" }} title="Physical-damage: 5%"></div>
                <div className="bg-primary-fixed-dim" style={{ width: "2%" }} title="Snow-Covered: 2%"></div>
              </div>

              {/* Distribution Ledger */}
              <div className="flex flex-col gap-2 font-body-sm text-body-sm">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-sm bg-secondary"></span>
                    <span className="text-on-surface">Clean</span>
                  </div>
                  <span className="font-code-id text-code-id font-semibold text-on-surface">
                    48% (682)
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-sm bg-surface-variant"></span>
                    <span className="text-on-surface">Dusty</span>
                  </div>
                  <span className="font-code-id text-code-id font-semibold text-on-surface">
                    24% (341)
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-sm bg-outline"></span>
                    <span className="text-on-surface">Bird-drop</span>
                  </div>
                  <span className="font-code-id text-code-id font-semibold text-on-surface">
                    12% (170)
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-sm bg-error"></span>
                    <span className="text-error font-medium">Electrical-damage</span>
                  </div>
                  <span className="font-code-id text-code-id font-semibold text-error">
                    9% (128)
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-sm bg-tertiary-container"></span>
                    <span className="text-on-surface">Physical-damage</span>
                  </div>
                  <span className="font-code-id text-code-id font-semibold text-on-surface">
                    5% (71)
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-sm bg-primary-fixed-dim"></span>
                    <span className="text-on-surface">Snow-Covered</span>
                  </div>
                  <span className="font-code-id text-code-id font-semibold text-on-surface">
                    2% (28)
                  </span>
                </div>
              </div>
            </div>

            {/* Active Defect Focus Preview Card */}
            <div className="bg-surface-container-lowest p-space-md rounded shadow-sm border border-outline-variant/30 flex flex-col">
              <div className="flex items-center justify-between mb-space-sm">
                <div className="flex items-center gap-1.5">
                  <span className="material-symbols-outlined text-primary text-base">
                    center_focus_strong
                  </span>
                  <span className="font-headline-sm text-headline-sm text-on-surface font-semibold">
                    Active Defect Focus
                  </span>
                </div>
                <span className="font-label-caps text-label-caps text-on-surface-variant bg-surface-container px-1.5 py-0.5 rounded font-semibold">
                  {isBackendOnline ? "LIVE TELEMETRY" : "DEMO SAMPLE"}
                </span>
              </div>
              <div className="relative w-full h-36 bg-surface-container-high rounded overflow-hidden mb-space-sm">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  className="w-full h-full object-cover"
                  alt="RGB solar panel showing localized discoloration anomaly"
                  src="https://lh3.googleusercontent.com/aida-public/AB6AXuClu9_LnwC1NQr9D9IL_YRk0G3iChTCUSzuRi3UCDO9Dfv9QtZq7gHbu38Hoj6g3gEY09fG0Ar6xS_--_c4k0lyAzGNlr-OZ7xtDPysKf1efYN1wCU_eRp7LPqcGGHmcVF1KlOj3utNLMbe_3pj1twTjdKsle077yF2JW57TmLovR6Ekw_pZxm5xjWY3geAATZ9R3srg_HXr9DJ2FDYxqU-I8b0YLbyUaeEHemeOf3hLeXt3WEJLAq-KA"
                />
                <div className="absolute top-2 left-2 px-1.5 py-0.5 bg-surface-container-lowest/90 backdrop-blur rounded font-code-id text-code-id font-semibold text-on-surface">
                  TARGET: {inspections[0]?.panel_id || "SP-HYD-001"}
                </div>
                <div className="absolute bottom-2 right-2 px-1.5 py-0.5 bg-error text-on-error rounded font-label-caps text-label-caps font-semibold">
                  {(inspections[0]?.predicted_class || "Electrical-damage").toUpperCase()} DETECTED
                </div>
              </div>
              <div className="flex flex-col gap-1 text-on-surface-variant font-label-sm text-label-sm">
                <div className="flex items-center justify-between">
                  <span>Visual Region Area:</span>
                  <strong className="text-on-surface font-code-id">
                    {inspections[0]?.visual_region_area_percent ?? 14.8}% of surface
                  </strong>
                </div>
                <div className="flex items-center justify-between">
                  <span>AI Confidence:</span>
                  <strong className="text-secondary font-code-id">
                    {inspections[0]?.confidence ? (inspections[0].confidence * 100).toFixed(1) : "96.8"}%
                  </strong>
                </div>
                <p className="text-[10px] text-on-surface-variant italic mt-1 text-center">
                  Visual region estimation only — NOT an exact defect boundary.
                </p>
              </div>
            </div>

            {/* Endpoint Readiness Widget */}
            <div className="bg-surface-container-lowest p-space-md rounded shadow-sm border border-outline-variant/30 flex flex-col">
              <div className="flex items-center justify-between mb-space-xs">
                <span className="font-headline-sm text-headline-sm text-on-surface font-semibold">
                  Endpoint Readiness
                </span>
                <span
                  className={`font-label-caps text-label-caps px-1.5 py-0.5 rounded font-semibold ${
                    isBackendOnline
                      ? "bg-secondary-container text-on-secondary-container"
                      : "bg-surface-container-high text-on-surface-variant"
                  }`}
                >
                  {isBackendOnline ? "FASTAPI READY" : "OFFLINE FALLBACK"}
                </span>
              </div>
              <span className="font-body-sm text-body-sm text-on-surface-variant mb-space-sm">
                Microservice link status
              </span>
              <div className="flex flex-col gap-2 font-code-id text-code-id">
                <div className="flex items-center justify-between p-2 bg-surface-container-low rounded border border-outline-variant/20">
                  <div className="flex items-center gap-1.5 truncate">
                    <span className="text-primary font-bold">GET</span>
                    <span className="text-on-surface truncate">/api/health</span>
                  </div>
                  <span
                    className={`px-2 py-0.5 rounded font-semibold text-xs whitespace-nowrap ${
                      isBackendOnline
                        ? "bg-secondary-container text-on-secondary-container"
                        : "bg-surface-container-high text-on-surface-variant"
                    }`}
                  >
                    {isBackendOnline ? "200 OK" : "OFFLINE"}
                  </span>
                </div>
                <div className="flex items-center justify-between p-2 bg-surface-container-low rounded border border-outline-variant/20">
                  <div className="flex items-center gap-1.5 truncate">
                    <span className="text-primary font-bold">GET</span>
                    <span className="text-on-surface truncate">/api/inspections</span>
                  </div>
                  <span
                    className={`px-2 py-0.5 rounded font-semibold text-xs whitespace-nowrap ${
                      isBackendOnline
                        ? "bg-secondary-container text-on-secondary-container"
                        : "bg-surface-container-high text-on-surface-variant"
                    }`}
                  >
                    {isBackendOnline ? "200 OK" : "OFFLINE"}
                  </span>
                </div>
                <div className="flex items-center justify-between p-2 bg-surface-container-low rounded border border-outline-variant/20">
                  <div className="flex items-center gap-1.5 truncate">
                    <span className="text-primary font-bold">GET</span>
                    <span className="text-on-surface truncate">/api/panels</span>
                  </div>
                  <span
                    className={`px-2 py-0.5 rounded font-semibold text-xs whitespace-nowrap ${
                      isBackendOnline
                        ? "bg-secondary-container text-on-secondary-container"
                        : "bg-surface-container-high text-on-surface-variant"
                    }`}
                  >
                    {isBackendOnline ? "200 OK" : "OFFLINE"}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Mandatory Statutory Disclaimer */}
        <Disclaimer />
      </div>
    </AppShell>
  );
}
