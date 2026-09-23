"use client";

import React, { useState, useEffect, use } from "react";
import Link from "next/link";
import { AppShell } from "@/components/layout/AppShell";
import { SeverityBadge, UrgencyBadge } from "@/components/common/StatusBadge";
import { Disclaimer } from "@/components/common/Disclaimer";
import { useToast } from "@/components/common/Toast";
import { MOCK_PANELS, MOCK_PANEL_001_INSPECTIONS } from "@/lib/mock-data";
import { fetchPanel, fetchPanelInspections } from "@/lib/api";
import { PanelResponse, InspectionResponse } from "@/types/inspection";

export default function PanelDetailsPage({
  params,
}: {
  params?: Promise<{ panelId?: string }>;
}) {
  const resolvedParams = params ? use(params) : undefined;
  const panelId = resolvedParams?.panelId || "SP-HYD-001";

  const { showToast } = useToast();
  const [activeLayer, setActiveLayer] = useState<"GRAD_CAM" | "RGB">("GRAD_CAM");

  // Fallback initial state
  const defaultPanel =
    MOCK_PANELS.find((p) => p.panel_id.toLowerCase() === panelId.toLowerCase()) ||
    MOCK_PANELS[0];

  const [panel, setPanel] = useState<PanelResponse>(defaultPanel);
  const [inspections, setInspections] = useState<InspectionResponse[]>(
    panelId === "SP-HYD-001" ? MOCK_PANEL_001_INSPECTIONS : []
  );

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
          }
        }
      } catch {
        // Fallback to default state
      }
    }
    loadData();
    return () => {
      isMounted = false;
    };
  }, [panelId]);

  return (
    <AppShell
      breadcrumbs={[
        { label: "Asset", href: "/" },
        { label: "Solar Farm Alpha", href: "/" },
        { label: "SEC-4", href: "/" },
        { label: `Module ${panel.panel_id}`, active: true },
      ]}
    >
      <div className="flex flex-col w-full gap-space-md">
        {/* Module Header Bar */}
        <section className="bg-surface-container-lowest p-space-md rounded-lg shadow-sm border border-outline-variant/30 flex flex-col md:flex-row md:items-center md:justify-between gap-space-sm">
          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-space-xs font-label-caps text-label-caps text-on-surface-variant">
              <span>ASSET TOPOLOGY</span>
              <span>/</span>
              <span className="text-primary font-semibold">MODULE SPECIFICATION</span>
            </div>
            <div className="flex items-center gap-3 flex-wrap">
              <h1 className="font-headline-lg text-headline-lg text-on-surface tracking-tight font-bold">
                PV Module:{" "}
                <span className="font-code-metric text-primary">
                  {panel.panel_id}
                </span>
              </h1>
              <span className="px-2 py-0.5 rounded bg-secondary-container text-on-secondary-container font-label-caps text-label-caps font-semibold">
                OPERATIONAL (DERATED)
              </span>
            </div>
            <div className="flex items-center gap-4 text-on-surface-variant font-label-sm text-label-sm mt-0.5 flex-wrap">
              <span>
                Location:{" "}
                <strong className="text-on-surface">{panel.location}</strong>
              </span>
              <span>•</span>
              <span>
                Commissioning:{" "}
                <strong className="text-on-surface">
                  {panel.installation_date || "Jan 15, 2022"}
                </strong>
              </span>
              <span>•</span>
              <span>
                Array Position:{" "}
                <strong className="text-on-surface">
                  {panel.rack_coordinates || "Rack 04 · Row 12"}
                </strong>
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2 self-start md:self-auto">
            <button
              onClick={() => showToast("Exporting comprehensive module dossier...")}
              className="inline-flex items-center gap-1.5 px-3 py-2 rounded bg-surface-container hover:bg-surface-container-high text-on-surface font-body-sm text-body-sm transition-colors shadow-sm border border-outline-variant/30"
            >
              <span className="material-symbols-outlined text-base">download</span>
              Export Dossier
            </button>
            <Link
              href="/new-inspection"
              className="inline-flex items-center gap-1.5 px-3 py-2 rounded bg-primary hover:bg-primary-container text-on-primary font-headline-sm text-headline-sm transition-all shadow-sm"
            >
              <span className="material-symbols-outlined text-base">
                nest_cam_floodlight
              </span>
              Initiate New Scan
            </Link>
          </div>
        </section>

        {/* Top-Tier Critical Telemetry Bento Grid (5 Cards) */}
        <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-space-sm">
          {/* Metric 1: Total Scans */}
          <div className="bg-surface-container-lowest p-space-md rounded shadow-sm border border-outline-variant/30 flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <span className="font-label-caps text-label-caps uppercase text-on-surface-variant tracking-wider">
                Total Scans
              </span>
              <span className="material-symbols-outlined text-primary text-lg">
                history_toggle_off
              </span>
            </div>
            <div className="mt-2">
              <div className="font-code-metric text-code-metric text-on-surface">
                {inspections.length > 0 ? inspections.length : (panel.total_scans || 0)}{" "}
                <span className="text-xs font-normal text-on-surface-variant">
                  Runs
                </span>
              </div>
              <p className="font-label-sm text-label-sm text-on-surface-variant mt-0.5">
                {panel.created_at ? `Registered ${new Date(panel.created_at).toLocaleDateString()}` : "Operational registry record"}
              </p>
            </div>
          </div>

          {/* Metric 2: Detected Fault */}
          <div className="bg-surface-container-lowest p-space-md rounded shadow-sm border border-outline-variant/30 flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <span className="font-label-caps text-label-caps uppercase text-on-surface-variant tracking-wider">
                Latest Fault
              </span>
              <span className="material-symbols-outlined text-error text-lg">
                bolt
              </span>
            </div>
            <div className="mt-2">
              <div className="font-headline-sm text-headline-sm text-error truncate font-semibold">
                {inspections[0]?.predicted_class || panel.latest_fault || "Clean"}
              </div>
              <p className="font-code-id text-code-id text-on-surface-variant mt-0.5 truncate">
                AI Visual Classification · 6 Classes
              </p>
            </div>
          </div>

          {/* Metric 3: Latest Severity */}
          <div className="bg-surface-container-lowest p-space-md rounded shadow-sm border border-outline-variant/30 flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <span className="font-label-caps text-label-caps uppercase text-on-surface-variant tracking-wider">
                Severity
              </span>
              <span className="h-2 w-2 rounded-full bg-error animate-pulse"></span>
            </div>
            <div className="mt-2">
              <div className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded bg-error-container text-on-error-container font-label-caps text-label-caps font-bold tracking-wider">
                <span className="material-symbols-outlined text-xs">warning</span>
                {inspections[0]?.severity || panel.latest_severity || "LOW"}
              </div>
              <p className="font-label-sm text-label-sm text-on-surface-variant mt-1.5">
                AI-Assisted Visual Severity Estimate
              </p>
            </div>
          </div>

          {/* Metric 4: Latest Urgency */}
          <div className="bg-surface-container-lowest p-space-md rounded shadow-sm border border-outline-variant/30 flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <span className="font-label-caps text-label-caps uppercase text-on-surface-variant tracking-wider">
                Triage Status
              </span>
              <span className="material-symbols-outlined text-error text-lg">
                notification_important
              </span>
            </div>
            <div className="mt-2">
              <div className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-surface-container-highest text-error font-label-caps text-label-caps font-bold">
                {inspections[0]?.urgency || panel.latest_urgency || "ROUTINE"}
              </div>
              <p className="font-label-sm text-label-sm text-on-surface-variant mt-1.5">
                Escalation SLA: &lt; 24 Hours
              </p>
            </div>
          </div>

          {/* Metric 5: Last Scan Timestamp */}
          <div className="bg-surface-container-lowest p-space-md rounded shadow-sm border border-outline-variant/30 flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <span className="font-label-caps text-label-caps uppercase text-on-surface-variant tracking-wider">
                Last Inspection
              </span>
              <span className="material-symbols-outlined text-secondary text-lg">
                schedule
              </span>
            </div>
            <div className="mt-2">
              <div className="font-code-id text-code-id text-on-surface font-semibold">
                {inspections[0]?.inspection_timestamp
                  ? (inspections[0].inspection_timestamp.includes("T")
                      ? new Date(inspections[0].inspection_timestamp).toLocaleDateString("en-US", {
                          month: "short",
                          day: "numeric",
                          year: "numeric",
                        })
                      : inspections[0].inspection_timestamp)
                  : panel.last_inspected || "N/A"}
              </div>
              <p className="font-label-sm text-label-sm text-on-surface-variant mt-0.5">
                {inspections[0]?.inspection_timestamp && inspections[0].inspection_timestamp.includes("T")
                  ? `${new Date(inspections[0].inspection_timestamp).toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" })} UTC`
                  : "Inspection Record"}
              </p>
            </div>
          </div>
        </section>

        {/* Middle Tier: Dual Viewport Architecture (5 cols vs 7 cols) */}
        <div className="grid grid-cols-1 xl:grid-cols-12 gap-space-md">
          {/* Section A: Physical Asset Specifications & Health Telemetry (5 cols) */}
          <div className="xl:col-span-5 flex flex-col gap-space-md">
            <div className="bg-surface-container-lowest p-space-md rounded shadow-sm border border-outline-variant/30 flex flex-col justify-between h-full">
              <div>
                <div className="flex items-center justify-between pb-space-sm mb-space-md bg-surface-container-low px-2 py-1.5 rounded border border-outline-variant/20">
                  <div className="flex items-center gap-1.5">
                    <span className="material-symbols-outlined text-primary text-base">
                      tune
                    </span>
                    <h2 className="font-headline-sm text-headline-sm text-on-surface font-semibold">
                      Asset Specifications
                    </h2>
                  </div>
                  <span className="font-label-caps text-label-caps text-on-surface-variant">
                    REF: IEC-61215
                  </span>
                </div>

                {/* Specification Rows */}
                <div className="flex flex-col gap-2">
                  <div className="flex items-center justify-between py-1 bg-surface-container-low/50 px-2 rounded border border-outline-variant/10">
                    <span className="font-body-md text-body-md text-on-surface-variant">
                      Installation Date
                    </span>
                    <span className="font-code-id text-code-id text-on-surface font-medium">
                      {panel.installation_date || "Jan 15, 2022"}
                    </span>
                  </div>
                  <div className="flex items-center justify-between py-1 bg-surface px-2 rounded border border-outline-variant/10">
                    <span className="font-body-md text-body-md text-on-surface-variant">
                      Rated Peak Capacity
                    </span>
                    <span className="font-code-id text-code-id text-on-surface font-semibold">
                      {panel.rated_power_wp || 450} Wp
                    </span>
                  </div>
                  <div className="flex items-center justify-between py-1 bg-surface-container-low/50 px-2 rounded border border-outline-variant/10">
                    <span className="font-body-md text-body-md text-on-surface-variant">
                      Inverter Association
                    </span>
                    <div className="flex items-center gap-1">
                      <span className="material-symbols-outlined text-xs text-primary">
                        developer_board
                      </span>
                      <span className="font-code-id text-code-id text-primary font-semibold">
                        {panel.inverter_string || "String INV-04-A"}
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between py-1 bg-surface px-2 rounded border border-outline-variant/10">
                    <span className="font-body-md text-body-md text-on-surface-variant">
                      Azimuth &amp; Tilt
                    </span>
                    <span className="font-code-id text-code-id text-on-surface">
                      {panel.azimuth_tilt || "28° South-Facing (180° S)"}
                    </span>
                  </div>
                  <div className="flex items-center justify-between py-1 bg-surface-container-low/50 px-2 rounded border border-outline-variant/10">
                    <span className="font-body-md text-body-md text-on-surface-variant">
                      Module Technology
                    </span>
                    <span className="font-code-id text-code-id text-on-surface">
                      {panel.technology || "Monocrystalline N-Type"}
                    </span>
                  </div>
                  <div className="flex items-center justify-between py-1 bg-surface px-2 rounded border border-outline-variant/10">
                    <span className="font-body-md text-body-md text-on-surface-variant">
                      Array Coordinates
                    </span>
                    <span className="font-code-id text-code-id text-on-surface">
                      {panel.rack_coordinates || "Rack 04 · Row 12 · Pos 03"}
                    </span>
                  </div>
                </div>
              </div>

              {/* Health Metric Visual Meter */}
              <div className="mt-space-md pt-space-sm bg-surface-container-low p-space-sm rounded border border-outline-variant/20">
                <div className="flex items-center justify-between mb-1.5">
                  <div className="flex items-center gap-1.5">
                    <span className="font-label-caps text-label-caps uppercase text-on-surface-variant">
                      Operational Health Index
                    </span>
                    <span
                      className="material-symbols-outlined text-xs text-error"
                      title="Degraded rating caused by hotspot anomaly"
                    >
                      error
                    </span>
                  </div>
                  <span className="font-code-metric text-code-metric text-error font-semibold">
                    {panel.health_index || 71}%
                  </span>
                </div>
                <div className="w-full bg-surface-container-highest h-2 rounded overflow-hidden flex">
                  <div
                    className="bg-error h-full rounded-l"
                    style={{ width: `${panel.health_index || 71}%` }}
                  ></div>
                  <div
                    className="bg-error-container/60 h-full"
                    style={{ width: `${100 - (panel.health_index || 71)}%` }}
                  ></div>
                </div>
                <p className="font-label-sm text-label-sm text-on-surface-variant mt-2 flex items-center gap-1">
                  <span className="material-symbols-outlined text-xs text-error">
                    trending_down
                  </span>
                  Visual health index based on AI-assisted classification and detected surface anomalies.
                </p>
              </div>
            </div>
          </div>

          {/* Diagnostic Optical RGB & Grad-CAM Attention Viewport (7 cols) */}
          <div className="xl:col-span-7 bg-surface-container-lowest p-space-md rounded shadow-sm border border-outline-variant/30 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between pb-space-sm mb-space-sm bg-surface-container-low px-2 py-1.5 rounded border border-outline-variant/20">
                <div className="flex items-center gap-2">
                  <span className="material-symbols-outlined text-primary text-base">
                    visibility
                  </span>
                  <span className="font-headline-sm text-headline-sm text-on-surface font-semibold">
                    Optical RGB &amp; AI Attention Viewport
                  </span>
                  <span className="font-label-caps text-label-caps px-1.5 py-0.5 bg-primary/10 text-primary rounded font-bold">
                    GRAD-CAM SALIENCY
                  </span>
                </div>
                <div className="flex items-center gap-2 font-code-id text-code-id text-on-surface-variant">
                  <span>EfficientNet-B0</span>
                  <span>·</span>
                  <span>Input: 224×224</span>
                </div>
              </div>

              {/* Viewport Display Container */}
              <div className="relative w-full h-64 sm:h-72 rounded overflow-hidden bg-inverse-surface flex items-center justify-center">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  className="w-full h-full object-cover"
                  alt="Optical RGB solar panel inspection capture"
                  src={
                    activeLayer === "GRAD_CAM"
                      ? "https://lh3.googleusercontent.com/aida-public/AB6AXuBTQZK5eOuP_wR4SuHW_doQ7vTB7nX4yRLqcoeDY1ckYJVcPC-lFhkePC60bbRNCMfAkRcLwwwTzjfJD2h8rPjd4snm5WjRPGi9jsPZXBE0sWKqNKcwzFrZsHym--C1KRO1s8kHDO67NNDVv2yi_lUXzv6gqsiZvWlvKRxOI37GpzrIxs2o13zX2-NAMbEueokGBIb9jM6IGttmHBdUzBBws3lFlSdFjWJ2DXLI2WW25MyctUCm24zrzQ"
                      : "https://lh3.googleusercontent.com/aida-public/AB6AXuClu9_LnwC1NQr9D9IL_YRk0G3iChTCUSzuRi3UCDO9Dfv9QtZq7gHbu38Hoj6g3gEY09fG0Ar6xS_--_c4k0lyAzGNlr-OZ7xtDPysKf1efYN1wCU_eRp7LPqcGGHmcVF1KlOj3utNLMbe_3pj1twTjdKsle077yF2JW57TmLovR6Ekw_pZxm5xjWY3geAATZ9R3srg_HXr9DJ2FDYxqU-I8b0YLbyUaeEHemeOf3hLeXt3WEJLAq-KA"
                  }
                />
                {/* HUD Overlay */}
                <div className="absolute inset-0 bg-gradient-to-t from-inverse-surface/80 via-transparent to-inverse-surface/30 p-space-md flex flex-col justify-between pointer-events-none">
                  <div className="flex items-center justify-between text-on-primary">
                    <div className="flex items-center gap-1.5 font-code-id text-code-id bg-inverse-surface/90 px-2 py-1 rounded backdrop-blur-sm">
                      <span className="inline-block w-2 h-2 rounded-full bg-secondary animate-ping"></span>
                      <span>INSPECTION TARGET: PV MODULE SURFACE</span>
                    </div>
                    <div className="font-label-caps text-label-caps bg-inverse-surface/90 px-2 py-1 rounded">
                      {activeLayer === "GRAD_CAM" ? "GRAD-CAM ATTENTION MAP" : "ORIGINAL RGB OPTICAL"}
                    </div>
                  </div>

                  {/* Visual region anomaly target crosshair */}
                  <div className="absolute top-1/2 left-1/3 -translate-x-1/2 -translate-y-1/2 flex flex-col items-center">
                    <div className="w-16 h-12 rounded border-2 border-error border-dashed flex items-center justify-center bg-error/20">
                      <span className="material-symbols-outlined text-error text-sm">
                        crisis_alert
                      </span>
                    </div>
                    <div className="mt-1 bg-inverse-surface text-on-primary font-code-id text-code-id px-1.5 py-0.5 rounded shadow text-[10px]">
                      AI Attention Peak (Saliency 0.94)
                    </div>
                  </div>

                  <div className="flex items-center justify-between text-on-primary text-xs font-code-id">
                    <span className="bg-inverse-surface/80 px-2 py-0.5 rounded">
                      Approx. Region: ~14.8% Surface
                    </span>
                    <span className="bg-inverse-surface/80 px-2 py-0.5 rounded">
                      Threshold: τ ≥ 0.60
                    </span>
                  </div>
                </div>
              </div>

              {/* Spectrum Legend & Comparison Toggles */}
              <div className="mt-space-sm pt-space-xs flex items-center justify-between flex-wrap gap-2">
                <div className="flex items-center gap-2">
                  <span className="font-label-caps text-label-caps text-on-surface-variant">
                    SALIENCY SCALE:
                  </span>
                  <div className="h-2.5 w-36 rounded bg-gradient-to-r from-primary via-secondary-container to-error"></div>
                  <span className="font-code-id text-code-id text-on-surface-variant">
                    0.0 (Min) — 1.0 (Max)
                  </span>
                </div>
                <div className="flex items-center gap-1">
                  <button
                    type="button"
                    onClick={() => {
                      setActiveLayer("GRAD_CAM");
                      showToast("Layer switched to Grad-CAM Attention Map.");
                    }}
                    className={`px-2 py-1 rounded font-body-sm text-body-sm font-medium transition-colors ${
                      activeLayer === "GRAD_CAM"
                        ? "bg-primary text-on-primary"
                        : "bg-surface-container-low hover:bg-surface-container text-on-surface"
                    }`}
                  >
                    Grad-CAM Attention
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setActiveLayer("RGB");
                      showToast("Layer switched to Original RGB Optical.");
                    }}
                    className={`px-2 py-1 rounded font-body-sm text-body-sm font-medium transition-colors ${
                      activeLayer === "RGB"
                        ? "bg-primary text-on-primary"
                        : "bg-surface-container-low hover:bg-surface-container text-on-surface"
                    }`}
                  >
                    Original RGB Capture
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Section B: Dedicated Inspection Chronology for Panel */}
        <section className="bg-surface-container-lowest p-space-md rounded shadow-sm border border-outline-variant/30">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-space-sm pb-space-sm mb-space-md bg-surface-container-low px-3 py-2 rounded border border-outline-variant/20">
            <div>
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-primary text-base">
                  table_chart
                </span>
                <h2 className="font-headline-sm text-headline-sm text-on-surface font-semibold">
                  Inspection History for {panel.panel_id}
                </h2>
              </div>
              <span className="font-code-id text-code-id text-on-surface-variant mt-0.5 block">
                Endpoint: GET /api/panels/{panel.panel_id}/inspections
              </span>
            </div>
            <div className="flex items-center gap-space-sm">
              <div className="inline-flex items-center gap-1 text-on-surface-variant font-body-sm text-body-sm bg-surface px-2 py-1 rounded border border-outline-variant/20">
                <span className="material-symbols-outlined text-sm">filter_alt</span>
                <span>
                  Showing: {inspections.length} of {panel.total_scans || 14} records
                </span>
              </div>
              <button
                onClick={() =>
                  showToast(
                    `Exported ${inspections.length} records for ${panel.panel_id} to CSV.`
                  )
                }
                className="p-1 rounded bg-surface text-on-surface-variant hover:text-on-surface transition-colors border border-outline-variant/20"
                title="Export Ledger CSV"
              >
                <span className="material-symbols-outlined text-base">
                  download
                </span>
              </button>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left font-body-sm text-body-sm border-collapse">
              <thead>
                <tr className="bg-surface-container-low text-on-surface-variant font-label-caps text-label-caps uppercase tracking-wider border-b border-outline-variant/20">
                  <th className="py-2.5 px-3">Inspection ID</th>
                  <th className="py-2.5 px-3">Date / UTC</th>
                  <th className="py-2.5 px-3">Detected Fault</th>
                  <th className="py-2.5 px-3">Confidence</th>
                  <th className="py-2.5 px-3">Severity</th>
                  <th className="py-2.5 px-3">Urgency</th>
                  <th className="py-2.5 px-3">Sensor Source</th>
                  <th className="py-2.5 px-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-outline-variant/20">
                {inspections.map((row) => (
                  <tr
                    key={row.inspection_id}
                    className="hover:bg-surface-container-low/60 transition-colors"
                  >
                    <td className="py-2.5 px-3 font-code-id text-code-id font-semibold text-primary">
                      #{row.inspection_id}
                    </td>
                    <td className="py-2.5 px-3 font-code-id text-code-id text-on-surface">
                      {row.inspection_timestamp.includes("T")
                        ? new Date(row.inspection_timestamp).toLocaleDateString(
                            "en-US",
                            {
                              month: "short",
                              day: "numeric",
                              year: "numeric",
                              hour: "2-digit",
                              minute: "2-digit",
                            }
                          )
                        : row.inspection_timestamp}
                    </td>
                    <td className="py-2.5 px-3 font-semibold">
                      <span className="inline-flex items-center gap-1">
                        <span
                          className={`material-symbols-outlined text-sm ${
                            row.predicted_class === "Electrical-damage"
                              ? "text-error"
                              : row.predicted_class === "Clean"
                              ? "text-secondary"
                              : "text-outline"
                          }`}
                        >
                          {row.predicted_class === "Electrical-damage"
                            ? "error"
                            : row.predicted_class === "Clean"
                            ? "check_circle"
                            : "grain"}
                        </span>
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
                    <td className="py-2.5 px-3 font-code-id text-code-id font-semibold text-on-surface">
                      {(row.confidence * 100).toFixed(1)}%
                    </td>
                    <td className="py-2.5 px-3">
                      <SeverityBadge severity={row.severity} />
                    </td>
                    <td className="py-2.5 px-3">
                      <UrgencyBadge urgency={row.urgency} />
                    </td>
                    <td className="py-2.5 px-3 text-on-surface-variant flex items-center gap-1 pt-3">
                      <span className="material-symbols-outlined text-xs text-primary">
                        flight
                      </span>
                      {row.sensor_source || "Drone Alpha (Optical RGB)"}
                    </td>
                    <td className="py-2.5 px-3 text-right">
                      <Link
                        href={`/inspection-result/${row.inspection_id}`}
                        className="inline-block px-2.5 py-1 rounded bg-surface-container-high hover:bg-surface-container text-on-surface font-body-sm text-body-sm font-semibold transition-colors shadow-sm"
                      >
                        View Diagnostics
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* Section C: Field Maintenance Log & API Telemetry (2 cols) */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-space-md">
          {/* Maintenance Ticket Tracker Card */}
          <div className="bg-surface-container-lowest p-space-md rounded shadow-sm border border-outline-variant/30 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-space-sm bg-surface-container-low px-2 py-1.5 rounded border border-outline-variant/20">
                <div className="flex items-center gap-1.5">
                  <span className="material-symbols-outlined text-error text-base">
                    confirmation_number
                  </span>
                  <span className="font-headline-sm text-headline-sm text-on-surface font-semibold">
                    Active Maintenance Ticket
                  </span>
                </div>
                <span className="px-2 py-0.5 rounded font-label-caps text-label-caps font-bold bg-error-container text-on-error-container">
                  DISPATCH ASSIGNED
                </span>
              </div>
              <div className="p-space-sm bg-surface-container-low rounded flex flex-col gap-2 border border-outline-variant/20">
                <div className="flex items-center justify-between">
                  <span className="font-code-id text-code-id font-bold text-on-surface">
                    Ticket #TKT-4412
                  </span>
                  <span className="font-label-caps text-label-caps text-error font-semibold">
                    Pending Field Dispatch
                  </span>
                </div>
                <p className="font-body-sm text-body-sm text-on-surface">
                  Assigned to:{" "}
                  <strong className="text-on-surface">Eng. Marcus Vance</strong> for
                  physical panel surface inspection and visual fault verification.
                </p>
                <div className="flex items-center gap-space-md text-xs font-code-id text-on-surface-variant pt-1">
                  <span>Created: Oct 24, 2023 · 14:30 UTC</span>
                  <span>Priority: Severity Level 1</span>
                </div>
              </div>
            </div>
            <div className="flex items-center justify-between mt-space-md pt-2 border-t border-outline-variant/20">
              <span className="font-label-sm text-label-sm text-on-surface-variant">
                SLA Deadline: 18 hours remaining
              </span>
              <button
                onClick={() =>
                  showToast("Work Order WO-4412 opened in dispatch portal.")
                }
                className="px-3 py-1.5 rounded bg-primary text-on-primary font-headline-sm text-headline-sm hover:bg-primary-container transition-colors shadow-sm"
              >
                Open Work Order
              </button>
            </div>
          </div>

          {/* API Route Mapping & Developer Context Card */}
          <div className="bg-surface-container-lowest p-space-md rounded shadow-sm border border-outline-variant/30 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-space-sm bg-surface-container-low px-2 py-1.5 rounded border border-outline-variant/20">
                <div className="flex items-center gap-1.5">
                  <span className="material-symbols-outlined text-primary text-base">
                    terminal
                  </span>
                  <span className="font-headline-sm text-headline-sm text-on-surface font-semibold">
                    Developer API Integration
                  </span>
                </div>
                <span className="font-label-caps text-label-caps text-secondary font-bold">
                  200 OK
                </span>
              </div>
              <div className="space-y-2">
                {/* Route 1 */}
                <div className="p-2 rounded bg-surface border border-outline-variant/20 font-code-id text-code-id flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="px-1.5 py-0.5 rounded bg-secondary-container text-on-secondary-container font-bold text-[10px]">
                      GET
                    </span>
                    <span className="text-on-surface">
                      /api/panels/{panel.panel_id}
                    </span>
                  </div>
                  <button
                    className="text-outline hover:text-on-surface transition-colors p-1"
                    onClick={() => {
                      navigator.clipboard.writeText(
                        `GET /api/panels/${panel.panel_id}`
                      );
                      showToast(`Copied: GET /api/panels/${panel.panel_id}`);
                    }}
                    title="Copy endpoint"
                  >
                    <span className="material-symbols-outlined text-sm">
                      content_copy
                    </span>
                  </button>
                </div>
                {/* Route 2 */}
                <div className="p-2 rounded bg-surface border border-outline-variant/20 font-code-id text-code-id flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="px-1.5 py-0.5 rounded bg-secondary-container text-on-secondary-container font-bold text-[10px]">
                      GET
                    </span>
                    <span className="text-on-surface">
                      /api/panels/{panel.panel_id}/inspections
                    </span>
                  </div>
                  <button
                    className="text-outline hover:text-on-surface transition-colors p-1"
                    onClick={() => {
                      navigator.clipboard.writeText(
                        `GET /api/panels/${panel.panel_id}/inspections`
                      );
                      showToast(
                        `Copied: GET /api/panels/${panel.panel_id}/inspections`
                      );
                    }}
                    title="Copy endpoint"
                  >
                    <span className="material-symbols-outlined text-sm">
                      content_copy
                    </span>
                  </button>
                </div>
              </div>
            </div>
            <div className="mt-space-md flex items-center justify-between pt-2 border-t border-outline-variant/20">
              <span className="font-label-sm text-label-sm text-on-surface-variant font-code-id">
                FastAPI REST API · EfficientNet-B0
              </span>
              <a
                className="font-code-id text-code-id text-primary font-semibold hover:underline inline-flex items-center gap-1"
                href="http://127.0.0.1:8000/docs"
                target="_blank"
                rel="noreferrer"
              >
                Open Swagger Doc{" "}
                <span className="material-symbols-outlined text-xs">
                  arrow_forward
                </span>
              </a>
            </div>
          </div>
        </section>

        {/* Statutory Disclaimer */}
        <Disclaimer />
      </div>
    </AppShell>
  );
}
