"use client";

import React, { useState, useEffect, use } from "react";
import Link from "next/link";
import { AppShell } from "@/components/layout/AppShell";
import { SeverityBadge, UrgencyBadge } from "@/components/common/StatusBadge";
import { Disclaimer } from "@/components/common/Disclaimer";
import { useToast } from "@/components/common/Toast";
import { MOCK_INSPECTIONS } from "@/lib/mock-data";
import { fetchInspection } from "@/lib/api";
import { InspectionResponse } from "@/types/inspection";

export default function InspectionResultPage({
  params,
}: {
  params?: Promise<{ id?: string }>;
}) {
  const resolvedParams = params ? use(params) : undefined;
  const inspectionId = resolvedParams?.id || "INS-2023-8841";

  const { showToast } = useToast();
  const [isGeneratingPdf, setIsGeneratingPdf] = useState(false);

  // Initial fallback while fetching from API
  const defaultInspection =
    MOCK_INSPECTIONS.find(
      (item) =>
        String(item.inspection_id).toLowerCase() === inspectionId.toLowerCase()
    ) || MOCK_INSPECTIONS[0];

  const [inspection, setInspection] = useState<InspectionResponse>(defaultInspection);

  useEffect(() => {
    let isMounted = true;
    async function loadData() {
      try {
        const data = await fetchInspection(inspectionId);
        if (isMounted && data) {
          setInspection(data);
        }
      } catch {
        // Fallback to default inspection
      }
    }
    loadData();
    return () => {
      isMounted = false;
    };
  }, [inspectionId]);

  const handleDownloadPdf = () => {
    setIsGeneratingPdf(true);
    setTimeout(() => {
      setIsGeneratingPdf(false);
      showToast("PDF Engineering Report generated and downloaded.");
    }, 1500);
  };

  return (
    <AppShell
      breadcrumbs={[
        { label: "Asset", href: "/" },
        { label: "Solar Farm Alpha", href: "/" },
        { label: "SEC-4", href: "/" },
        { label: `Dossier: ${inspection.inspection_id}`, active: true },
      ]}
    >
      <div className="flex flex-col w-full gap-space-md">
        {/* Top Context Banner & Primary ID Bar */}
        <div className="flex flex-col md:flex-row md:items-center md:justify-between bg-surface-container-lowest p-space-md rounded-lg shadow-sm border border-outline-variant/30 gap-space-sm">
          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-space-sm flex-wrap">
              <span className="px-space-xs py-0.5 bg-primary/10 text-primary rounded font-label-caps text-label-caps">
                AI OPTICAL INSPECTION
              </span>
              <span className="font-label-caps text-label-caps text-on-surface-variant font-medium">
                SESSION ID:
              </span>
              <span className="font-code-id text-code-id text-on-surface font-semibold bg-surface-container-low px-1.5 py-0.5 rounded border border-outline-variant/20">
                #{inspection.inspection_id}
              </span>
            </div>
            <div className="font-headline-lg text-headline-lg text-on-surface tracking-tight">
              Inspection Diagnostic Report:{" "}
              <span className="font-code-metric text-headline-lg text-primary font-bold">
                #{inspection.inspection_id}
              </span>
            </div>
          </div>

          {/* Status Badge & Actions */}
          <div className="flex items-center gap-space-md self-start md:self-auto">
            <div className="flex flex-col text-right">
              <div className="flex items-center gap-1.5 justify-end">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-secondary opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-secondary"></span>
                </span>
                <span className="font-label-caps text-label-caps text-secondary font-semibold">
                  AI INFERENCE COMPLETED IN 38ms
                </span>
              </div>
              <div className="font-code-id text-code-id text-on-surface-variant flex items-center gap-1 justify-end mt-0.5">
                <span className="material-symbols-outlined text-sm text-outline">
                  schedule
                </span>
                <span>
                  {inspection.inspection_timestamp.includes("T")
                    ? new Date(inspection.inspection_timestamp).toUTCString()
                    : inspection.inspection_timestamp}
                </span>
              </div>
            </div>
            <div className="flex items-center gap-1 bg-surface-container-low p-1 rounded border border-outline-variant/20">
              <button
                onClick={() => showToast("Sending dossier to connected printer...")}
                className="p-1.5 hover:bg-surface-container-highest rounded text-on-surface-variant transition-colors"
                title="Print Telemetry Feed"
              >
                <span className="material-symbols-outlined text-lg">print</span>
              </button>
              <button
                onClick={() => {
                  navigator.clipboard.writeText(JSON.stringify(inspection, null, 2));
                  showToast("Raw inspection JSON copied to clipboard.");
                }}
                className="p-1.5 hover:bg-surface-container-highest rounded text-on-surface-variant transition-colors"
                title="Raw JSON Metadata"
              >
                <span className="material-symbols-outlined text-lg">data_object</span>
              </button>
            </div>
          </div>
        </div>

        {/* Key Metadata Summary Grid (4 Cells) */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-space-sm">
          {/* Cell 1: Panel & Location */}
          <div className="bg-surface-container-lowest p-space-md rounded-lg shadow-sm border border-outline-variant/30 flex flex-col justify-between">
            <div className="flex items-center justify-between text-on-surface-variant">
              <span className="font-label-caps text-label-caps uppercase">
                TARGET MODULE ASSET
              </span>
              <span className="material-symbols-outlined text-base text-primary">
                solar_power
              </span>
            </div>
            <div className="mt-2 flex items-center justify-between">
              <Link
                href={`/panel-details/${inspection.panel_id}`}
                className="font-code-metric text-code-metric text-primary hover:text-primary-container transition-colors tracking-tight underline decoration-primary/30 underline-offset-4"
              >
                {inspection.panel_id}
              </Link>
              <span className="px-1.5 py-0.5 bg-surface-container text-on-surface-variant font-label-caps text-label-caps rounded">
                STRING-04
              </span>
            </div>
            <div className="mt-2 flex items-center gap-1 font-body-sm text-body-sm text-on-surface-variant">
              <span className="material-symbols-outlined text-sm text-outline">
                location_on
              </span>
              <span>
                {inspection.location}{" "}
                <span className="text-outline-variant font-code-id">(R-14.2)</span>
              </span>
            </div>
          </div>

          {/* Cell 2: Predicted Fault & Classification */}
          <div className="bg-surface-container-lowest p-space-md rounded-lg shadow-sm border border-outline-variant/30 flex flex-col justify-between">
            <div className="flex items-center justify-between text-on-surface-variant">
              <span className="font-label-caps text-label-caps uppercase">
                PREDICTED ANOMALY CLASS
              </span>
              <span className="material-symbols-outlined text-base text-error">
                electric_bolt
              </span>
            </div>
            <div className="mt-2">
              <div className="font-headline-sm text-headline-sm text-error font-bold flex items-center gap-1.5">
                <span>{inspection.predicted_class}</span>
              </div>
              <div className="font-body-sm text-body-sm text-on-surface-variant font-medium mt-0.5 truncate">
                Sub: {inspection.sub_fault || "Localized Anomaly"}
              </div>
            </div>
            <div className="mt-2 flex items-center gap-2">
              <span className="font-label-caps text-label-caps text-on-surface-variant">
                APPROX VISUAL REGION:
              </span>
              <span className="font-code-id text-code-id text-primary font-semibold">
                {typeof inspection.visual_region_area_percent === "number"
                  ? `${inspection.visual_region_area_percent.toFixed(1)}% OF SURFACE`
                  : "14.8% OF SURFACE"}
              </span>
            </div>
          </div>

          {/* Cell 3: Confidence Score & Probability Bar */}
          <div className="bg-surface-container-lowest p-space-md rounded-lg shadow-sm border border-outline-variant/30 flex flex-col justify-between">
            <div className="flex items-center justify-between text-on-surface-variant">
              <span className="font-label-caps text-label-caps uppercase">
                MODEL INFERENCE CONFIDENCE
              </span>
              <span className="material-symbols-outlined text-base text-primary">
                model_training
              </span>
            </div>
            <div className="mt-2 flex items-baseline justify-between">
              <span className="font-code-metric text-headline-xl text-on-surface font-bold">
                {(inspection.confidence * 100).toFixed(1)}
                <span className="text-headline-md text-primary font-semibold">%</span>
              </span>
              <span className="font-label-caps text-label-caps text-secondary font-semibold bg-secondary-container/40 px-1.5 py-0.5 rounded border border-secondary/20">
                p &gt; 0.95 HIGH
              </span>
            </div>
            <div className="mt-2 flex flex-col gap-1">
              <div className="w-full bg-surface-container-highest rounded-full h-1.5 overflow-hidden">
                <div
                  className="bg-primary h-1.5 rounded-full transition-all duration-500"
                  style={{ width: `${inspection.confidence * 100}%` }}
                ></div>
              </div>
              <div className="flex justify-between font-label-caps text-label-caps text-on-surface-variant">
                <span>Softmax Dist.</span>
                <span>σ = 0.012</span>
              </div>
            </div>
          </div>

          {/* Cell 4: Operational Status & Urgency Flags */}
          <div className="bg-surface-container-lowest p-space-md rounded-lg shadow-sm border border-outline-variant/30 flex flex-col justify-between">
            <div className="flex items-center justify-between text-on-surface-variant">
              <span className="font-label-caps text-label-caps uppercase">
                TRIAGE PRIORITY LEDGER
              </span>
              <span className="material-symbols-outlined text-base text-error">
                priority_high
              </span>
            </div>
            <div className="mt-2 flex flex-col gap-1.5">
              <div className="flex items-center justify-between">
                <span className="font-label-caps text-label-caps text-on-surface-variant">
                  VISUAL SEVERITY:
                </span>
                <SeverityBadge severity={inspection.severity} />
              </div>
              <div className="flex items-center justify-between">
                <span className="font-label-caps text-label-caps text-on-surface-variant">
                  URGENCY:
                </span>
                <UrgencyBadge urgency={inspection.urgency} />
              </div>
            </div>
            <div className="mt-2 flex items-center justify-between pt-1 bg-surface-container-low px-2 py-1 rounded border border-outline-variant/20">
              <span className="font-label-caps text-label-caps text-error font-semibold">
                MANUAL DISPATCH:
              </span>
              <span className="font-code-id text-code-id text-error font-bold flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-error animate-pulse"></span>
                {inspection.manual_inspection_recommended
                  ? "REQUIRED (YES)"
                  : "NOT REQUIRED"}
              </span>
            </div>
          </div>
        </div>

        {/* Confidence Warning Callout (When Applicable) */}
        {inspection.confidence_warning && (
          <div className="bg-error-container/40 p-space-md rounded-lg shadow-sm border border-error/30 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-space-md">
            <div className="flex items-start gap-space-md">
              <div className="p-2 bg-error text-on-error rounded-lg flex items-center justify-center shrink-0">
                <span className="material-symbols-outlined text-xl">warning</span>
              </div>
              <div className="flex flex-col">
                <span className="font-headline-sm text-headline-sm text-on-error-container font-semibold">
                  Confidence Warning &amp; Sensor Cross-Check Required
                </span>
                <p className="font-body-md text-body-md text-on-surface-variant mt-0.5">
                  {inspection.confidence_warning}
                </p>
              </div>
            </div>
            <button
              onClick={() => showToast("Anomaly trigger confirmed by engineer.")}
              className="shrink-0 px-space-md py-1.5 bg-error text-on-error font-body-md text-body-md font-semibold rounded hover:bg-on-error-container transition-colors shadow-sm flex items-center gap-1.5"
            >
              <span className="material-symbols-outlined text-base">
                verified_user
              </span>
              Confirm Anomaly Trigger
            </button>
          </div>
        )}

        {/* Diagnostic Visual Analysis (3 Panels) */}
        <div className="flex flex-col gap-space-xs">
          <div className="flex items-center justify-between px-1">
            <div className="flex items-center gap-space-sm">
              <span className="font-headline-md text-headline-md text-on-surface font-semibold">
                Diagnostic Visual Analysis
              </span>
              <span className="px-2 py-0.5 bg-surface-container-high rounded font-code-id text-code-id text-on-surface-variant">
                OPTICAL RGB • EfficientNet-B0 (224×224)
              </span>
            </div>
            <div className="flex items-center gap-1 font-label-caps text-label-caps text-on-surface-variant">
              <span>ZOOM: 100%</span>
              <span>•</span>
              <span>SCALE: RECTIFIED</span>
            </div>
          </div>

          {/* 3-Column Image Comparison Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-space-sm">
            {/* Panel 1: Original Image */}
            <div className="bg-surface-container-lowest rounded-lg overflow-hidden shadow-sm border border-outline-variant/30 flex flex-col">
              <div className="p-space-sm bg-surface-container-low flex items-center justify-between border-b border-outline-variant/20">
                <div className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-primary"></span>
                  <span className="font-headline-sm text-headline-sm text-on-surface font-bold">
                    1. Original Image
                  </span>
                </div>
                <span className="font-label-caps text-label-caps text-on-surface-variant bg-surface-container-lowest px-1.5 py-0.5 rounded border border-outline-variant/20">
                  RGB OPTICAL
                </span>
              </div>
              <div className="relative h-72 w-full bg-surface-dim overflow-hidden group">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  className="w-full h-full object-cover"
                  alt="Original RGB optical panel capture"
                  src={
                    inspection.original_image_url ||
                    "https://lh3.googleusercontent.com/aida-public/AB6AXuClu9_LnwC1NQr9D9IL_YRk0G3iChTCUSzuRi3UCDO9Dfv9QtZq7gHbu38Hoj6g3gEY09fG0Ar6xS_--_c4k0lyAzGNlr-OZ7xtDPysKf1efYN1wCU_eRp7LPqcGGHmcVF1KlOj3utNLMbe_3pj1twTjdKsle077yF2JW57TmLovR6Ekw_pZxm5xjWY3geAATZ9R3srg_HXr9DJ2FDYxqU-I8b0YLbyUaeEHemeOf3hLeXt3WEJLAq-KA"
                  }
                />
                <div className="absolute top-2 left-2 bg-inverse-surface/80 backdrop-blur-sm text-inverse-on-surface px-2 py-1 rounded font-code-id text-code-id flex flex-col gap-0.5">
                  <span className="font-label-caps text-label-caps text-primary-fixed-dim">
                    COORDINATES
                  </span>
                  <span>Lat: 17.3850° N</span>
                  <span>Lon: 78.4867° E</span>
                </div>
                <div className="absolute bottom-2 right-2 bg-inverse-surface/80 backdrop-blur-sm text-inverse-on-surface px-2 py-1 rounded font-code-id text-code-id">
                  Optical RGB (224×224 AI Input)
                </div>
                <div className="absolute inset-0 pointer-events-none opacity-20 flex items-center justify-center">
                  <div className="w-16 h-16 border-t border-b border-on-surface"></div>
                  <div className="h-16 w-16 border-l border-r border-on-surface absolute"></div>
                </div>
              </div>
              <div className="p-space-sm bg-surface-container-lowest flex items-center justify-between text-on-surface-variant font-body-sm text-body-sm border-t border-outline-variant/20">
                <span>Optical RGB Surface Capture</span>
                <span className="font-code-id text-code-id text-on-surface font-medium">
                  Tensor: 224×224×3
                </span>
              </div>
            </div>

            {/* Panel 2: AI Attention / Grad-CAM */}
            <div className="bg-surface-container-lowest rounded-lg overflow-hidden shadow-sm border border-outline-variant/30 flex flex-col">
              <div className="p-space-sm bg-surface-container-low flex items-center justify-between border-b border-outline-variant/20">
                <div className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-error"></span>
                  <span className="font-headline-sm text-headline-sm text-on-surface font-bold">
                    2. AI Attention / Grad-CAM
                  </span>
                </div>
                <span className="font-label-caps text-label-caps text-error bg-error-container px-1.5 py-0.5 rounded font-bold">
                  LAYER: top_conv
                </span>
              </div>
              <div className="relative h-72 w-full bg-surface-dim overflow-hidden group">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  className="w-full h-full object-cover filter contrast-125"
                  alt="Grad-CAM AI attention heatmap"
                  src={
                    inspection.gradcam_image_url ||
                    "https://lh3.googleusercontent.com/aida-public/AB6AXuBTQZK5eOuP_wR4SuHW_doQ7vTB7nX4yRLqcoeDY1ckYJVcPC-lFhkePC60bbRNCMfAkRcLwwwTzjfJD2h8rPjd4snm5WjRPGi9jsPZXBE0sWKqNKcwzFrZsHym--C1KRO1s8kHDO67NNDVv2yi_lUXzv6gqsiZvWlvKRxOI37GpzrIxs2o13zX2-NAMbEueokGBIb9jM6IGttmHBdUzBBws3lFlSdFjWJ2DXLI2WW25MyctUCm24zrzQ"
                  }
                />
                <div className="absolute inset-0 pointer-events-none mix-blend-color-burn opacity-60 bg-gradient-to-tr from-primary/30 via-transparent to-error/70"></div>
                <div className="absolute top-1/3 left-1/4 w-32 h-28 rounded-full bg-gradient-to-r from-error via-amber-400 to-primary opacity-75 blur-xl pointer-events-none transform -rotate-12"></div>
                {/* Grad-CAM Saliency Scale (0.0 to 1.0) */}
                <div className="absolute top-2 right-2 bg-inverse-surface/85 backdrop-blur-sm p-1.5 rounded flex flex-col items-center gap-1">
                  <span className="font-label-caps text-label-caps text-inverse-on-surface text-[10px]">
                    SALIENCY
                  </span>
                  <div className="w-2.5 h-20 rounded bg-gradient-to-b from-error via-amber-300 via-secondary to-primary"></div>
                  <span className="font-code-id text-code-id text-inverse-on-surface text-[10px]">
                    1.0 Max
                  </span>
                  <span className="font-code-id text-code-id text-inverse-on-surface text-[10px]">
                    0.0 Min
                  </span>
                </div>
                <div className="absolute bottom-2 left-2 bg-inverse-surface/80 backdrop-blur-sm text-inverse-on-surface px-2 py-1 rounded font-code-id text-code-id">
                  AI Attention Peak Saliency: 0.94
                </div>
              </div>
              <div className="p-space-sm bg-surface-container-lowest flex items-center justify-between text-on-surface-variant font-body-sm text-body-sm border-t border-outline-variant/20">
                <span>EfficientNet-B0 Conv Saliency Map</span>
                <span className="font-code-id text-code-id text-secondary font-semibold">
                  TENSOR: VALID
                </span>
              </div>
            </div>

            {/* Panel 3: Approximate Visual Region */}
            <div className="bg-surface-container-lowest rounded-lg overflow-hidden shadow-sm border border-outline-variant/30 flex flex-col">
              <div className="p-space-sm bg-surface-container-low flex items-center justify-between border-b border-outline-variant/20">
                <div className="flex items-center gap-1.5 truncate">
                  <span className="w-2 h-2 rounded-full bg-error"></span>
                  <span className="font-headline-sm text-headline-sm text-on-surface font-bold truncate">
                    3. Approximate Visual Region
                  </span>
                </div>
                <span className="font-label-caps text-label-caps text-error bg-error-container px-1.5 py-0.5 rounded font-bold uppercase shrink-0">
                  ESTIMATE ONLY
                </span>
              </div>
              <div className="relative h-72 w-full bg-surface-dim overflow-hidden">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  className="w-full h-full object-cover"
                  alt="Approximate visual region with bounding box"
                  src={
                    inspection.region_image_url ||
                    "https://lh3.googleusercontent.com/aida-public/AB6AXuDbFDONeqnQUktFSLCv03a2aeumzUqoFQvnB2DboBJE3RGKn9-dMxa8QYC00CUGe_YQQZ1Ra_ddGhi9InhApMWUtTkjnNSsS64QubI3WEOgcIsMi_uIKeddqJtRqvJyBat-dCKErDHciQpiLsQWBwZ2eBjF8e69v7PYDgeOphhHf3QF1BIhDDPLa4hVoLA7P4NHBqYcE3aQhWz921AOm2dWaKDWLzcdb8k92JeUPrAOfqhbTwtkQQfBWA"
                  }
                />
                {/* Approximate Fault Region Bounding Box */}
                <div className="absolute top-[28%] left-[24%] w-36 h-24 bg-error/15 rounded border-2 border-error pointer-events-none flex flex-col justify-between p-1.5 shadow-md">
                  <div className="flex justify-between items-start">
                    <span className="px-1 py-0.5 bg-error text-on-error font-label-caps text-label-caps rounded font-bold">
                      VISUAL REGION
                    </span>
                    <span className="font-code-id text-code-id text-error font-bold bg-surface-container-lowest/90 px-1 rounded">
                      ~{(inspection.visual_region_area_percent ?? 14.8).toFixed(1)}% Area
                    </span>
                  </div>
                  <div className="flex justify-between items-end">
                    <span className="font-code-id text-code-id text-on-surface font-bold bg-surface-container-lowest/90 px-1 rounded text-[11px]">
                      {inspection.predicted_class}
                    </span>
                    <span className="w-2 h-2 bg-error rounded-full animate-ping"></span>
                  </div>
                </div>
                {/* Specific Mandatory Label per Requirement */}
                <div className="absolute bottom-2 left-2 right-2 bg-inverse-surface/90 backdrop-blur-sm text-inverse-on-surface px-2.5 py-1.5 rounded">
                  <div className="font-code-id text-code-id font-semibold text-error-container text-[11px] text-center">
                    Visual region estimation only — NOT an exact defect boundary.
                  </div>
                </div>
              </div>
              <div className="p-space-sm bg-surface-container-lowest flex items-center justify-between text-on-surface-variant font-body-sm text-body-sm border-t border-outline-variant/20">
                <span>Activation Threshold: τ ≥ 0.60</span>
                <span className="font-code-id text-code-id text-on-surface font-medium">
                  BBox Approximation
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Bento Section: Maintenance Recommendation & Telemetry */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-space-sm">
          {/* Maintenance Directive (8 Cols) */}
          <div className="lg:col-span-8 bg-surface-container-lowest p-space-md rounded-lg shadow-sm border border-outline-variant/30 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between pb-space-sm bg-surface-container-low px-space-sm py-2 rounded border border-outline-variant/20">
                <div className="flex items-center gap-space-sm">
                  <div className="p-1.5 bg-primary text-on-primary rounded">
                    <span className="material-symbols-outlined text-lg">
                      engineering
                    </span>
                  </div>
                  <div>
                    <span className="font-headline-sm text-headline-sm text-on-surface font-bold">
                      Field Engineering Maintenance Directive
                    </span>
                    <div className="font-label-caps text-label-caps text-on-surface-variant">
                      WORK ORDER TICKET PROTOCOL: LEVEL-2 PRIORITY
                    </div>
                  </div>
                </div>
                <span className="px-2 py-0.5 bg-error text-on-error font-label-caps text-label-caps rounded font-bold">
                  SLA: 24 HOURS
                </span>
              </div>

              {/* Action item */}
              <div className="mt-space-md bg-surface-container-low p-space-md rounded border border-outline-variant/20">
                <div className="flex items-start gap-space-sm">
                  <span className="material-symbols-outlined text-error text-xl shrink-0 mt-0.5">
                    assignment_late
                  </span>
                  <div className="flex flex-col gap-1">
                    <span className="font-body-md text-body-md text-on-surface font-bold uppercase tracking-tight">
                      Action Item Directive:
                    </span>
                    <p className="font-body-lg text-body-lg text-on-surface leading-relaxed">
                      {inspection.maintenance_action}
                    </p>
                  </div>
                </div>
              </div>

              {/* Required field equipment */}
              <div className="mt-space-md">
                <span className="font-label-caps text-label-caps text-on-surface-variant uppercase block mb-space-xs font-bold">
                  Required Field Equipment &amp; Diagnostics
                </span>
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-2">
                  <div className="flex items-center gap-2 p-2 bg-surface-container rounded border border-outline-variant/20">
                    <span className="material-symbols-outlined text-primary text-base">
                      photo_camera
                    </span>
                    <span className="font-body-sm text-body-sm text-on-surface font-medium">
                      Digital Optical Kit
                    </span>
                  </div>
                  <div className="flex items-center gap-2 p-2 bg-surface-container rounded border border-outline-variant/20">
                    <span className="material-symbols-outlined text-primary text-base">
                      handyman
                    </span>
                    <span className="font-body-sm text-body-sm text-on-surface font-medium">
                      MC4 Disconnect Tool
                    </span>
                  </div>
                  <div className="flex items-center gap-2 p-2 bg-surface-container rounded border border-outline-variant/20">
                    <span className="material-symbols-outlined text-primary text-base">
                      speed
                    </span>
                    <span className="font-body-sm text-body-sm text-on-surface font-medium">
                      Calibrated Multimeter
                    </span>
                  </div>
                  <div className="flex items-center gap-2 p-2 bg-surface-container rounded border border-outline-variant/20">
                    <span className="material-symbols-outlined text-primary text-base">
                      grid_view
                    </span>
                    <span className="font-body-sm text-body-sm text-on-surface font-medium">
                      Submodule (360W Mono)
                    </span>
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-space-md pt-space-sm flex items-center justify-between font-label-caps text-label-caps text-on-surface-variant border-t border-outline-variant/20">
              <span>ASSIGNED CREW: REGION-1 DISPATCH</span>
              <span>SUPERVISOR SIGN-OFF: PENDING</span>
            </div>
          </div>

          {/* AI Inspection Metrics Ledger (4 Cols) */}
          <div className="lg:col-span-4 bg-surface-container-lowest p-space-md rounded-lg shadow-sm border border-outline-variant/30 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between pb-space-xs border-b border-outline-variant/20">
                <span className="font-headline-sm text-headline-sm text-on-surface font-bold">
                  AI Assessment Ledger
                </span>
                <span className="font-label-caps text-label-caps text-primary font-bold">
                  EFFICIENTNET-B0
                </span>
              </div>
              <div className="mt-space-sm flex flex-col">
                <div className="flex items-center justify-between py-2 bg-surface-container-low px-2 rounded mb-1 border border-outline-variant/20">
                  <span className="font-body-sm text-body-sm text-on-surface-variant">
                    Predicted Fault
                  </span>
                  <span className="font-code-id text-code-id text-on-surface font-bold">
                    {inspection.predicted_class}
                  </span>
                </div>
                <div className="flex items-center justify-between py-2 px-2">
                  <span className="font-body-sm text-body-sm text-on-surface-variant">
                    Classification Confidence
                  </span>
                  <span className="font-code-id text-code-id text-primary font-bold">
                    {(inspection.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex items-center justify-between py-2 bg-surface-container-low px-2 rounded mb-1 border border-outline-variant/20">
                  <span className="font-body-sm text-body-sm text-on-surface-variant">
                    Visual Severity
                  </span>
                  <span className="font-code-id text-code-id text-error font-bold">
                    {inspection.severity} (Visual Est.)
                  </span>
                </div>
                <div className="flex items-center justify-between py-2 px-2">
                  <span className="font-body-sm text-body-sm text-on-surface-variant">
                    Approx. Region Area
                  </span>
                  <span className="font-code-id text-code-id text-primary font-bold">
                    {(inspection.visual_region_area_percent ?? 14.8).toFixed(1)}% surface
                  </span>
                </div>
                <div className="flex items-center justify-between py-2 bg-surface-container-low px-2 rounded mb-1 border border-outline-variant/20">
                  <span className="font-body-sm text-body-sm text-on-surface-variant">
                    Manual Physical Inspection
                  </span>
                  <span className="font-label-caps text-label-caps text-on-secondary-container bg-secondary-container px-1.5 py-0.5 rounded font-bold">
                    {inspection.manual_inspection_recommended ? "RECOMMENDED" : "ROUTINE"}
                  </span>
                </div>
              </div>
            </div>
            <div className="mt-space-sm p-2 bg-surface rounded flex items-center justify-between border border-outline-variant/20">
              <div className="flex items-center gap-1.5">
                <span className="material-symbols-outlined text-sm text-outline">
                  policy
                </span>
                <span className="font-label-caps text-label-caps text-on-surface-variant">
                  STATUS: VERIFIED
                </span>
              </div>
              <button
                onClick={() => showToast("Inspection parameters refreshed.")}
                className="font-body-sm text-body-sm text-primary font-semibold hover:underline"
              >
                Refresh
              </button>
            </div>
          </div>
        </div>

        {/* Mandatory Statutory Disclaimer Footer */}
        <Disclaimer />

        {/* Workflow Action Button Bar */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-space-sm bg-surface-container-lowest p-space-md rounded-lg shadow-sm border border-outline-variant/30">
          <Link
            href="/inspection-history"
            className="w-full sm:w-auto px-space-md py-2 bg-surface-container text-on-surface hover:bg-surface-container-high rounded font-body-md text-body-md font-semibold transition-colors flex items-center justify-center gap-1.5 shadow-sm border border-outline-variant/30"
          >
            <span className="material-symbols-outlined text-base">
              arrow_back
            </span>
            Back to Inspections
          </Link>
          <div className="flex items-center flex-wrap sm:flex-nowrap gap-space-sm w-full sm:w-auto">
            <button
              onClick={() => showToast("Flagged for manual engineering dispatch review.")}
              className="flex-1 sm:flex-none px-space-md py-2 bg-error-container text-on-error-container hover:bg-error/20 rounded font-body-md text-body-md font-semibold transition-colors flex items-center justify-center gap-1.5 shadow-sm border border-error/20"
            >
              <span className="material-symbols-outlined text-base">flag</span>
              Flag for Manual Review
            </button>
            <button
              onClick={() => showToast("Work Order WO-4412 created and queued.")}
              className="flex-1 sm:flex-none px-space-md py-2 bg-surface-container-high text-on-surface hover:bg-surface-container-highest rounded font-body-md text-body-md font-semibold transition-colors flex items-center justify-center gap-1.5 shadow-sm border border-outline-variant/30"
            >
              <span className="material-symbols-outlined text-base">
                calendar_add_on
              </span>
              Schedule Work Order
            </button>
            <button
              onClick={handleDownloadPdf}
              disabled={isGeneratingPdf}
              className="w-full sm:w-auto px-space-md py-2 bg-primary text-on-primary hover:bg-primary-container rounded font-body-md text-body-md font-semibold transition-colors flex items-center justify-center gap-1.5 shadow-sm cursor-pointer disabled:opacity-75"
            >
              {isGeneratingPdf ? (
                <>
                  <span className="material-symbols-outlined text-base animate-spin">
                    refresh
                  </span>
                  <span>Generating PDF...</span>
                </>
              ) : (
                <>
                  <span className="material-symbols-outlined text-base">
                    picture_as_pdf
                  </span>
                  <span>Download PDF Engineering Report</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </AppShell>
  );
}
