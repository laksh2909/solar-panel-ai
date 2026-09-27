"use client";

import React, { useState, useEffect, use } from "react";
import Image from "next/image";
import Link from "next/link";
import { ArrowLeft, AlertTriangle } from "lucide-react";
import { AppShell } from "@/components/layout/AppShell";
import { SeverityBadge, UrgencyBadge } from "@/components/common/StatusBadge";
import { Disclaimer } from "@/components/common/Disclaimer";
import { MOCK_INSPECTIONS } from "@/lib/mock-data";
import { fetchInspection } from "@/lib/api";
import { InspectionResponse } from "@/types/inspection";
import {
  formatFaultName,
  formatInspectionDate,
  getConfidenceFeedback,
  getSeverityFeedback,
} from "@/lib/formatters";

const FALLBACK_ORIGINAL_IMG =
  "https://lh3.googleusercontent.com/aida-public/AB6AXuClu9_LnwC1NQr9D9IL_YRk0G3iChTCUSzuRi3UCDO9Dfv9QtZq7gHbu38Hoj6g3gEY09fG0Ar6xS_--_c4k0lyAzGNlr-OZ7xtDPysKf1efYN1wCU_eRp7LPqcGGHmcVF1KlOj3utNLMbe_3pj1twTjdKsle077yF2JW57TmLovR6Ekw_pZxm5xjWY3geAATZ9R3srg_HXr9DJ2FDYxqU-I8b0YLbyUaeEHemeOf3hLeXt3WEJLAq-KA";

const FALLBACK_GRADCAM_IMG =
  "https://lh3.googleusercontent.com/aida-public/AB6AXuBTQZK5eOuP_wR4SuHW_doQ7vTB7nX4yRLqcoeDY1ckYJVcPC-lFhkePC60bbRNCMfAkRcLwwwTzjfJD2h8rPjd4snm5WjRPGi9jsPZXBE0sWKqNKcwzFrZsHym--C1KRO1s8kHDO67NNDVv2yi_lUXzv6gqsiZvWlvKRxOI37GpzrIxs2o13zX2-NAMbEueokGBIb9jM6IGttmHBdUzBBws3lFlSdFjWJ2DXLI2WW25MyctUCm24zrzQ";

const FALLBACK_REGION_IMG =
  "https://lh3.googleusercontent.com/aida-public/AB6AXuDbFDONeqnQUktFSLCv03a2aeumzUqoFQvnB2DboBJE3RGKn9-dMxa8QYC00CUGe_YQQZ1Ra_ddGhi9InhApMWUtTkjnNSsS64QubI3WEOgcIsMi_uIKeddqJtRqvJyBat-dCKErDHciQpiLsQWBwZ2eBjF8e69v7PYDgeOphhHf3QF1BIhDDPLa4hVoLA7P4NHBqYcE3aQhWz921AOm2dWaKDWLzcdb8k92JeUPrAOfqhbTwtkQQfBWA";

export default function InspectionResultPage({
  params,
}: {
  params?: Promise<{ id?: string }>;
}) {
  const resolvedParams = params ? use(params) : undefined;
  const inspectionId = resolvedParams?.id || "23";

  const [inspection, setInspection] = useState<InspectionResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    async function loadData() {
      try {
        setLoading(true);
        setError(null);
        const data = await fetchInspection(inspectionId);
        if (!isMounted) return;
        if (!data) {
          setError("Inspection result could not be loaded.");
          return;
        }
        setInspection(data);
      } catch (err) {
        console.error("Failed to load inspection:", err);
        if (isMounted) {
          setError("Inspection result could not be loaded.");
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    }

    loadData();
    return () => {
      isMounted = false;
    };
  }, [inspectionId]);

  const record = inspection || MOCK_INSPECTIONS[0];
  const confidencePercent = record.confidence <= 1 ? record.confidence * 100 : record.confidence;
  const confidenceNormalized = Math.min(Math.max(confidencePercent / 100, 0), 1);
  const confidenceInfo = getConfidenceFeedback(record.confidence);
  const confidenceHeadline =
    confidenceNormalized >= 0.8
      ? "High confidence in this prediction"
      : confidenceNormalized >= 0.6
      ? "Moderate confidence in this prediction"
      : "Low confidence — manual review recommended";
  const severityExplanation = getSeverityFeedback(record.severity);
  const maintenanceActions =
    Array.isArray((record as InspectionResponse & { maintenance_actions?: string[] }).maintenance_actions) &&
    (record as InspectionResponse & { maintenance_actions?: string[] }).maintenance_actions!.length > 0
      ? (record as InspectionResponse & { maintenance_actions?: string[] }).maintenance_actions!
      : [
          record.maintenance_action || "Continue routine visual monitoring.",
          "Inspect the affected area for any additional visible issues.",
          "Reassess during the next scheduled inspection cycle.",
        ].slice(0, 3);

  const confidenceTone =
    confidencePercent >= 80
      ? { label: "High confidence", color: "text-emerald-700", bar: "bg-emerald-500" }
      : confidencePercent >= 60
      ? { label: "Moderate confidence", color: "text-amber-700", bar: "bg-amber-500" }
      : { label: "Low confidence", color: "text-rose-700", bar: "bg-rose-500" };

  const regionPercentage =
    typeof record.visual_region_area_percent === "number" && Number.isFinite(record.visual_region_area_percent)
      ? record.visual_region_area_percent
      : null;

  const getActionDescription = (index: number) => {
    const primaryFault = record.predicted_class;
    const severity = record.severity;

    if (primaryFault === "Electrical-damage") {
      if (index === 0) {
        return severity === "HIGH"
          ? "Arrange a qualified electrical inspection as soon as possible."
          : severity === "MEDIUM"
          ? "Schedule a qualified electrical inspection."
          : "Schedule a qualified inspection during routine maintenance.";
      }
      if (index === 1) {
        return severity === "HIGH"
          ? "Inspect visible module connections and affected areas for signs of electrical damage."
          : severity === "MEDIUM"
          ? "Inspect the visible affected areas and module connections."
          : "Review the affected area for any visible changes.";
      }
      return severity === "HIGH"
        ? "Document the findings and perform a follow-up assessment after any corrective work."
        : severity === "MEDIUM"
        ? "Record the findings and follow up after inspection."
        : "Monitor the panel during the next inspection cycle.";
    }

    if (primaryFault === "Physical-damage") {
      if (index === 0) {
        return severity === "HIGH"
          ? "Arrange a qualified physical inspection of the affected panel."
          : severity === "MEDIUM"
          ? "Schedule a physical inspection."
          : "Include the panel in routine physical inspection.";
      }
      if (index === 1) {
        return severity === "HIGH"
          ? "Document the visible damaged area and check for additional visible damage."
          : severity === "MEDIUM"
          ? "Review the visible affected area for additional damage."
          : "Monitor the visible affected area for changes.";
      }
      return severity === "HIGH"
        ? "Perform a follow-up assessment to determine whether further maintenance is required."
        : severity === "MEDIUM"
        ? "Record the findings and follow up after inspection."
        : "Reassess during the next inspection cycle.";
    }

    if (primaryFault === "Bird-drop") {
      if (index === 0) {
        return severity === "HIGH"
          ? "Prioritize cleaning of the affected panel surface."
          : severity === "MEDIUM"
          ? "Clean the affected panel surface."
          : "Clean the affected surface as part of routine maintenance.";
      }
      if (index === 1) {
        return severity === "HIGH"
          ? "Inspect the panel after cleaning for remaining visible contamination or damage."
          : severity === "MEDIUM"
          ? "Inspect the panel after cleaning."
          : "Inspect the panel after cleaning.";
      }
      return severity === "HIGH"
        ? "Record the inspection result and monitor the panel during the next cycle."
        : "Continue routine monitoring.";
    }

    if (primaryFault === "Dusty") {
      if (index === 0) {
        return severity === "HIGH"
          ? "Prioritize cleaning of the affected panel surface."
          : severity === "MEDIUM"
          ? "Schedule panel cleaning."
          : "Include the panel in the routine cleaning schedule.";
      }
      if (index === 1) {
        return severity === "HIGH"
          ? "Inspect the panel after cleaning to confirm visible accumulation has been removed."
          : severity === "MEDIUM"
          ? "Inspect the panel after cleaning."
          : "Check the panel surface after cleaning.";
      }
      return severity === "HIGH"
        ? "Perform a follow-up visual inspection."
        : "Continue routine monitoring.";
    }

    if (primaryFault === "Snow-Covered") {
      if (index === 0) {
        return severity === "HIGH"
          ? "Prioritize safe snow removal using an appropriate procedure."
          : severity === "MEDIUM"
          ? "Schedule safe snow removal."
          : "Monitor the panel until the surface is clear.";
      }
      if (index === 1) {
        return severity === "HIGH"
          ? "Inspect the panel after snow removal for visible damage."
          : severity === "MEDIUM"
          ? "Inspect the panel after the surface is clear."
          : "Inspect the panel after snow removal.";
      }
      return severity === "HIGH"
        ? "Perform a follow-up inspection once the panel surface is clear."
        : "Continue routine monitoring.";
    }

    if (primaryFault === "Clean") {
      if (index === 0) {
        return "Continue routine visual monitoring.";
      }
      if (index === 1) {
        return "Maintain the panel according to the site's normal cleaning schedule.";
      }
      return "Reinspect periodically for newly visible faults.";
    }

    if (index === 0) {
      return "Follow the site maintenance workflow for the affected panel.";
    }
    if (index === 1) {
      return "Inspect the affected area for any additional visible issues.";
    }
    return "Reassess during the next scheduled inspection cycle.";
  };

  if (loading) {
    return (
      <AppShell
        breadcrumbs={[
          { label: "Dashboard", href: "/" },
          { label: "Inspection History", href: "/inspection-history" },
          { label: "Loading…", active: true },
        ]}
      >
        <div className="max-w-6xl mx-auto w-full space-y-6">
          <div className="h-24 w-full animate-pulse rounded-2xl bg-surface-container/70" />
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {Array.from({ length: 3 }).map((_, index) => (
              <div key={index} className="h-40 rounded-2xl bg-surface-container/70 animate-pulse" />
            ))}
          </div>
          <div className="h-72 rounded-2xl bg-surface-container/70 animate-pulse" />
        </div>
      </AppShell>
    );
  }

  if (error) {
    return (
      <AppShell
        breadcrumbs={[
          { label: "Dashboard", href: "/" },
          { label: "Inspection History", href: "/inspection-history" },
          { label: "Result unavailable", active: true },
        ]}
      >
        <div className="max-w-3xl mx-auto w-full">
          <div className="rounded-2xl border border-outline-variant/40 bg-surface-container-lowest p-8 text-center shadow-sm">
            <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-amber-100 text-amber-700">
              <AlertTriangle className="h-6 w-6" />
            </div>
            <h1 className="text-[28px] font-bold text-on-surface tracking-tight">Inspection result could not be loaded.</h1>
            <p className="mt-3 text-[15px] leading-6 text-on-surface-variant">
              Please try again from the inspection history page.
            </p>
            <Link
              href="/inspection-history"
              className="mt-6 inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2.5 text-sm font-semibold text-on-primary shadow-sm transition-colors hover:bg-primary/90"
            >
              <ArrowLeft className="h-4 w-4" />
              Back to Inspection History
            </Link>
          </div>
        </div>
      </AppShell>
    );
  }

  return (
    <AppShell
      breadcrumbs={[
        { label: "Dashboard", href: "/" },
        { label: "Inspection History", href: "/inspection-history" },
        { label: `Inspection #${record.inspection_id}`, active: true },
      ]}
    >
      <div className="flex flex-col gap-6 max-w-6xl mx-auto w-full pb-8">
        <div className="flex flex-col gap-3">
          <div className="flex items-center gap-2 text-[12px] font-semibold uppercase tracking-[0.12em] text-primary">
            Inspection Result
          </div>
          <h1 className="text-[28px] md:text-[32px] font-bold text-on-surface tracking-tight leading-tight">
            {formatFaultName(record.predicted_class)} Detected
          </h1>
          <p className="text-[14px] text-on-surface-variant leading-relaxed">
            AI-assisted visual inspection of the uploaded solar panel image.
          </p>
        </div>

        <section className="rounded-2xl border border-outline-variant/30 bg-surface-container-lowest p-5 shadow-sm">
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            <div className="rounded-xl border border-outline-variant/20 bg-surface p-4">
              <p className="text-[12px] font-semibold uppercase tracking-[0.12em] text-on-surface-variant">
                Detected Fault
              </p>
              <p className="mt-2 text-[22px] font-bold text-on-surface leading-snug">
                {formatFaultName(record.predicted_class)}
              </p>
            </div>
            <div className="rounded-xl border border-outline-variant/20 bg-surface p-4">
              <p className="text-[12px] font-semibold uppercase tracking-[0.12em] text-on-surface-variant">
                Confidence
              </p>
              <p className="mt-2 text-[22px] font-bold text-on-surface leading-snug">
                {confidencePercent.toFixed(1)}%
              </p>
            </div>
            <div className="rounded-xl border border-outline-variant/20 bg-surface p-4">
              <p className="text-[12px] font-semibold uppercase tracking-[0.12em] text-on-surface-variant">
                Visual Severity
              </p>
              <div className="mt-2 flex items-center gap-2">
                <SeverityBadge severity={record.severity} />
              </div>
              <p className="mt-2 text-[13px] leading-5 text-on-surface-variant">
                {severityExplanation}
              </p>
            </div>
            <div className="rounded-xl border border-outline-variant/20 bg-surface p-4">
              <p className="text-[12px] font-semibold uppercase tracking-[0.12em] text-on-surface-variant">
                Panel ID
              </p>
              <p className="mt-2 text-[16px] font-semibold text-on-surface break-all leading-relaxed">
                {record.panel_id || "AUTO-XXX"}
              </p>
            </div>
            <div className="rounded-xl border border-outline-variant/20 bg-surface p-4 md:col-span-2 xl:col-span-2">
              <p className="text-[12px] font-semibold uppercase tracking-[0.12em] text-on-surface-variant">
                Location
              </p>
              <p className="mt-2 text-[16px] font-semibold text-on-surface leading-relaxed">
                {record.location || "Location not specified"}
              </p>
            </div>
          </div>
        </section>

        <section className="rounded-2xl border border-outline-variant/30 bg-surface-container-lowest p-5 shadow-sm">
          <div className="flex items-center justify-between gap-3 pb-3 border-b border-outline-variant/20">
            <h2 className="text-[18px] md:text-[20px] font-semibold text-on-surface tracking-tight">
              AI Confidence
            </h2>
            <span className={`text-[13px] font-semibold ${confidenceTone.color}`}>
              {confidenceTone.label}
            </span>
          </div>

          <div className="mt-5 space-y-4">
            <div className="flex items-end justify-between gap-3">
              <div>
                <div className="text-[28px] md:text-[32px] font-bold text-on-surface leading-none">
                  {confidencePercent.toFixed(1)}%
                </div>
              </div>
              <div className="text-[12px] font-medium uppercase tracking-[0.12em] text-on-surface-variant">
                {confidenceInfo.level === "high" ? "High" : confidenceInfo.level === "medium" ? "Moderate" : "Low"}
              </div>
            </div>

            <div>
              <p className="text-[16px] font-semibold text-on-surface leading-relaxed">
                {confidenceHeadline}
              </p>
              <p className="mt-2 text-[14px] leading-6 text-on-surface-variant">
                Confidence indicates how strongly the AI model supports this predicted fault class. It is not the model’s overall accuracy.
              </p>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between gap-3 text-[12px] font-medium uppercase tracking-[0.12em] text-on-surface-variant">
                <span>Prediction support</span>
                <span>{confidencePercent.toFixed(1)}%</span>
              </div>
              <div className="h-2.5 w-full overflow-hidden rounded-full bg-surface-container">
                <div
                  className={`h-full rounded-full ${confidenceTone.bar}`}
                  style={{ width: `${Math.min(100, Math.max(0, confidencePercent))}%` }}
                  aria-label={`Confidence level ${confidencePercent.toFixed(1)} percent`}
                  role="progressbar"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={Number(confidencePercent.toFixed(1))}
                />
              </div>
            </div>

            <div className="rounded-xl border border-outline-variant/20 bg-surface p-3.5 text-[14px] leading-6 text-on-surface-variant">
              <span className="font-semibold text-on-surface">Confidence = </span>
              how strongly the model supports this particular prediction.
              <span className="font-semibold text-on-surface"> Accuracy = </span>
              how often a model is correct across an evaluated dataset.
            </div>
          </div>
        </section>

        <section className="rounded-2xl border border-outline-variant/30 bg-surface-container-lowest p-5 shadow-sm">
          <div className="mb-4">
            <h2 className="text-[18px] md:text-[20px] font-semibold text-on-surface tracking-tight">
              Visual Evidence
            </h2>
            <p className="mt-1 text-[13px] leading-6 text-on-surface-variant">
              These images help explain where the model focused when producing the prediction.
            </p>
          </div>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
            <div className="flex h-full flex-col overflow-hidden rounded-xl border border-outline-variant/30 bg-surface shadow-sm">
              <div className="flex items-center justify-between border-b border-outline-variant/20 bg-surface-container/70 px-3 py-2.5">
                <span className="text-[13px] font-semibold text-on-surface">Original Image</span>
                <span className="text-[11px] text-on-surface-variant">Submitted image</span>
              </div>
              <div className="flex flex-col gap-3 p-3">
                <p className="text-[13px] leading-5 text-on-surface-variant">
                  The solar panel image submitted for inspection.
                </p>
                <div className="relative h-64 w-full overflow-hidden rounded-lg bg-surface-container/60">
                  {record.original_image_url ? (
                    <Image
                      src={record.original_image_url || FALLBACK_ORIGINAL_IMG}
                      alt="Original solar panel inspection image"
                      width={800}
                      height={600}
                      className="h-full w-full object-cover"
                    />
                  ) : (
                    <div className="flex h-full items-center justify-center px-4 text-center text-[13px] text-on-surface-variant">
                      Visual evidence unavailable
                    </div>
                  )}
                </div>
              </div>
            </div>

            <div className="flex h-full flex-col overflow-hidden rounded-xl border border-outline-variant/30 bg-surface shadow-sm">
              <div className="flex items-center justify-between border-b border-outline-variant/20 bg-surface-container/70 px-3 py-2.5">
                <span className="text-[13px] font-semibold text-on-surface">AI Attention</span>
                <span className="text-[11px] text-primary">Grad-CAM</span>
              </div>
              <div className="flex flex-col gap-3 p-3">
                <p className="text-[13px] leading-5 text-on-surface-variant">
                  Grad-CAM visualization showing areas that influenced the model&apos;s prediction.
                </p>
                <div className="relative h-64 w-full overflow-hidden rounded-lg bg-surface-container/60">
                  {record.gradcam_image_url ? (
                    <Image
                      src={record.gradcam_image_url || FALLBACK_GRADCAM_IMG}
                      alt="Grad-CAM AI attention visualization"
                      width={800}
                      height={600}
                      className="h-full w-full object-cover"
                    />
                  ) : (
                    <div className="flex h-full items-center justify-center px-4 text-center text-[13px] text-on-surface-variant">
                      Visual evidence unavailable
                    </div>
                  )}
                </div>
              </div>
            </div>

            <div className="flex h-full flex-col overflow-hidden rounded-xl border border-outline-variant/30 bg-surface shadow-sm md:col-span-2 xl:col-span-1">
              <div className="flex items-center justify-between border-b border-outline-variant/20 bg-surface-container/70 px-3 py-2.5">
                <span className="text-[13px] font-semibold text-on-surface">Approximate Visual Region</span>
                <span className="text-[11px] text-rose-700">Approximate</span>
              </div>
              <div className="flex flex-col gap-3 p-3">
                <p className="text-[13px] leading-5 text-on-surface-variant">
                  An approximate region highlighted from the AI attention map.
                </p>
                <div className="relative h-64 w-full overflow-hidden rounded-lg bg-surface-container/60">
                  {record.region_image_url ? (
                    <Image
                      src={record.region_image_url || FALLBACK_REGION_IMG}
                      alt="Approximate visual region highlighted by the AI"
                      width={800}
                      height={600}
                      className="h-full w-full object-cover"
                    />
                  ) : (
                    <div className="flex h-full items-center justify-center px-4 text-center text-[13px] text-on-surface-variant">
                      Visual evidence unavailable
                    </div>
                  )}
                </div>
                {regionPercentage !== null ? (
                  <div className="rounded-lg border border-outline-variant/20 bg-surface-container/40 px-2.5 py-1.5 text-[12px] font-medium text-on-surface-variant">
                    Approximate highlighted area: {regionPercentage.toFixed(1)}%
                  </div>
                ) : null}
              </div>
            </div>
          </div>

          <div className="mt-4 space-y-2">
            <p className="text-[13px] leading-6 text-on-surface-variant">
              Visual evidence is approximate. The highlighted region is not a precise defect boundary or measurement.
            </p>
            <p className="text-[13px] leading-6 text-on-surface-variant">
              The attention map helps explain where the model focused when producing the prediction.
            </p>
          </div>
        </section>

        <section className="rounded-2xl border border-outline-variant/30 bg-surface-container-lowest p-5 shadow-sm">
          <h2 className="text-[18px] md:text-[20px] font-semibold text-on-surface tracking-tight">
            How to Read This Result
          </h2>

          <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
            <div className="rounded-xl border border-outline-variant/20 bg-surface p-4">
              <p className="text-[15px] font-semibold text-on-surface">What the AI predicted</p>
              <p className="mt-2 text-[14px] leading-6 text-on-surface-variant">
                The AI identified the most likely fault class from the submitted solar-panel image: {formatFaultName(record.predicted_class)}.
              </p>
            </div>

            <div className="rounded-xl border border-outline-variant/20 bg-surface p-4">
              <p className="text-[15px] font-semibold text-on-surface">How confident the AI is</p>
              <p className="mt-2 text-[14px] leading-6 text-on-surface-variant">
                This percentage shows how strongly the model supports this particular prediction. It is not the model&apos;s overall accuracy.
              </p>
            </div>

            <div className="rounded-xl border border-outline-variant/20 bg-surface p-4">
              <p className="text-[15px] font-semibold text-on-surface">What the AI focused on</p>
              <p className="mt-2 text-[14px] leading-6 text-on-surface-variant">
                The attention visualization shows areas that influenced the model&apos;s prediction. These highlighted areas are approximate and are not exact defect boundaries.
              </p>
            </div>

            <div className="rounded-xl border border-outline-variant/20 bg-surface p-4">
              <p className="text-[15px] font-semibold text-on-surface">What happens next</p>
              <p className="mt-2 text-[14px] leading-6 text-on-surface-variant">
                Severity and urgency are used to guide the recommended inspection or maintenance action shown below.
              </p>
            </div>
          </div>

          <p className="mt-4 text-[13px] leading-6 text-on-surface-variant">
            AI-assisted visual assessment — final maintenance decisions should be made by a qualified person when required.
          </p>
        </section>

        <section className="rounded-2xl border border-outline-variant/30 bg-surface-container-lowest p-5 shadow-sm">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <h2 className="text-[18px] md:text-[20px] font-semibold text-on-surface tracking-tight">
              Recommended Action
            </h2>
            <UrgencyBadge urgency={record.urgency} />
          </div>

          <div className="mt-4 space-y-3">
            {maintenanceActions.map((action, index) => {
              const actionLabel =
                index === 0 ? "Primary action" : index === 1 ? "Secondary action" : "Follow-up action";

              return (
                <div
                  key={`${action}-${index}`}
                  className="flex gap-3 rounded-xl border border-outline-variant/20 bg-surface p-3.5"
                >
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-[13px] font-bold text-primary">
                    {String(index + 1).padStart(2, "0")}
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="text-[11px] font-semibold uppercase tracking-[0.12em] text-on-surface-variant">
                      {actionLabel}
                    </div>
                    <p className="mt-1 text-[15px] font-semibold text-on-surface leading-relaxed break-words">
                      {action}
                    </p>
                    <p className="mt-1 text-[14px] leading-6 text-on-surface-variant">
                      {getActionDescription(index)}
                    </p>
                  </div>
                </div>
              );
            })}
          </div>

          <p className="mt-4 text-[13px] leading-6 text-on-surface-variant">
            AI-assisted workflow guidance. This is not a certified engineering diagnosis.
          </p>
        </section>

        {record.manual_inspection_recommended === undefined || record.manual_inspection_recommended === null ? (
          <section className="rounded-2xl border border-outline-variant/30 bg-surface-container-lowest p-4 shadow-sm">
            <div className="flex items-start gap-3">
              <div className="mt-0.5 h-5 w-5 shrink-0 rounded-full border border-outline-variant/50 bg-surface" />
              <div>
                <h2 className="text-[18px] font-semibold text-on-surface">Manual review status unavailable</h2>
                <p className="mt-1 text-[14px] leading-6 text-on-surface-variant">
                  Please refer to the inspection details or retry the inspection.
                </p>
              </div>
            </div>
          </section>
        ) : record.manual_inspection_recommended ? (
          <section className="rounded-2xl border border-amber-200 bg-amber-50 p-4 text-amber-900 shadow-sm">
            <div className="flex items-start gap-3">
              <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-amber-600" />
              <div>
                <h2 className="text-[18px] font-semibold">Manual review recommended</h2>
                <p className="mt-1 text-[14px] leading-6 text-amber-800">
                  This inspection result should be reviewed by a qualified person before maintenance or repair decisions are made.
                </p>
                {record.confidence_warning ? (
                  <p className="mt-2 text-[14px] leading-6 text-amber-800">
                    {record.confidence_warning}
                  </p>
                ) : null}
              </div>
            </div>
          </section>
        ) : (
          <section className="rounded-2xl border border-outline-variant/30 bg-surface-container-lowest p-4 shadow-sm">
            <div className="flex items-start gap-3">
              <div className="mt-0.5 h-5 w-5 shrink-0 rounded-full border border-outline-variant/50 bg-surface" />
              <div>
                <h2 className="text-[18px] font-semibold text-on-surface">Manual review not specifically flagged</h2>
                <p className="mt-1 text-[14px] leading-6 text-on-surface-variant">
                  The current inspection rules did not flag this result for additional manual review.
                </p>
              </div>
            </div>
          </section>
        )}

        <details className="group rounded-2xl border border-outline-variant/30 bg-surface-container-lowest p-5 shadow-sm" open>
          <summary className="cursor-pointer list-none text-[18px] md:text-[20px] font-semibold text-on-surface tracking-tight">
            Inspection Details
          </summary>

          <div className="mt-5 space-y-4">
            <div>
              <h3 className="text-[15px] font-semibold text-on-surface">Inspection</h3>
              <div className="mt-3 grid gap-3 sm:grid-cols-2">
                <div className="rounded-xl border border-outline-variant/20 bg-surface p-3.5">
                  <p className="text-[12px] uppercase tracking-[0.12em] text-on-surface-variant">Inspection ID</p>
                  <p className="mt-2 text-[15px] font-medium text-on-surface break-all">
                    {record.inspection_id ? String(record.inspection_id) : "Not provided"}
                  </p>
                </div>
                <div className="rounded-xl border border-outline-variant/20 bg-surface p-3.5">
                  <p className="text-[12px] uppercase tracking-[0.12em] text-on-surface-variant">Inspection date/time</p>
                  <p className="mt-2 text-[15px] font-medium text-on-surface break-words">
                    {record.inspection_timestamp ? formatInspectionDate(record.inspection_timestamp) : "Not provided"}
                  </p>
                </div>
                <div className="rounded-xl border border-outline-variant/20 bg-surface p-3.5 sm:col-span-2">
                  <p className="text-[12px] uppercase tracking-[0.12em] text-on-surface-variant">Image file</p>
                  <p className="mt-2 text-[15px] font-medium text-on-surface break-all">
                    {record.image_filename || "Not provided"}
                  </p>
                </div>
              </div>
            </div>

            <div>
              <h3 className="text-[15px] font-semibold text-on-surface">Panel</h3>
              <div className="mt-3 grid gap-3 sm:grid-cols-2">
                <div className="rounded-xl border border-outline-variant/20 bg-surface p-3.5">
                  <p className="text-[12px] uppercase tracking-[0.12em] text-on-surface-variant">Panel ID</p>
                  <p className="mt-2 text-[15px] font-medium text-on-surface break-all">
                    {record.panel_id || "Not provided"}
                  </p>
                </div>
                <div className="rounded-xl border border-outline-variant/20 bg-surface p-3.5">
                  <p className="text-[12px] uppercase tracking-[0.12em] text-on-surface-variant">Location</p>
                  <p className="mt-2 text-[15px] font-medium text-on-surface break-words">
                    {record.location || "Not provided"}
                  </p>
                </div>
              </div>
            </div>

            <div>
              <h3 className="text-[15px] font-semibold text-on-surface">AI Result</h3>
              <div className="mt-3 grid gap-3 sm:grid-cols-2">
                <div className="rounded-xl border border-outline-variant/20 bg-surface p-3.5">
                  <p className="text-[12px] uppercase tracking-[0.12em] text-on-surface-variant">Predicted fault</p>
                  <p className="mt-2 text-[15px] font-medium text-on-surface break-words">
                    {record.predicted_class ? formatFaultName(record.predicted_class) : "Not provided"}
                  </p>
                </div>
                <div className="rounded-xl border border-outline-variant/20 bg-surface p-3.5">
                  <p className="text-[12px] uppercase tracking-[0.12em] text-on-surface-variant">AI confidence</p>
                  <p className="mt-2 text-[15px] font-medium text-on-surface">
                    {Number.isFinite(confidencePercent) ? `${confidencePercent.toFixed(1)}%` : "Not provided"}
                  </p>
                </div>
                <div className="rounded-xl border border-outline-variant/20 bg-surface p-3.5">
                  <p className="text-[12px] uppercase tracking-[0.12em] text-on-surface-variant">Severity</p>
                  <p className="mt-2 text-[15px] font-medium text-on-surface">
                    {record.severity || "Not provided"}
                  </p>
                </div>
                <div className="rounded-xl border border-outline-variant/20 bg-surface p-3.5">
                  <p className="text-[12px] uppercase tracking-[0.12em] text-on-surface-variant">Urgency</p>
                  <p className="mt-2 text-[15px] font-medium text-on-surface">
                    {record.urgency || "Not provided"}
                  </p>
                </div>
              </div>
            </div>

            {(typeof record.visual_region_area_percent === "number" ||
              typeof record.manual_inspection_recommended === "boolean" ||
              record.confidence_warning) && (
              <div>
                <h3 className="text-[15px] font-semibold text-on-surface">AI-Assisted Assessment</h3>
                <div className="mt-3 grid gap-3 sm:grid-cols-2">
                  {typeof record.visual_region_area_percent === "number" && Number.isFinite(record.visual_region_area_percent) ? (
                    <div className="rounded-xl border border-outline-variant/20 bg-surface p-3.5">
                      <p className="text-[12px] uppercase tracking-[0.12em] text-on-surface-variant">Approximate highlighted area</p>
                      <p className="mt-2 text-[15px] font-medium text-on-surface">
                        {record.visual_region_area_percent.toFixed(1)}%
                      </p>
                    </div>
                  ) : null}
                  {typeof record.manual_inspection_recommended === "boolean" ? (
                    <div className="rounded-xl border border-outline-variant/20 bg-surface p-3.5">
                      <p className="text-[12px] uppercase tracking-[0.12em] text-on-surface-variant">Manual review</p>
                      <p className="mt-2 text-[15px] font-medium text-on-surface">
                        {record.manual_inspection_recommended ? "Recommended" : "Not specifically flagged"}
                      </p>
                    </div>
                  ) : null}
                  {record.confidence_warning ? (
                    <div className="rounded-xl border border-outline-variant/20 bg-surface p-3.5 sm:col-span-2">
                      <p className="text-[12px] uppercase tracking-[0.12em] text-on-surface-variant">Confidence warning</p>
                      <p className="mt-2 text-[15px] font-medium text-on-surface break-words">
                        {record.confidence_warning}
                      </p>
                    </div>
                  ) : null}
                </div>
              </div>
            )}
          </div>
        </details>

        <div className="pt-2">
          <div className="flex flex-col gap-3 rounded-2xl border border-outline-variant/30 bg-surface-container-lowest p-3.5 shadow-sm sm:flex-row sm:items-center sm:justify-between">
            <div className="flex flex-col gap-2 sm:flex-row sm:flex-wrap">
              <Link
                href="/inspection-history"
                className="inline-flex items-center justify-center rounded-lg border border-outline-variant/30 bg-surface px-3.5 py-2.5 text-sm font-medium text-on-surface transition-colors hover:bg-surface-container"
              >
                <ArrowLeft className="mr-2 h-4 w-4" />
                Back to Inspection History
              </Link>

              <Link
                href="/new-inspection"
                className="inline-flex items-center justify-center rounded-lg bg-primary px-3.5 py-2.5 text-sm font-semibold text-on-primary transition-colors hover:bg-primary/90"
              >
                New Inspection
              </Link>

              {record.panel_id && !record.panel_id.toUpperCase().startsWith("AUTO-") ? (
                <Link
                  href={`/panel-details/${encodeURIComponent(record.panel_id)}`}
                  className="inline-flex items-center justify-center rounded-lg border border-outline-variant/30 bg-surface px-3.5 py-2.5 text-sm font-medium text-on-surface transition-colors hover:bg-surface-container"
                >
                  View Panel Details
                </Link>
              ) : null}
            </div>
          </div>
        </div>

        <Disclaimer />
      </div>
    </AppShell>
  );
}
