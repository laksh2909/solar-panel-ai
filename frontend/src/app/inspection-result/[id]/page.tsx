"use client";

import React, { useState, useEffect, use } from "react";
import Link from "next/link";
import {
  ArrowLeft,
  Camera,
  AlertTriangle,
  Download,
  ExternalLink,
  Eye,
  Crosshair,
} from "lucide-react";
import { AppShell } from "@/components/layout/AppShell";
import { SeverityBadge, UrgencyBadge } from "@/components/common/StatusBadge";
import { Disclaimer } from "@/components/common/Disclaimer";
import { useToast } from "@/components/common/Toast";
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

  const { showToast } = useToast();
  const [inspection, setInspection] = useState<InspectionResponse | null>(null);

  useEffect(() => {
    let isMounted = true;
    async function loadData() {
      try {
        const data = await fetchInspection(inspectionId);
        if (isMounted && data) {
          setInspection(data);
        }
      } catch (err) {
        console.error("Failed to load inspection:", err);
      }
    }
    loadData();
    return () => {
      isMounted = false;
    };
  }, [inspectionId]);

  const record = inspection || MOCK_INSPECTIONS[0];
  const confidencePercent = record.confidence <= 1 ? record.confidence * 100 : record.confidence;
  const confidenceInfo = getConfidenceFeedback(record.confidence);
  const severityExplanation = getSeverityFeedback(record.severity);

  return (
    <AppShell
      breadcrumbs={[
        { label: "Dashboard", href: "/" },
        { label: "Inspection History", href: "/inspection-history" },
        { label: `Inspection #${record.inspection_id}`, active: true },
      ]}
    >
      <div className="flex flex-col gap-6 max-w-6xl mx-auto w-full">
        {/* Navigation & Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs font-semibold text-primary uppercase tracking-wider">
                Inspection Result
              </span>
              <span className="text-xs text-on-surface-variant font-medium">
                #{record.inspection_id}
              </span>
            </div>
            <h1 className="text-2xl font-bold text-on-surface tracking-tight">
              {formatFaultName(record.predicted_class)} Detected
            </h1>
          </div>

          <div className="flex items-center gap-3">
            <Link
              href="/inspection-history"
              className="inline-flex items-center gap-1.5 px-3.5 py-2 bg-surface border border-outline-variant/30 hover:bg-surface-container text-on-surface text-xs font-semibold rounded-lg shadow-sm transition-colors"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>All History</span>
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

        {/* Confidence Warning (if applicable) */}
        {(record.confidence_warning || record.confidence < 0.6) && (
          <div className="p-4 bg-amber-50 text-amber-900 rounded-xl border border-amber-200 flex items-start gap-3 text-sm">
            <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
            <div>
              <span className="font-semibold block">
                Low Confidence Warning — Manual Review Recommended
              </span>
              <p className="text-xs text-amber-800 mt-0.5">
                {record.confidence_warning ||
                  "Model confidence is below 60%. A physical on-site check is advised before taking maintenance action."}
              </p>
            </div>
          </div>
        )}

        {/* Primary Overview Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Card 1: Detected Fault */}
          <div className="bg-surface-container-lowest p-5 rounded-xl border border-outline-variant/30 shadow-sm flex flex-col justify-between">
            <span className="text-xs font-semibold text-on-surface-variant uppercase tracking-wider">
              Detected Fault
            </span>
            <div className="mt-3">
              <div className="text-xl font-bold text-on-surface">
                {formatFaultName(record.predicted_class)}
              </div>
              <p className="text-xs text-on-surface-variant mt-1">
                Visual fault classification
              </p>
            </div>
            <div className="mt-3 pt-3 border-t border-outline-variant/20 flex items-center justify-between text-xs">
              <span className="text-on-surface-variant">Urgency:</span>
              <UrgencyBadge urgency={record.urgency} />
            </div>
          </div>

          {/* Card 2: Confidence */}
          <div className="bg-surface-container-lowest p-5 rounded-xl border border-outline-variant/30 shadow-sm flex flex-col justify-between">
            <span className="text-xs font-semibold text-on-surface-variant uppercase tracking-wider">
              Confidence
            </span>
            <div className="mt-3">
              <div className="text-3xl font-bold text-on-surface tracking-tight">
                {confidencePercent.toFixed(1)}%
              </div>
              <p
                className={`text-xs font-medium mt-1 ${
                  confidenceInfo.level === "high"
                    ? "text-emerald-700"
                    : confidenceInfo.level === "medium"
                    ? "text-amber-700"
                    : "text-rose-700"
                }`}
              >
                {confidenceInfo.text}
              </p>
            </div>
            {/* Confidence Progress Bar */}
            <div className="mt-3 pt-3 border-t border-outline-variant/20">
              <div className="w-full bg-surface-container h-2 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all ${
                    confidenceInfo.level === "high"
                      ? "bg-emerald-500"
                      : confidenceInfo.level === "medium"
                      ? "bg-amber-500"
                      : "bg-rose-500"
                  }`}
                  style={{ width: `${Math.min(100, Math.max(5, confidencePercent))}%` }}
                />
              </div>
            </div>
          </div>

          {/* Card 3: Visual Severity */}
          <div className="bg-surface-container-lowest p-5 rounded-xl border border-outline-variant/30 shadow-sm flex flex-col justify-between">
            <span className="text-xs font-semibold text-on-surface-variant uppercase tracking-wider">
              Visual Severity
            </span>
            <div className="mt-3">
              <div className="flex items-center gap-2">
                <SeverityBadge severity={record.severity} />
                <span className="text-xs text-on-surface-variant">
                  (~{record.visual_region_area_percent?.toFixed(1) ?? "14.8"}% area)
                </span>
              </div>
              <p className="text-xs text-on-surface-variant mt-2 leading-relaxed">
                {severityExplanation}
              </p>
            </div>
            <div className="mt-3 pt-3 border-t border-outline-variant/20 text-[11px] text-on-surface-variant">
              AI-assisted visual estimate only
            </div>
          </div>

          {/* Card 4: Panel & Location */}
          <div className="bg-surface-container-lowest p-5 rounded-xl border border-outline-variant/30 shadow-sm flex flex-col justify-between">
            <span className="text-xs font-semibold text-on-surface-variant uppercase tracking-wider">
              Panel Information
            </span>
            <div className="mt-3">
              <div className="flex items-center gap-2 flex-wrap">
                <Link
                  href={`/panel-details/${record.panel_id}`}
                  className="text-lg font-bold text-primary hover:underline inline-flex items-center gap-1.5"
                >
                  <span>{record.panel_id}</span>
                  <ExternalLink className="w-3.5 h-3.5" />
                </Link>
                {String(record.panel_id).startsWith("AUTO-") && (
                  <span className="text-[10px] font-semibold bg-amber-50 text-amber-700 border border-amber-200 px-1.5 py-0.5 rounded">
                    Auto-assigned
                  </span>
                )}
              </div>
              <p className="text-xs text-on-surface-variant mt-1">
                {record.location}
              </p>
              {String(record.panel_id).startsWith("AUTO-") && (
                <p className="text-[10px] text-on-surface-variant mt-1 leading-relaxed">
                  Automatically assigned — no Panel ID was provided.
                </p>
              )}
            </div>
            <div className="mt-3 pt-3 border-t border-outline-variant/20 text-xs text-on-surface-variant">
              Inspected: {formatInspectionDate(record.inspection_timestamp)}
            </div>
          </div>

        </div>

        {/* Three Visual Panels: Original, AI Attention, Approximate Visual Region */}
        <div className="flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-base font-semibold text-on-surface">
                Visual Inspection Analysis
              </h2>
              <p className="text-xs text-on-surface-variant mt-0.5">
                Comparison of input capture, attention heatmap, and estimated visual region
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Panel 1: Original Image */}
            <div className="bg-surface-container-lowest rounded-xl border border-outline-variant/30 shadow-sm overflow-hidden flex flex-col">
              <div className="p-3.5 bg-surface-container-low/50 border-b border-outline-variant/20 flex items-center justify-between">
                <span className="text-xs font-semibold text-on-surface">
                  1. Original Image
                </span>
                <span className="text-[11px] text-on-surface-variant">
                  RGB Optical
                </span>
              </div>
              <div className="relative h-64 w-full bg-slate-900/5 flex items-center justify-center overflow-hidden">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={record.original_image_url || FALLBACK_ORIGINAL_IMG}
                  alt="Original solar panel capture"
                  className="w-full h-full object-cover"
                />
              </div>
              <div className="p-3 text-xs text-on-surface-variant bg-surface-container-lowest border-t border-outline-variant/20">
                Original RGB image of the solar panel.
              </div>
            </div>

            {/* Panel 2: AI Attention */}
            <div className="bg-surface-container-lowest rounded-xl border border-outline-variant/30 shadow-sm overflow-hidden flex flex-col">
              <div className="p-3.5 bg-surface-container-low/50 border-b border-outline-variant/20 flex items-center justify-between">
                <span className="text-xs font-semibold text-on-surface flex items-center gap-1.5">
                  <Eye className="w-3.5 h-3.5 text-primary" />
                  <span>2. AI Attention</span>
                </span>
                <span className="text-[11px] text-primary font-medium">
                  Grad-CAM
                </span>
              </div>
              <div className="relative h-64 w-full bg-slate-900/5 flex items-center justify-center overflow-hidden">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={record.gradcam_image_url || FALLBACK_GRADCAM_IMG}
                  alt="AI attention heatmap"
                  className="w-full h-full object-cover"
                />
              </div>
              <div className="p-3 text-xs text-on-surface-variant bg-surface-container-lowest border-t border-outline-variant/20">
                Highlights the image areas that influenced the model&apos;s prediction.
              </div>
            </div>

            {/* Panel 3: Approximate Visual Region */}
            <div className="bg-surface-container-lowest rounded-xl border border-outline-variant/30 shadow-sm overflow-hidden flex flex-col">
              <div className="p-3.5 bg-surface-container-low/50 border-b border-outline-variant/20 flex items-center justify-between">
                <span className="text-xs font-semibold text-on-surface flex items-center gap-1.5">
                  <Crosshair className="w-3.5 h-3.5 text-rose-600" />
                  <span>3. Approximate Visual Region</span>
                </span>
                <span className="text-[11px] text-rose-700 bg-rose-50 px-1.5 py-0.5 rounded font-medium border border-rose-200">
                  Estimate
                </span>
              </div>
              <div className="relative h-64 w-full bg-slate-900/5 flex items-center justify-center overflow-hidden">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={record.region_image_url || FALLBACK_REGION_IMG}
                  alt="Approximate visual region with bounding box"
                  className="w-full h-full object-cover"
                />
                {/* Clarification banner */}
                <div className="absolute bottom-2 left-2 right-2 bg-slate-900/85 backdrop-blur-sm text-white px-2.5 py-1.5 rounded-lg text-center text-[11px]">
                  Estimated region only — not an exact defect boundary
                </div>
              </div>
              <div className="p-3 text-xs text-on-surface-variant bg-surface-container-lowest border-t border-outline-variant/20">
                Shows an estimated visual region based on model attention. This is not an exact defect boundary.
              </div>
            </div>
          </div>
        </div>

        {/* Recommended Action Card */}
        <div className="bg-surface-container-lowest p-6 rounded-xl border border-outline-variant/30 shadow-sm flex flex-col gap-4">
          <div className="flex items-center justify-between pb-3 border-b border-outline-variant/20">
            <h2 className="text-base font-semibold text-on-surface">
              Recommended Action
            </h2>
            <UrgencyBadge urgency={record.urgency} />
          </div>

          <div className="bg-surface-container-low/60 p-4 rounded-xl border border-outline-variant/20">
            <p className="text-base font-semibold text-on-surface">
              {record.maintenance_action || "Continue routine visual monitoring."}
            </p>
            <p className="text-xs text-on-surface-variant mt-1.5 leading-relaxed">
              {record.predicted_class === "Electrical-damage" || record.predicted_class === "Physical-damage"
                ? "Schedule inspection by a qualified technician to verify module connections and cell integrity."
                : record.predicted_class === "Clean"
                ? "Panel surface is clean and clear of noticeable obstructions. No immediate maintenance is required."
                : "Schedule surface cleaning and visual re-inspection to restore optimal solar irradiation."}
            </p>
          </div>

          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pt-2 text-xs text-on-surface-variant">
            <span>
              Recommendation generated based on detected class ({formatFaultName(record.predicted_class)}) and visual severity ({record.severity || "LOW"}).
            </span>
            <button
              type="button"
              onClick={() => showToast("Inspection summary exported.")}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-surface border border-outline-variant/30 hover:bg-surface-container text-on-surface font-medium transition-colors shrink-0"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Export Summary</span>
            </button>
          </div>
        </div>

        {/* Disclaimer */}
        <Disclaimer />
      </div>
    </AppShell>
  );
}
