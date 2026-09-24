import React from "react";
import { SeverityLevel, UrgencyLevel } from "@/types/inspection";

interface SeverityBadgeProps {
  severity: SeverityLevel | string;
  className?: string;
  showEstimateHint?: boolean;
}

export function SeverityBadge({
  severity,
  className = "",
  showEstimateHint = false,
}: SeverityBadgeProps) {
  const norm = (severity || "").toUpperCase();

  let badgeStyle = "bg-surface-container text-on-surface-variant border-outline-variant/40";
  let label = "LOW";

  if (norm === "HIGH" || norm === "CRITICAL") {
    badgeStyle = "bg-error-container text-on-error-container border-error/30 font-semibold";
    label = "HIGH";
  } else if (norm === "MEDIUM") {
    badgeStyle = "bg-amber-100 text-amber-900 border-amber-300 font-medium";
    label = "MEDIUM";
  } else {
    badgeStyle = "bg-emerald-50 text-emerald-800 border-emerald-200 font-medium";
    label = "LOW";
  }

  return (
    <span
      className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs border tracking-wide uppercase ${badgeStyle} ${className}`}
      title={showEstimateHint ? "AI-assisted visual severity estimate" : undefined}
    >
      <span>{label}</span>
      {showEstimateHint && (
        <span className="text-[10px] opacity-75 font-normal lowercase">(est.)</span>
      )}
    </span>
  );
}

interface UrgencyBadgeProps {
  urgency: UrgencyLevel | string;
  className?: string;
}

export function UrgencyBadge({ urgency, className = "" }: UrgencyBadgeProps) {
  const norm = (urgency || "").toUpperCase();

  switch (norm) {
    case "IMMEDIATE REVIEW":
    case "IMMEDIATE_REVIEW":
    case "PRIORITY":
      return (
        <span
          className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-red-100 text-red-800 border border-red-200 tracking-wide uppercase ${className}`}
        >
          PRIORITY
        </span>
      );
    case "MANUAL REVIEW":
      return (
        <span
          className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-orange-100 text-orange-800 border border-orange-200 tracking-wide uppercase ${className}`}
        >
          MANUAL REVIEW
        </span>
      );
    case "SCHEDULED":
      return (
        <span
          className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-50 text-blue-800 border border-blue-200 tracking-wide uppercase ${className}`}
        >
          SCHEDULED
        </span>
      );
    case "ROUTINE":
    default:
      return (
        <span
          className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-slate-100 text-slate-700 border border-slate-200 tracking-wide uppercase ${className}`}
        >
          ROUTINE
        </span>
      );
  }
}
