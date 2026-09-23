import React from "react";
import { SeverityLevel, UrgencyLevel } from "@/types/inspection";

interface SeverityBadgeProps {
  severity: SeverityLevel | string;
  className?: string;
}

export function SeverityBadge({ severity, className = "" }: SeverityBadgeProps) {
  const norm = (severity || "").toUpperCase();

  switch (norm) {
    case "HIGH":
    case "CRITICAL":
      return (
        <span
          className={`inline-block px-2 py-0.5 rounded font-label-caps text-label-caps font-bold bg-error-container text-on-error-container border border-error/20 ${className}`}
        >
          HIGH
        </span>
      );
    case "MEDIUM":
      return (
        <span
          className={`inline-block px-2 py-0.5 rounded font-label-caps text-label-caps font-bold bg-surface-container-high text-on-surface border border-outline-variant/40 ${className}`}
        >
          MEDIUM
        </span>
      );
    case "LOW":
    default:
      return (
        <span
          className={`inline-block px-2 py-0.5 rounded font-label-caps text-label-caps font-semibold bg-surface-container text-on-surface-variant border border-outline-variant/30 ${className}`}
        >
          LOW
        </span>
      );
  }
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
      return (
        <span
          className={`inline-block px-2 py-0.5 rounded font-label-caps text-label-caps font-bold bg-error text-on-error shadow-sm ${className}`}
        >
          IMMEDIATE REVIEW
        </span>
      );
    case "PRIORITY":
      return (
        <span
          className={`inline-block px-2 py-0.5 rounded font-label-caps text-label-caps font-bold bg-tertiary-container text-on-tertiary-container border border-primary/20 ${className}`}
        >
          PRIORITY
        </span>
      );
    case "SCHEDULED":
      return (
        <span
          className={`inline-block px-2 py-0.5 rounded font-label-caps text-label-caps font-semibold bg-surface-variant text-on-surface-variant border border-outline-variant/30 ${className}`}
        >
          SCHEDULED
        </span>
      );
    case "ROUTINE":
    default:
      return (
        <span
          className={`inline-block px-2 py-0.5 rounded font-label-caps text-label-caps font-semibold bg-surface-container-high text-on-surface-variant border border-outline-variant/30 ${className}`}
        >
          ROUTINE
        </span>
      );
  }
}
