import React from "react";

interface DisclaimerProps {
  className?: string;
  variant?: "full" | "compact";
}

export function Disclaimer({ className = "", variant = "full" }: DisclaimerProps) {
  if (variant === "compact") {
    return (
      <div
        className={`bg-surface-container-low px-space-sm py-1.5 rounded border border-outline-variant/30 flex items-center gap-2 text-on-surface-variant font-body-sm text-body-sm ${className}`}
      >
        <span className="material-symbols-outlined text-outline text-base shrink-0">
          policy
        </span>
        <p className="leading-tight text-[11px]">
          AI-assisted visual assessment for inspection workflow support. Severity and
          maintenance recommendations are not a certified engineering diagnosis and
          do not directly measure electrical power loss, temperature, crack depth,
          or structural integrity.
        </p>
      </div>
    );
  }

  return (
    <div
      className={`bg-surface-container-low p-space-md rounded-lg shadow-sm border border-outline-variant/30 flex items-start gap-space-md ${className}`}
    >
      <span className="material-symbols-outlined text-outline text-xl shrink-0 mt-0.5">
        policy
      </span>
      <div className="flex flex-col gap-0.5">
        <span className="font-label-caps text-label-caps text-on-surface font-bold uppercase tracking-wider">
          Statutory Diagnostic Disclaimer • ISO/IEC 17025 Conformity Note
        </span>
        <p className="font-body-sm text-body-sm text-on-surface-variant leading-normal">
          AI-assisted visual assessment for inspection workflow support. Severity and
          maintenance recommendations are not a certified engineering diagnosis and
          do not directly measure electrical power loss, temperature, crack depth, or
          structural integrity. Physical dispatch and manual electrical testing by
          qualified personnel are necessary prior to undertaking warranty claims or
          hardware decommission.
        </p>
      </div>
    </div>
  );
}
