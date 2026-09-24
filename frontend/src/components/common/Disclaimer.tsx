import React from "react";
import { Info, ShieldAlert } from "lucide-react";

interface DisclaimerProps {
  className?: string;
  variant?: "full" | "compact";
}

export function Disclaimer({ className = "", variant = "full" }: DisclaimerProps) {
  if (variant === "compact") {
    return (
      <div
        className={`bg-surface-container-low px-3 py-2 rounded-lg border border-outline-variant/30 flex items-center gap-2 text-on-surface-variant font-body-sm text-xs ${className}`}
      >
        <Info className="w-4 h-4 text-outline shrink-0" />
        <p className="leading-relaxed">
          AI-assisted guidance only. This result is not a certified engineering diagnosis.
        </p>
      </div>
    );
  }

  return (
    <div
      className={`bg-surface-container-low/70 p-4 rounded-lg border border-outline-variant/30 flex items-start gap-3 ${className}`}
    >
      <ShieldAlert className="w-5 h-5 text-outline shrink-0 mt-0.5" />
      <div className="flex flex-col gap-0.5">
        <span className="font-semibold text-xs text-on-surface">
          Advisory Disclaimer
        </span>
        <p className="font-body-sm text-xs text-on-surface-variant leading-relaxed">
          AI-assisted guidance only. This result is not a certified engineering diagnosis and does not directly measure electrical power output, temperature, or structural integrity. Physical inspection by a qualified technician is recommended before performing maintenance.
        </p>
      </div>
    </div>
  );
}
