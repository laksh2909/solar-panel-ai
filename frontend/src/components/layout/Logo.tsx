import React from "react";

export function Logo({ className = "h-8 w-auto" }: { className?: string }) {
  return (
    <div className="flex items-center gap-2">
      <svg
        viewBox="0 0 160 40"
        className={className}
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <rect
          x="2"
          y="6"
          width="28"
          height="28"
          rx="6"
          fill="#0284C7"
          fillOpacity="0.1"
          stroke="#0284C7"
          strokeWidth="1.5"
        />
        <path
          d="M7 15H25M7 25H25M16 6V34"
          stroke="#0284C7"
          strokeWidth="1.5"
          strokeLinecap="round"
        />
        <circle cx="21" cy="11" r="3" fill="#059669" />
        <text
          x="38"
          y="24"
          fontFamily="system-ui, -apple-system, sans-serif"
          fontSize="15"
          fontWeight="700"
          fill="#0F172A"
          letterSpacing="-0.3px"
        >
          SOLAR<tspan fill="#0284C7">SCAN</tspan>
          <tspan fill="#64748B" fontSize="11" fontWeight="500">
            {" "}
            AI
          </tspan>
        </text>
        <text
          x="38"
          y="33"
          fontFamily="system-ui, -apple-system, sans-serif"
          fontSize="8"
          fontWeight="600"
          fill="#94A3B8"
          letterSpacing="0.8px"
        >
          INSPECTION SUITE
        </text>
      </svg>
    </div>
  );
}
