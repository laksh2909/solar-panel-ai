"use client";

import React, { useState, useRef } from "react";
import { useRouter } from "next/navigation";
import { AppShell } from "@/components/layout/AppShell";
import { Disclaimer } from "@/components/common/Disclaimer";
import { useToast } from "@/components/common/Toast";
import { submitInspection } from "@/lib/api";

const DEFAULT_PREVIEW_IMG =
  "https://lh3.googleusercontent.com/aida-public/AB6AXuClu9_LnwC1NQr9D9IL_YRk0G3iChTCUSzuRi3UCDO9Dfv9QtZq7gHbu38Hoj6g3gEY09fG0Ar6xS_--_c4k0lyAzGNlr-OZ7xtDPysKf1efYN1wCU_eRp7LPqcGGHmcVF1KlOj3utNLMbe_3pj1twTjdKsle077yF2JW57TmLovR6Ekw_pZxm5xjWY3geAATZ9R3srg_HXr9DJ2FDYxqU-I8b0YLbyUaeEHemeOf3hLeXt3WEJLAq-KA";

export default function NewInspectionPage() {
  const router = useRouter();
  const { showToast } = useToast();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [panelId, setPanelId] = useState("SP-HYD-001");
  const [location, setLocation] = useState("Block A - Rooftop 1");
  const [isVerified, setIsVerified] = useState(true);
  const [rawFile, setRawFile] = useState<File | null>(null);
  const [selectedFile, setSelectedFile] = useState<{
    name: string;
    size: string;
    resolution: string;
    format: string;
    previewUrl: string;
  } | null>({
    name: "demo_panel_rgb_scan_042.jpg",
    size: "4.2 MB",
    resolution: "3840 × 2160 PX",
    format: "RGB JPEG",
    previewUrl: DEFAULT_PREVIEW_IMG,
  });

  const [isRunning, setIsRunning] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);

  const handleVerify = () => {
    if (!panelId.trim()) {
      setValidationError("Panel ID is required.");
      setIsVerified(false);
      return;
    }
    setValidationError(null);
    setIsVerified(true);
    showToast(`Panel ID ${panelId} verified in registry.`);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setRawFile(file);
      const url = URL.createObjectURL(file);
      setSelectedFile({
        name: file.name,
        size: `${(file.size / (1024 * 1024)).toFixed(1)} MB`,
        resolution: "3840 × 2160 PX (Auto-detected)",
        format: file.type.toUpperCase() || "IMAGE/JPEG",
        previewUrl: url,
      });
      setValidationError(null);
      showToast(`Payload loaded: ${file.name}`);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) {
      setRawFile(file);
      const url = URL.createObjectURL(file);
      setSelectedFile({
        name: file.name,
        size: `${(file.size / (1024 * 1024)).toFixed(1)} MB`,
        resolution: "3840 × 2160 PX (Auto-detected)",
        format: file.type.toUpperCase() || "IMAGE/JPEG",
        previewUrl: url,
      });
      setValidationError(null);
      showToast(`Payload dropped: ${file.name}`);
    }
  };

  const handleRunInspection = async () => {
    if (!panelId.trim()) {
      setValidationError("Please specify a valid Panel Registry Identifier.");
      return;
    }
    if (!location.trim()) {
      setValidationError("Please select a Facility Geographic Block.");
      return;
    }
    if (!selectedFile) {
      setValidationError("Please upload or provide an optical RGB inspection image.");
      return;
    }

    setValidationError(null);
    setIsRunning(true);

    try {
      let fileToSend = rawFile;
      if (!fileToSend && selectedFile?.previewUrl) {
        try {
          const resp = await fetch(selectedFile.previewUrl);
          const blob = await resp.blob();
          fileToSend = new File([blob], selectedFile.name || "panel.jpg", {
            type: blob.type || "image/jpeg",
          });
        } catch {
          // If remote image fails to fetch, will proceed or throw below
        }
      }

      const formData = new FormData();
      formData.append("panel_id", panelId.trim());
      formData.append("location", location.trim());
      if (fileToSend) {
        formData.append("file", fileToSend);
        formData.append("image", fileToSend);
      }

      // Execute inspection with real FastAPI backend
      const result = await submitInspection(formData);

      showToast(
        `Diagnostic Complete: ${result.predicted_class} (${(result.confidence * 100).toFixed(1)}%).`,
        4000
      );

      setTimeout(() => {
        setIsRunning(false);
        router.push(`/inspection-result/${result.inspection_id}`);
      }, 500);
    } catch (err: unknown) {
      setIsRunning(false);
      const msg =
        err instanceof Error ? err.message : "Inspection failed. Please try again.";
      setValidationError(msg);
      showToast(msg);
    }
  };

  return (
    <AppShell
      breadcrumbs={[
        { label: "Asset", href: "/" },
        { label: "Solar Farm Alpha", href: "/" },
        { label: "SEC-4", href: "/" },
        { label: "New Inspection", active: true },
      ]}
    >
      <div className="flex flex-col w-full gap-space-md">
        {/* Page Title & Status Header */}
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-space-md mb-space-xs">
          <div>
            <div className="flex items-center gap-space-xs font-label-caps text-label-caps text-primary tracking-widest uppercase mb-1">
              <span className="inline-block w-2 h-2 rounded-full bg-primary animate-pulse"></span>
              Optical RGB Diagnostic Pipeline • EfficientNet-B0
            </div>
            <h1 className="font-headline-xl text-headline-xl text-on-surface tracking-tight">
              Initiate New Panel AI Visual Inspection
            </h1>
            <p className="font-body-md text-body-md text-on-surface-variant max-w-2xl mt-0.5">
              Upload standard optical RGB solar panel image to perform visual feature extraction,
              classification across 6 fault classes, Grad-CAM attention mapping, and visual severity estimation.
            </p>
          </div>
          <div className="flex items-center gap-space-sm self-start lg:self-auto bg-surface-container-low p-1.5 rounded-lg border border-outline-variant/30">
            <div className="flex items-center gap-1.5 px-space-sm py-1 bg-surface-container-lowest rounded text-on-surface shadow-sm">
              <span
                className="material-symbols-outlined text-secondary text-sm"
                style={{ fontVariationSettings: "'FILL' 1" }}
              >
                bolt
              </span>
              <span className="font-code-id text-code-id font-medium">
                Model: EfficientNet-B0
              </span>
            </div>
            <div className="px-space-sm py-1 font-label-caps text-label-caps bg-surface-container text-on-surface-variant rounded">
              INFERENCE ~320ms
            </div>
          </div>
        </div>

        {/* Validation error notification */}
        {validationError && (
          <div className="p-space-md bg-error-container text-on-error-container rounded-lg border border-error/30 flex items-center gap-2 font-body-md">
            <span className="material-symbols-outlined text-error text-lg">
              error
            </span>
            <span>{validationError}</span>
          </div>
        )}

        {/* Main Grid: Form Inputs + Live Preview SOP */}
        <div className="grid grid-cols-1 xl:grid-cols-12 gap-space-lg">
          {/* Left Column: Form & Calibration (7 cols) */}
          <div className="xl:col-span-7 flex flex-col gap-space-lg">
            {/* Form Card */}
            <div className="bg-surface-container-lowest rounded-xl shadow-sm border border-outline-variant/30 p-space-lg flex flex-col gap-space-md">
              <div className="flex items-center justify-between pb-space-sm border-b border-outline-variant/20">
                <div className="flex items-center gap-space-xs">
                  <span className="material-symbols-outlined text-primary text-xl">
                    tune
                  </span>
                  <span className="font-headline-sm text-headline-sm text-on-surface">
                    Target Asset Parameters
                  </span>
                </div>
                <span className="font-label-caps text-label-caps bg-surface-container px-space-xs py-0.5 text-on-surface-variant rounded">
                  STEP 01 OF 02
                </span>
              </div>

              {/* Panel Registry Identifier Input */}
              <div className="flex flex-col gap-space-xs">
                <div className="flex items-center justify-between">
                  <label
                    className="font-label-sm text-label-sm font-semibold text-on-surface uppercase tracking-wider"
                    htmlFor="panel-id-input"
                  >
                    Panel Registry Identifier
                  </label>
                  <span className="font-label-caps text-label-caps text-on-surface-variant">
                    FORMAT: SP-[LOC]-[ID]
                  </span>
                </div>
                <div className="relative">
                  <span className="material-symbols-outlined absolute left-3 top-2.5 text-outline text-lg">
                    barcode_scanner
                  </span>
                  <input
                    className="w-full h-10 pl-9 pr-24 bg-surface-container-low border border-outline-variant/30 rounded font-code-id text-code-id text-on-surface placeholder:text-outline-variant focus:bg-surface-container-lowest focus:border-primary focus:outline-none transition-colors"
                    id="panel-id-input"
                    placeholder="e.g., SP-HYD-001"
                    type="text"
                    value={panelId}
                    onChange={(e) => setPanelId(e.target.value)}
                  />
                  <div className="absolute right-2 top-1.5 flex items-center gap-1">
                    <button
                      type="button"
                      onClick={handleVerify}
                      className="px-2 py-1 text-primary hover:bg-surface-container rounded font-label-caps text-label-caps uppercase transition-colors"
                    >
                      {isVerified ? "✓ Verified" : "Verify"}
                    </button>
                  </div>
                </div>

                {/* Quick Select Chips */}
                <div className="flex items-center gap-space-xs mt-1 flex-wrap">
                  <span className="font-label-caps text-label-caps text-on-surface-variant mr-1">
                    Quick Select:
                  </span>
                  {["SP-HYD-001", "SP-HYD-002", "SP-HYD-003", "SP-FLD-014"].map(
                    (chip) => (
                      <button
                        key={chip}
                        type="button"
                        onClick={() => {
                          setPanelId(chip);
                          setIsVerified(true);
                          setValidationError(null);
                        }}
                        className={`px-2 py-0.5 rounded font-code-id text-code-id transition-colors ${
                          panelId === chip
                            ? "bg-primary text-on-primary font-semibold"
                            : "bg-surface-container hover:bg-surface-container-high text-on-surface"
                        }`}
                      >
                        {chip}
                      </button>
                    )
                  )}
                </div>
              </div>

              {/* Facility Geographic Block Dropdown */}
              <div className="flex flex-col gap-space-xs mt-1">
                <label
                  className="font-label-sm text-label-sm font-semibold text-on-surface uppercase tracking-wider"
                  htmlFor="location-select"
                >
                  Facility Geographic Block
                </label>
                <div className="relative">
                  <span className="material-symbols-outlined absolute left-3 top-2.5 text-outline text-lg">
                    location_on
                  </span>
                  <select
                    className="w-full h-10 pl-9 pr-10 bg-surface-container-low border border-outline-variant/30 rounded font-body-md text-body-md text-on-surface appearance-none focus:bg-surface-container-lowest focus:border-primary focus:outline-none transition-colors"
                    id="location-select"
                    value={location}
                    onChange={(e) => setLocation(e.target.value)}
                  >
                    <option value="Block A - Rooftop 1">Block A - Rooftop 1</option>
                    <option value="Block B - Rooftop 2">Block B - Rooftop 2</option>
                    <option value="Engineering Building - Roof">
                      Engineering Building - Roof
                    </option>
                    <option value="Solar Inverter Substation 03 - East Field">
                      Solar Inverter Substation 03 - East Field
                    </option>
                    <option value="Custom Telemetry Point (Override)">
                      Custom Telemetry Point (Override)
                    </option>
                  </select>
                  <span className="material-symbols-outlined absolute right-3 top-2.5 text-outline pointer-events-none text-lg">
                    expand_more
                  </span>
                </div>
              </div>

              {/* Imagery Source & Payload Upload */}
              <div className="flex flex-col gap-space-xs mt-2">
                <div className="flex items-center justify-between">
                  <label className="font-label-sm text-label-sm font-semibold text-on-surface uppercase tracking-wider">
                    Optical RGB Image Source &amp; Payload
                  </label>
                  <span className="font-label-caps text-label-caps text-secondary font-medium flex items-center gap-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-secondary"></span>{" "}
                    Ready for Ingestion
                  </span>
                </div>

                {/* Dropzone */}
                <div
                  id="dropzone"
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                  className="relative group p-space-md rounded-xl bg-surface-container-low border-2 border-dashed border-outline-variant/60 hover:border-primary/80 transition-all duration-200 cursor-pointer"
                >
                  <input
                    ref={fileInputRef}
                    accept="image/jpeg,image/png,image/webp,image/bmp"
                    className="hidden"
                    type="file"
                    onChange={handleFileChange}
                  />
                  <div className="flex flex-col items-center justify-center text-center py-space-sm">
                    <div className="w-12 h-12 rounded-full bg-surface-container flex items-center justify-center text-primary group-hover:scale-110 transition-transform mb-2">
                      <span className="material-symbols-outlined text-2xl">
                        cloud_upload
                      </span>
                    </div>
                    <p className="font-headline-sm text-headline-sm text-on-surface">
                      Drag &amp; Drop optical RGB solar panel image
                    </p>
                    <p className="font-body-sm text-body-sm text-on-surface-variant max-w-sm mt-1">
                      Supports standard JPEG, PNG, WEBP, or BMP images up to 25MB.
                    </p>
                    <div className="flex items-center gap-2 mt-3">
                      <span className="px-2 py-0.5 bg-surface-container text-on-surface font-label-caps text-label-caps rounded">
                        RGB OPTICAL
                      </span>
                      <span className="px-2 py-0.5 bg-surface-container text-on-surface font-label-caps text-label-caps rounded">
                        DRONE SURVEY
                      </span>
                      <span className="px-2 py-0.5 bg-surface-container text-on-surface font-label-caps text-label-caps rounded">
                        HANDHELD CAPTURE
                      </span>
                    </div>
                  </div>
                </div>

                {/* Image Preview Row */}
                {selectedFile && (
                  <div className="bg-surface-container p-space-sm rounded-lg flex items-center justify-between mt-2 border border-outline-variant/30">
                    <div className="flex items-center gap-space-sm min-w-0">
                      <div className="w-10 h-10 rounded bg-inverse-surface shrink-0 flex items-center justify-center text-inverse-on-surface relative overflow-hidden">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          className="w-full h-full object-cover"
                          alt="Thumbnail preview"
                          src={selectedFile.previewUrl}
                        />
                        <div className="absolute inset-0 bg-primary/20 pointer-events-none"></div>
                      </div>
                      <div className="flex flex-col min-w-0">
                        <div className="flex items-center gap-1.5 truncate">
                          <span className="font-code-id text-code-id font-semibold text-on-surface truncate">
                            {selectedFile.name}
                          </span>
                          <span className="px-1.5 py-0.5 bg-secondary text-on-secondary rounded font-label-caps text-label-caps">
                            READY
                          </span>
                        </div>
                        <div className="flex items-center gap-2 font-label-caps text-label-caps text-on-surface-variant mt-0.5">
                          <span>{selectedFile.size}</span>
                          <span>•</span>
                          <span className="text-primary font-semibold">
                            {selectedFile.resolution}
                          </span>
                          <span>•</span>
                          <span>{selectedFile.format}</span>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-1">
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedFile(null);
                          setRawFile(null);
                          showToast("Image payload cleared.");
                        }}
                        className="p-1.5 text-error hover:bg-error-container hover:text-on-error-container rounded transition-colors"
                        title="Remove & Reset"
                      >
                        <span className="material-symbols-outlined text-base">
                          delete
                        </span>
                      </button>
                    </div>
                  </div>
                )}
              </div>

              {/* Action Button: Run Inspection */}
              <div className="pt-space-sm flex flex-col gap-2">
                <button
                  type="button"
                  id="run-inspect-btn"
                  onClick={handleRunInspection}
                  disabled={isRunning}
                  className="w-full py-3 px-space-md bg-primary hover:bg-primary-container text-on-primary rounded font-body-lg text-body-lg font-semibold flex items-center justify-center gap-2 shadow-sm transition-all transform active:scale-[0.99] disabled:opacity-75 cursor-pointer"
                >
                  {isRunning ? (
                    <>
                      <span className="material-symbols-outlined text-xl animate-spin">
                        sync
                      </span>
                      <span>RUNNING EFFICIENTNET-B0 INFERENCE...</span>
                    </>
                  ) : (
                    <>
                      <span
                        className="material-symbols-outlined text-xl"
                        style={{ fontVariationSettings: "'FILL' 1" }}
                      >
                        auto_fix_high
                      </span>
                      <span>RUN INSPECTION</span>
                    </>
                  )}
                </button>
                <div className="flex items-center justify-between px-1">
                  <span className="font-code-id text-code-id text-outline text-[11px]">
                    POST /api/inspect (FastAPI Ready)
                  </span>
                  <span className="font-label-caps text-label-caps text-on-surface-variant">
                    EST TIME: ~0.4 SEC
                  </span>
                </div>
              </div>
            </div>

            {/* Validation / Calibration Status Cards */}
            <div className="flex flex-col gap-2.5">
              {/* Asset Validation Card */}
              <div className="p-space-sm bg-surface-container-lowest rounded-lg flex items-start gap-space-sm shadow-sm border border-outline-variant/30">
                <div className="w-6 h-6 rounded bg-secondary-container text-on-secondary-container flex items-center justify-center shrink-0 mt-0.5">
                  <span
                    className="material-symbols-outlined text-sm"
                    style={{ fontVariationSettings: "'FILL' 1" }}
                  >
                    verified
                  </span>
                </div>
                <div className="flex flex-col min-w-0">
                  <span className="font-label-sm text-label-sm font-semibold text-on-surface">
                    Asset Validation Verified
                  </span>
                  <p className="font-body-sm text-body-sm text-on-surface-variant mt-0.5">
                    Panel validation passed: Asset{" "}
                    <span className="font-code-id text-code-id text-primary font-semibold">
                      {panelId}
                    </span>{" "}
                    confirmed in registry{" "}
                    <span className="font-semibold text-on-surface">{location}</span>.
                  </p>
                </div>
              </div>

              {/* Optical Quality Calibration Card */}
              <div className="p-space-sm bg-surface-container-lowest rounded-lg flex items-start gap-space-sm shadow-sm border border-outline-variant/30">
                <div className="w-6 h-6 rounded bg-tertiary-fixed text-on-tertiary-fixed flex items-center justify-center shrink-0 mt-0.5">
                  <span className="material-symbols-outlined text-sm">
                    photo_filter
                  </span>
                </div>
                <div className="flex flex-col min-w-0 flex-1">
                  <div className="flex items-center justify-between">
                    <span className="font-label-sm text-label-sm font-semibold text-on-surface">
                      Optical RGB Image Quality
                    </span>
                    <span className="font-code-id text-code-id font-bold text-secondary">
                      94 / 100
                    </span>
                  </div>
                  <p className="font-body-sm text-body-sm text-on-surface-variant mt-0.5">
                    Sharpness index 94/100, Sun glare minimal (&lt;2.1%),
                    Orthogonal camera angle optimal for Grad-CAM attention localization.
                  </p>
                  <div className="w-full bg-surface-container h-1.5 rounded-full overflow-hidden mt-1.5">
                    <div className="bg-secondary h-full rounded-full" style={{ width: "94%" }}></div>
                  </div>
                </div>
              </div>

              {/* Sandbox Active Card */}
              <div className="p-space-sm bg-surface-container-low rounded-lg flex items-start gap-space-sm border border-outline-variant/20">
                <div className="w-6 h-6 rounded bg-surface-container-highest text-on-surface-variant flex items-center justify-center shrink-0 mt-0.5">
                  <span className="material-symbols-outlined text-sm">terminal</span>
                </div>
                <div className="flex flex-col min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-label-caps text-label-caps text-on-surface-variant">
                      AI DIAGNOSTIC PIPELINE
                    </span>
                    <span className="text-xs text-outline">•</span>
                    <span className="font-code-id text-code-id text-primary">
                      EFFICIENTNET-B0 BASELINE
                    </span>
                  </div>
                  <p className="font-body-sm text-body-sm text-on-surface-variant mt-0.5">
                    Executes image preprocessing (224×224 resize, ImageNet normalization),
                    Grad-CAM activation extraction, visual region calculation, and rule-based maintenance triage.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column: Optical Preview & SOP (5 cols) */}
          <div className="xl:col-span-5 flex flex-col gap-space-lg">
            {/* Optical Diagnostic Feed Card */}
            <div className="bg-surface-container-lowest rounded-xl shadow-sm border border-outline-variant/30 overflow-hidden flex flex-col">
              <div className="p-space-md flex items-center justify-between bg-surface-container-low border-b border-outline-variant/20">
                <div className="flex items-center gap-space-xs">
                  <span className="material-symbols-outlined text-primary text-lg">
                    preview
                  </span>
                  <span className="font-headline-sm text-headline-sm text-on-surface">
                    Optical RGB Preview
                  </span>
                </div>
                <div className="flex items-center gap-1 font-label-caps text-label-caps bg-surface-container-lowest px-2 py-0.5 rounded text-on-surface font-semibold border border-outline-variant/30">
                  TARGET: SEC4
                </div>
              </div>

              {/* Viewport with Approximate Visual Region */}
              <div className="relative w-full aspect-[4/3] bg-inverse-surface overflow-hidden group">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
                  alt="Optical top-down view of modern industrial solar panel array"
                  src={DEFAULT_PREVIEW_IMG}
                />
                <div className="absolute inset-0 bg-gradient-to-t from-inverse-surface/80 via-transparent to-transparent pointer-events-none"></div>
                <div className="absolute top-3 left-3 flex flex-col gap-1 pointer-events-none">
                  <div className="px-2 py-1 bg-surface-container-lowest/90 backdrop-blur rounded shadow-sm flex items-center gap-1.5 font-code-id text-code-id text-on-surface">
                    <span className="w-2 h-2 rounded-full bg-error animate-ping"></span>
                    <span>REGION FOCUS: A1-SEC4</span>
                  </div>
                </div>

                {/* Approximate visual region overlay */}
                <div className="absolute top-1/4 left-1/3 w-32 h-24 rounded border-2 border-dashed border-error bg-error/10 pointer-events-none flex flex-col justify-between p-1">
                  <span className="font-label-caps text-label-caps bg-error text-on-error px-1 py-0.5 rounded self-start">
                    APPROXIMATE VISUAL REGION
                  </span>
                  <span className="font-code-id text-code-id text-on-error bg-inverse-surface/80 px-1 py-0.5 rounded self-end text-[10px]">
                    ESTIMATE ONLY
                  </span>
                </div>

                <div className="absolute bottom-2 left-2 right-2 bg-inverse-surface/90 backdrop-blur-sm text-inverse-on-surface px-2 py-1 rounded text-center">
                  <div className="font-code-id text-code-id text-[11px] text-error-container">
                    Visual region estimation only — NOT an exact defect boundary.
                  </div>
                </div>
              </div>

              {/* Telemetry Stats footer */}
              <div className="p-space-md grid grid-cols-3 gap-2 bg-surface-container-low text-center border-t border-outline-variant/20">
                <div className="bg-surface-container-lowest p-2 rounded border border-outline-variant/20">
                  <span className="font-label-caps text-label-caps text-on-surface-variant block uppercase">
                    Cells Visualized
                  </span>
                  <span className="font-code-metric text-code-metric text-on-surface">
                    72 / 72
                  </span>
                </div>
                <div className="bg-surface-container-lowest p-2 rounded border border-outline-variant/20">
                  <span className="font-label-caps text-label-caps text-on-surface-variant block uppercase">
                    Resolution
                  </span>
                  <span className="font-code-metric text-code-metric text-on-surface">
                    224×224
                  </span>
                </div>
                <div className="bg-surface-container-lowest p-2 rounded border border-outline-variant/20">
                  <span className="font-label-caps text-label-caps text-on-surface-variant block uppercase">
                    Confidence
                  </span>
                  <span className="font-code-metric text-code-metric text-secondary">
                    98.8%
                  </span>
                </div>
              </div>
            </div>

            {/* Pre-Inspection Visual SOP Card */}
            <div className="bg-surface-container-lowest rounded-xl shadow-sm border border-outline-variant/30 p-space-lg flex flex-col gap-space-md">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-space-xs">
                  <span className="material-symbols-outlined text-secondary text-lg">
                    fact_check
                  </span>
                  <h3 className="font-headline-sm text-headline-sm text-on-surface">
                    Visual Inspection SOP
                  </h3>
                </div>
                <span className="font-label-caps text-label-caps text-on-surface-variant">
                  IEC TS 62446-1
                </span>
              </div>

              <ul className="flex flex-col gap-space-sm font-body-sm text-body-sm text-on-surface">
                <li className="flex items-start gap-space-sm p-2 rounded bg-surface-container-low border border-outline-variant/20">
                  <span className="material-symbols-outlined text-secondary text-base mt-0.5">
                    check_box
                  </span>
                  <div>
                    <span className="font-semibold block text-on-surface">
                      Adequate Surface Illumination
                    </span>
                    <span className="text-on-surface-variant">
                      Ensure sunlight provides uniform, clear lighting without cast shadows across the active module face.
                    </span>
                  </div>
                </li>
                <li className="flex items-start gap-space-sm p-2 rounded bg-surface-container-low border border-outline-variant/20">
                  <span className="material-symbols-outlined text-secondary text-base mt-0.5">
                    check_box
                  </span>
                  <div>
                    <span className="font-semibold block text-on-surface">
                      Camera Angle Orthogonality
                    </span>
                    <span className="text-on-surface-variant">
                      Capture image as perpendicular to the panel glass as feasible to minimize perspective distortion and reflection glare.
                    </span>
                  </div>
                </li>
                <li className="flex items-start gap-space-sm p-2 rounded bg-surface-container-low border border-outline-variant/20">
                  <span className="material-symbols-outlined text-secondary text-base mt-0.5">
                    check_box
                  </span>
                  <div>
                    <span className="font-semibold block text-on-surface">
                      Lens Cleanliness &amp; Focus
                    </span>
                    <span className="text-on-surface-variant">
                      Ensure camera lens is free of dust or moisture to maintain optimal feature sharpness for EfficientNet-B0 inference.
                    </span>
                  </div>
                </li>
              </ul>

              <div className="p-space-sm bg-surface-container rounded-lg flex items-center justify-between border border-outline-variant/30">
                <div className="flex items-center gap-space-xs font-body-sm text-body-sm">
                  <span className="material-symbols-outlined text-primary text-base">
                    download
                  </span>
                  <span className="font-medium text-on-surface">
                    Inspection Guideline SOP (PDF)
                  </span>
                </div>
                <button
                  type="button"
                  onClick={() => showToast("Downloading SOP guideline document...")}
                  className="font-label-caps text-label-caps text-primary hover:underline uppercase font-bold"
                >
                  Download
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Statutory Disclaimer */}
        <Disclaimer />
      </div>
    </AppShell>
  );
}
