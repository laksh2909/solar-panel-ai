"use client";

import React, { useState, useRef } from "react";
import { useRouter } from "next/navigation";
import { Upload, Camera, AlertCircle, CheckCircle2, Loader2, ArrowLeft } from "lucide-react";
import Link from "next/link";
import { AppShell } from "@/components/layout/AppShell";
import { Disclaimer } from "@/components/common/Disclaimer";
import { useToast } from "@/components/common/Toast";
import { submitInspection } from "@/lib/api";

const SAMPLE_PANEL_IMG =
  "https://lh3.googleusercontent.com/aida-public/AB6AXuClu9_LnwC1NQr9D9IL_YRk0G3iChTCUSzuRi3UCDO9Dfv9QtZq7gHbu38Hoj6g3gEY09fG0Ar6xS_--_c4k0lyAzGNlr-OZ7xtDPysKf1efYN1wCU_eRp7LPqcGGHmcVF1KlOj3utNLMbe_3pj1twTjdKsle077yF2JW57TmLovR6Ekw_pZxm5xjWY3geAATZ9R3srg_HXr9DJ2FDYxqU-I8b0YLbyUaeEHemeOf3hLeXt3WEJLAq-KA";

export default function NewInspectionPage() {
  const router = useRouter();
  const { showToast } = useToast();
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Panel ID and Location are OPTIONAL — start empty
  const [panelId, setPanelId] = useState("");
  const [location, setLocation] = useState("");
  const [rawFile, setRawFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [fileSize, setFileSize] = useState<string | null>(null);

  const [isRunning, setIsRunning] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);

  const hasImageSelected = Boolean(rawFile || previewUrl);

  const getSafeErrorMessage = (err: unknown) => {
    const message =
      err instanceof Error
        ? err.message
        : typeof err === "string"
        ? err
        : "The inspection request could not be completed.";

    const normalized = message.trim();
    if (!normalized) {
      return "Please try again.";
    }

    if (/image could not be read|resolution is too low|appears excessively blurry|too dark|too bright/i.test(normalized)) {
      const reason = normalized
        .replace(/\s+/g, " ")
        .replace(/^\s+|\s+$/g, "");
      return reason.endsWith(".") ? reason : `${reason}.`;
    }

    if (/network|fetch|timeout|connection|failed to fetch/i.test(normalized)) {
      return "The connection to the AI inspection service was interrupted. Please try again.";
    }

    if (/image|file|unsupported|invalid|bad request|required/i.test(normalized)) {
      return "The selected image could not be processed. Please try a different image.";
    }

    return "Please try again.";
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (!file.type.startsWith("image/")) {
        setValidationError("Please select a valid image file (JPEG, PNG, WEBP).");
        return;
      }
      setRawFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setFileName(file.name);
      setFileSize(`${(file.size / (1024 * 1024)).toFixed(2)} MB`);
      setValidationError(null);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) {
      if (!file.type.startsWith("image/")) {
        setValidationError("Please drop a valid image file (JPEG, PNG, WEBP).");
        return;
      }
      setRawFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setFileName(file.name);
      setFileSize(`${(file.size / (1024 * 1024)).toFixed(2)} MB`);
      setValidationError(null);
    }
  };

  const handleUseSampleImage = async () => {
    try {
      const resp = await fetch(SAMPLE_PANEL_IMG);
      const blob = await resp.blob();
      const file = new File([blob], "sample_panel.jpg", { type: "image/jpeg" });
      setRawFile(file);
      setPreviewUrl(SAMPLE_PANEL_IMG);
      setFileName("sample_panel.jpg");
      setFileSize(`${(blob.size / (1024 * 1024)).toFixed(2)} MB`);
      setValidationError(null);
      showToast("Sample solar panel image loaded.");
    } catch {
      setPreviewUrl(SAMPLE_PANEL_IMG);
      setFileName("sample_panel.jpg");
      setFileSize("1.2 MB");
    }
  };

  const handleStartInspection = async () => {
    // Only image is required — Panel ID and Location are OPTIONAL
    if (!rawFile && !previewUrl) {
      setValidationError("Please upload a solar panel image.");
      return;
    }

    setValidationError(null);
    setIsRunning(true);

    try {
      let fileToSend = rawFile;
      if (!fileToSend && previewUrl) {
        const resp = await fetch(previewUrl);
        const blob = await resp.blob();
        fileToSend = new File([blob], fileName || "panel.jpg", {
          type: blob.type || "image/jpeg",
        });
      }

      const formData = new FormData();
      // Send panel_id and location as-is (may be empty strings)
      // The backend will auto-generate panel_id if empty and use fallback location
      formData.append("panel_id", panelId.trim());
      formData.append("location", location.trim());
      if (fileToSend) {
        formData.append("file", fileToSend);
        formData.append("image", fileToSend);
      }

      const result = await submitInspection(formData);

      showToast("Inspection completed successfully.");
      router.push(`/inspection-result/${result.inspection_id}`);
    } catch (err: unknown) {
      const safeMessage = getSafeErrorMessage(err);
      const qualityMessage = /image could not be read|resolution is too low|appears excessively blurry|too dark|too bright/i.test(safeMessage)
        ? `Image quality is not suitable for inspection. ${safeMessage}`
        : `Inspection could not be completed. ${safeMessage}`;
      setValidationError(qualityMessage);
      setIsRunning(false);
      showToast("Inspection could not be completed.");
    }
  };

  return (
    <AppShell
      breadcrumbs={[
        { label: "Dashboard", href: "/" },
        { label: "New Inspection", active: true },
      ]}
    >
      <div className="flex flex-col gap-6 max-w-3xl mx-auto w-full">
        {/* Page Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-on-surface tracking-tight">
              New Inspection
            </h1>
            <p className="text-sm text-on-surface-variant mt-1">
              Upload a solar panel image to begin the AI inspection.
            </p>
          </div>
          <Link
            href="/"
            className="inline-flex items-center gap-1.5 text-xs font-medium text-on-surface-variant hover:text-on-surface transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Back to Dashboard</span>
          </Link>
        </div>

        {/* Validation Error Alert */}
        {validationError && (
          <div className="flex items-center gap-3 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800 shadow-sm">
            <AlertCircle className="h-5 w-5 shrink-0 text-amber-600" />
            <span>{validationError}</span>
          </div>
        )}

        {/* Inspection Form Card */}
        <div
          className="rounded-2xl border border-outline-variant/40 bg-surface-container-lowest p-5 shadow-sm sm:p-6"
          aria-busy={isRunning}
        >
          <div className="space-y-6">
            <div className="space-y-3">
              <div className="flex items-center justify-between gap-3">
                <label className="block text-sm font-semibold text-on-surface">
                  Upload Image <span className="text-rose-500">*</span>
                </label>
                {hasImageSelected && !isRunning ? (
                  <span className="rounded-full border border-emerald-200 bg-emerald-50 px-2 py-0.5 text-[10px] font-medium uppercase tracking-[0.08em] text-emerald-700">
                    Ready
                  </span>
                ) : null}
              </div>

              <p className="text-sm leading-relaxed text-on-surface-variant">
                Upload a clear solar-panel image. The AI will analyze the visible condition and generate an inspection result.
              </p>

              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept="image/jpeg,image/png,image/webp,image/bmp"
                className="hidden"
                disabled={isRunning}
              />

              {!previewUrl ? (
                <div
                  onDragOver={(e) => {
                    if (!isRunning) e.preventDefault();
                  }}
                  onDrop={isRunning ? undefined : handleDrop}
                  onClick={() => {
                    if (!isRunning) fileInputRef.current?.click();
                  }}
                  className="rounded-2xl border-2 border-dashed border-outline-variant/60 bg-surface-container-low/40 p-6 text-center transition-colors hover:border-primary/60 hover:bg-surface-container-low/60"
                >
                  <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-primary/10 text-primary">
                    <Upload className="h-5 w-5" />
                  </div>
                  <p className="text-sm font-semibold text-on-surface">
                    Drag and drop your panel image, or <span className="text-primary">browse</span>
                  </p>
                  <p className="mt-1 text-xs text-on-surface-variant">
                    Supports JPEG, PNG, WEBP (Standard RGB capture)
                  </p>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      if (!isRunning) handleUseSampleImage();
                    }}
                    disabled={isRunning}
                    className="mt-4 rounded-lg border border-outline-variant/40 bg-surface px-3 py-1.5 text-xs font-semibold text-primary transition-colors hover:bg-surface-container disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    Or click to load a sample panel image
                  </button>
                </div>
              ) : (
                <div className="overflow-hidden rounded-2xl border border-outline-variant/40 bg-surface-container-low">
                  <div className="flex h-72 items-center justify-center bg-black/5">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={previewUrl}
                      alt="Solar panel preview"
                      className="h-full w-full object-contain"
                    />
                  </div>
                  <div className="flex items-center justify-between gap-3 border-t border-outline-variant/20 bg-surface px-3.5 py-3 text-xs">
                    <div className="flex min-w-0 items-center gap-2">
                      <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-600" />
                      <span className="truncate font-medium text-on-surface">
                        {fileName || "Image loaded"}
                      </span>
                      {fileSize && <span className="text-on-surface-variant">({fileSize})</span>}
                    </div>
                    <button
                      type="button"
                      onClick={() => {
                        if (!isRunning) fileInputRef.current?.click();
                      }}
                      disabled={isRunning}
                      className="shrink-0 text-primary hover:underline disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      Change Image
                    </button>
                  </div>
                </div>
              )}
            </div>

            <div className="rounded-xl border border-outline-variant/30 bg-surface/70 p-4">
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <div>
                  <label
                    htmlFor="panel_id"
                    className="mb-1.5 block text-sm font-medium text-on-surface"
                  >
                    Panel ID <span className="text-xs text-on-surface-variant">(Optional)</span>
                  </label>
                  <input
                    id="panel_id"
                    type="text"
                    value={panelId}
                    onChange={(e) => setPanelId(e.target.value)}
                    placeholder="e.g. SP-HYD-001"
                    disabled={isRunning}
                    className="w-full rounded-lg border border-outline-variant/50 bg-surface px-3.5 py-2.5 text-sm text-on-surface placeholder:text-outline focus:border-primary focus:outline-none disabled:cursor-not-allowed disabled:opacity-70"
                  />
                </div>

                <div>
                  <label
                    htmlFor="location"
                    className="mb-1.5 block text-sm font-medium text-on-surface"
                  >
                    Location <span className="text-xs text-on-surface-variant">(Optional)</span>
                  </label>
                  <input
                    id="location"
                    type="text"
                    value={location}
                    onChange={(e) => setLocation(e.target.value)}
                    placeholder="e.g. Block A - Rooftop 1"
                    disabled={isRunning}
                    className="w-full rounded-lg border border-outline-variant/50 bg-surface px-3.5 py-2.5 text-sm text-on-surface placeholder:text-outline focus:border-primary focus:outline-none disabled:cursor-not-allowed disabled:opacity-70"
                  />
                </div>
              </div>

              <p className="mt-3 text-xs leading-relaxed text-on-surface-variant">
                Optional — add panel details if you want to track this inspection to a specific panel.
                If omitted, a unique ID will be auto-assigned and location will be recorded as
                &quot;Location not specified&quot;.
              </p>
            </div>

            <div className="flex flex-col gap-4 border-t border-outline-variant/20 pt-4 sm:flex-row sm:items-center sm:justify-between">
              <div className="min-w-0 flex-1">
                <span className="block text-xs text-on-surface-variant sm:text-left">
                  The AI will analyze the image, classify faults, and estimate visual severity.
                </span>

                {isRunning && (
                  <div
                    className="mt-3 flex items-center gap-2 rounded-lg border border-primary/20 bg-primary/5 px-3 py-2 text-sm text-on-surface"
                    role="status"
                    aria-live="polite"
                  >
                    <Loader2 className="h-4 w-4 shrink-0 animate-spin text-primary" />
                    <div className="min-w-0">
                      <p className="font-medium text-on-surface">AI inspection in progress</p>
                      <p className="text-xs text-on-surface-variant break-words">
                        Analyzing the solar panel image and preparing the inspection result.
                      </p>
                    </div>
                  </div>
                )}
              </div>

              <div className="flex w-full flex-col gap-2 sm:w-auto sm:flex-row">
                {validationError && !isRunning && (
                  <button
                    type="button"
                    onClick={handleStartInspection}
                    disabled={!hasImageSelected}
                    className="inline-flex items-center justify-center rounded-lg border border-outline-variant/40 bg-surface px-4 py-2.5 text-sm font-medium text-on-surface transition-colors hover:bg-surface-container disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    Try Again
                  </button>
                )}
                <button
                  type="button"
                  onClick={handleStartInspection}
                  disabled={isRunning || !hasImageSelected}
                  className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-primary px-5 py-2.5 text-sm font-semibold text-on-primary shadow-sm transition-colors hover:bg-primary-container disabled:cursor-not-allowed disabled:opacity-70 sm:w-auto"
                >
                  {isRunning ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      <span>Analyzing Image...</span>
                    </>
                  ) : (
                    <>
                      <Camera className="h-4 w-4" />
                      <span>Start Inspection</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Disclaimer */}
        <Disclaimer variant="compact" />
      </div>
    </AppShell>
  );
}
