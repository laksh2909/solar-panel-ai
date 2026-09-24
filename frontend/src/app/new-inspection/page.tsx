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

  const [panelId, setPanelId] = useState("SP-HYD-001");
  const [location, setLocation] = useState("Block A - Rooftop 1");
  const [rawFile, setRawFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [fileSize, setFileSize] = useState<string | null>(null);

  const [isRunning, setIsRunning] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);

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
    const trimmedPanelId = panelId.trim();
    const trimmedLocation = location.trim();

    if (!trimmedPanelId) {
      setValidationError("Please enter a Panel ID.");
      return;
    }
    if (!trimmedLocation) {
      setValidationError("Please enter the panel location.");
      return;
    }
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
      formData.append("panel_id", trimmedPanelId);
      formData.append("location", trimmedLocation);
      if (fileToSend) {
        formData.append("file", fileToSend);
        formData.append("image", fileToSend);
      }

      const result = await submitInspection(formData);

      showToast("Inspection completed successfully.");
      router.push(`/inspection-result/${result.inspection_id}`);
    } catch (err: unknown) {
      setIsRunning(false);
      const msg =
        err instanceof Error
          ? err.message
          : "An error occurred during inspection. Please try again.";
      setValidationError(msg);
      showToast(msg);
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
              Upload a solar panel image to begin the inspection.
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
          <div className="p-4 bg-rose-50 text-rose-800 rounded-xl border border-rose-200 flex items-center gap-3 text-sm">
            <AlertCircle className="w-5 h-5 text-rose-600 shrink-0" />
            <span>{validationError}</span>
          </div>
        )}

        {/* Inspection Form Card */}
        <div className="bg-surface-container-lowest p-6 sm:p-8 rounded-xl border border-outline-variant/30 shadow-sm flex flex-col gap-6">
          {/* Upload Area */}
          <div>
            <label className="block text-sm font-semibold text-on-surface mb-2">
              Upload Image
            </label>

            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="image/jpeg,image/png,image/webp,image/bmp"
              className="hidden"
            />

            {!previewUrl ? (
              <div
                onDragOver={(e) => e.preventDefault()}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-outline-variant/50 hover:border-primary/60 rounded-xl p-8 flex flex-col items-center justify-center text-center cursor-pointer transition-colors bg-surface-container-low/30 hover:bg-surface-container-low/60"
              >
                <div className="w-12 h-12 rounded-full bg-primary/10 text-primary flex items-center justify-center mb-3">
                  <Upload className="w-6 h-6" />
                </div>
                <p className="text-sm font-semibold text-on-surface">
                  Drag and drop your panel image, or{" "}
                  <span className="text-primary hover:underline">browse</span>
                </p>
                <p className="text-xs text-on-surface-variant mt-1">
                  Supports JPEG, PNG, WEBP (Standard RGB capture)
                </p>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleUseSampleImage();
                  }}
                  className="mt-4 text-xs font-semibold text-primary hover:underline px-3 py-1.5 rounded bg-surface border border-outline-variant/30"
                >
                  Or click to load a sample panel image
                </button>
              </div>
            ) : (
              <div className="relative rounded-xl overflow-hidden border border-outline-variant/30 bg-surface-container-low flex flex-col">
                <div className="relative w-full h-72 bg-black/5 flex items-center justify-center">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={previewUrl}
                    alt="Solar panel preview"
                    className="w-full h-full object-contain"
                  />
                </div>
                <div className="p-3.5 bg-surface flex items-center justify-between border-t border-outline-variant/20 text-xs">
                  <div className="flex items-center gap-2 truncate">
                    <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
                    <span className="font-medium text-on-surface truncate">
                      {fileName || "Image loaded"}
                    </span>
                    {fileSize && (
                      <span className="text-on-surface-variant">({fileSize})</span>
                    )}
                  </div>
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="text-primary font-medium hover:underline shrink-0 ml-3"
                  >
                    Change Image
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Panel ID & Location Fields */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label
                htmlFor="panel_id"
                className="block text-sm font-semibold text-on-surface mb-1.5"
              >
                Panel ID
              </label>
              <input
                id="panel_id"
                type="text"
                value={panelId}
                onChange={(e) => setPanelId(e.target.value)}
                placeholder="e.g. SP-HYD-001"
                className="w-full px-3.5 py-2 text-sm bg-surface border border-outline-variant/40 rounded-lg text-on-surface placeholder:text-outline focus:outline-none focus:border-primary transition-colors font-medium"
              />
            </div>

            <div>
              <label
                htmlFor="location"
                className="block text-sm font-semibold text-on-surface mb-1.5"
              >
                Location
              </label>
              <input
                id="location"
                type="text"
                value={location}
                onChange={(e) => setLocation(e.target.value)}
                placeholder="e.g. Block A - Rooftop 1"
                className="w-full px-3.5 py-2 text-sm bg-surface border border-outline-variant/40 rounded-lg text-on-surface placeholder:text-outline focus:outline-none focus:border-primary transition-colors"
              />
            </div>
          </div>

          {/* Submit Action */}
          <div className="pt-2 flex flex-col sm:flex-row items-center justify-between gap-4 border-t border-outline-variant/20">
            <span className="text-xs text-on-surface-variant text-center sm:text-left">
              The AI will analyze the image, classify faults, and estimate visual severity.
            </span>
            <button
              type="button"
              onClick={handleStartInspection}
              disabled={isRunning}
              className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-6 py-2.5 bg-primary hover:bg-primary-container text-on-primary font-semibold text-sm rounded-lg shadow-sm transition-all disabled:opacity-75 cursor-pointer"
            >
              {isRunning ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Analyzing Image...</span>
                </>
              ) : (
                <>
                  <Camera className="w-4 h-4" />
                  <span>Start Inspection</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Disclaimer */}
        <Disclaimer variant="compact" />
      </div>
    </AppShell>
  );
}
