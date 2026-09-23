import {
  HealthResponse,
  InspectionListResponse,
  InspectionResponse,
  PanelListResponse,
  PanelResponse,
} from "@/types/inspection";
import {
  MOCK_INSPECTIONS,
  MOCK_PANELS,
  MOCK_PANEL_001_INSPECTIONS,
} from "./mock-data";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ||
  process.env.NEXT_PUBLIC_API_URL ||
  "http://127.0.0.1:8000";

export async function fetchHealth(): Promise<HealthResponse> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/health`, {
      method: "GET",
      cache: "no-store",
    });
    if (res.ok) {
      const data = await res.json();
      return { ...data, connected: true };
    }
  } catch {
    // Fallback when backend is unavailable
  }
  return { status: "offline", service: "solar-panel-ai-mock", connected: false };
}

export async function fetchPanels(): Promise<PanelListResponse> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/panels`, {
      method: "GET",
      cache: "no-store",
    });
    if (res.ok) {
      return await res.json();
    }
  } catch {
    // Fallback when backend is unavailable
  }
  return { total: MOCK_PANELS.length, items: MOCK_PANELS };
}

export async function fetchPanel(panelId: string): Promise<PanelResponse | null> {
  try {
    const res = await fetch(
      `${API_BASE_URL}/api/panels/${encodeURIComponent(panelId.trim())}`,
      {
        method: "GET",
        cache: "no-store",
      }
    );
    if (res.ok) {
      return await res.json();
    }
  } catch {
    // Fallback when backend is unavailable
  }
  const found = MOCK_PANELS.find(
    (p) => p.panel_id.toLowerCase() === panelId.trim().toLowerCase()
  );
  return found || MOCK_PANELS[0];
}

export async function fetchPanelInspections(
  panelId: string
): Promise<InspectionListResponse> {
  try {
    const res = await fetch(
      `${API_BASE_URL}/api/panels/${encodeURIComponent(panelId.trim())}/inspections`,
      {
        method: "GET",
        cache: "no-store",
      }
    );
    if (res.ok) {
      const data = await res.json();
      if (Array.isArray(data)) {
        return { total: data.length, items: data };
      }
      return data;
    }
  } catch {
    // Fallback when backend is unavailable
  }
  if (panelId === "SP-HYD-001") {
    return {
      total: MOCK_PANEL_001_INSPECTIONS.length,
      items: MOCK_PANEL_001_INSPECTIONS,
    };
  }
  const filtered = MOCK_INSPECTIONS.filter(
    (i) => i.panel_id.toLowerCase() === panelId.trim().toLowerCase()
  );
  return {
    total: filtered.length,
    items: filtered.length > 0 ? filtered : [MOCK_INSPECTIONS[0]],
  };
}

export async function fetchInspections(
  page = 1,
  pageSize = 50
): Promise<InspectionListResponse> {
  try {
    const res = await fetch(
      `${API_BASE_URL}/api/inspections?page=${page}&page_size=${pageSize}`,
      {
        method: "GET",
        cache: "no-store",
      }
    );
    if (res.ok) {
      return await res.json();
    }
  } catch {
    // Fallback when backend is unavailable
  }
  return { total: MOCK_INSPECTIONS.length, items: MOCK_INSPECTIONS };
}

export async function fetchInspection(
  inspectionId: string | number
): Promise<InspectionResponse | null> {
  const idStr = String(inspectionId).trim();
  const numericId = parseInt(idStr, 10);

  if (!isNaN(numericId) && numericId > 0 && String(numericId) === idStr) {
    try {
      const res = await fetch(`${API_BASE_URL}/api/inspections/${numericId}`, {
        method: "GET",
        cache: "no-store",
      });
      if (res.ok) {
        return await res.json();
      }
    } catch {
      // Fallback when backend is unavailable
    }
  }

  const found = MOCK_INSPECTIONS.find(
    (i) => String(i.inspection_id).toLowerCase() === idStr.toLowerCase()
  );
  return found || MOCK_INSPECTIONS[0];
}

export async function submitInspection(
  formData: FormData
): Promise<InspectionResponse> {
  // Ensure both 'file' and 'image' are appended so whatever the endpoint expects is available
  if (formData.has("file") && !formData.has("image")) {
    formData.append("image", formData.get("file") as Blob);
  } else if (formData.has("image") && !formData.has("file")) {
    formData.append("file", formData.get("image") as Blob);
  }

  try {
    const res = await fetch(`${API_BASE_URL}/api/inspect`, {
      method: "POST",
      body: formData,
    });

    if (res.ok) {
      return await res.json();
    }

    // Backend returned 4xx or 5xx
    const errData = await res.json().catch(() => null);
    const detailMsg =
      errData?.detail || `Inspection service returned error HTTP ${res.status}`;
    throw new Error(detailMsg);
  } catch (err: unknown) {
    // Re-throw server validation errors
    if (
      err instanceof Error &&
      !err.message.includes("Failed to fetch") &&
      !err.message.includes("NetworkError") &&
      !err.message.includes("fetch failed")
    ) {
      throw err;
    }

    // Network failure / backend offline: fallback
    console.warn(
      "FastAPI backend unavailable at",
      API_BASE_URL,
      "- using simulated fallback."
    );
    const panelId = (formData.get("panel_id") as string) || "SP-HYD-001";
    const location =
      (formData.get("location") as string) || "Block A - Rooftop 1";

    return {
      ...MOCK_INSPECTIONS[0],
      inspection_id: `INS-${new Date().getFullYear()}-${Math.floor(
        1000 + Math.random() * 9000
      )}`,
      panel_id: panelId,
      location: location,
      visual_region_area_percent: 14.8,
      is_demo_sample: true,
      inspection_timestamp: new Date().toISOString(),
    };
  }
}
