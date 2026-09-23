export type FaultType =
  | "Bird-drop"
  | "Clean"
  | "Dusty"
  | "Electrical-damage"
  | "Physical-damage"
  | "Snow-Covered";

export type SeverityLevel = "LOW" | "MEDIUM" | "HIGH";

export type UrgencyLevel =
  | "ROUTINE"
  | "SCHEDULED"
  | "PRIORITY"
  | "IMMEDIATE REVIEW"
  | "IMMEDIATE_REVIEW";

export interface HealthResponse {
  status: string;
  service: string;
  connected?: boolean;
}

export interface InspectionResponse {
  inspection_id: number | string;
  panel_id: string;
  location: string;
  image_filename: string;
  inspection_timestamp: string;
  predicted_class: FaultType;
  confidence: number;
  visual_region_area_percent: number;
  severity: SeverityLevel;
  urgency: UrgencyLevel;
  maintenance_action: string;
  manual_inspection_recommended: boolean;
  confidence_warning?: string | null;
  // Visual feature & explainability metadata
  sub_fault?: string;
  bbox?: [number, number, number, number];
  original_image_url?: string;
  gradcam_image_url?: string;
  region_image_url?: string;
  sensor_source?: string;
  is_demo_sample?: boolean;
}

export interface InspectionListResponse {
  total: number;
  page?: number;
  page_size?: number;
  items: InspectionResponse[];
}

export interface PanelResponse {
  id: number;
  panel_id: string;
  location: string;
  created_at?: string | null;
  installation_date?: string;
  rated_power_wp?: number;
  inverter_string?: string;
  azimuth_tilt?: string;
  technology?: string;
  rack_coordinates?: string;
  health_index?: number;
  total_scans?: number;
  latest_fault?: FaultType;
  latest_severity?: SeverityLevel;
  latest_urgency?: UrgencyLevel;
  last_inspected?: string;
  is_demo_sample?: boolean;
}

export interface PanelListResponse {
  total: number;
  items: PanelResponse[];
}
