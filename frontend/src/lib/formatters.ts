export function formatFaultName(fault?: string): string {
  if (!fault) return "Unknown";
  switch (fault) {
    case "Clean":
      return "Clean";
    case "Dusty":
      return "Dust";
    case "Bird-drop":
      return "Bird Droppings";
    case "Electrical-damage":
      return "Electrical Damage";
    case "Physical-damage":
      return "Physical Damage";
    case "Snow-Covered":
      return "Snow Covered";
    default:
      return fault.replace(/-/g, " ");
  }
}

export function formatInspectionDate(dateStr?: string): string {
  if (!dateStr) return "N/A";
  try {
    const date = new Date(dateStr);
    if (isNaN(date.getTime())) {
      return dateStr;
    }
    return date.toLocaleString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return dateStr;
  }
}

export function getConfidenceFeedback(confidence: number): {
  level: "high" | "medium" | "low";
  text: string;
} {
  const percent = confidence <= 1 ? confidence * 100 : confidence;
  if (percent >= 85) {
    return {
      level: "high",
      text: "High confidence",
    };
  }
  if (percent >= 60) {
    return {
      level: "medium",
      text: "Moderate confidence",
    };
  }
  return {
    level: "low",
    text: "Low confidence — manual review recommended",
  };
}

export function getSeverityFeedback(severity?: string): string {
  const norm = (severity || "").toUpperCase();
  switch (norm) {
    case "HIGH":
    case "CRITICAL":
      return "Large visual region detected. Priority inspection recommended.";
    case "MEDIUM":
      return "Moderate visual anomaly detected. Scheduled inspection recommended.";
    case "LOW":
    default:
      return "Minor or no visible surface defect detected. Routine monitoring.";
  }
}
