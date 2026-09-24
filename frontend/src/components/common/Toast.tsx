"use client";

import React, { createContext, useContext, useState, useCallback } from "react";
import { CheckCircle2, X } from "lucide-react";

interface ToastContextType {
  showToast: (message: string, duration?: number) => void;
}

const ToastContext = createContext<ToastContextType>({
  showToast: () => {},
});

export function useToast() {
  return useContext(ToastContext);
}

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  const showToast = useCallback((message: string, duration = 3000) => {
    setToastMessage(message);
    setTimeout(() => {
      setToastMessage((current) => (current === message ? null : current));
    }, duration);
  }, []);

  return (
    <ToastContext.Provider value={{ showToast }}>
      {children}
      {toastMessage && (
        <div className="fixed bottom-6 right-6 bg-inverse-surface text-inverse-on-surface px-4 py-3 rounded-lg shadow-xl flex items-center gap-3 z-50 border border-outline-variant/20 animate-fade-in max-w-md">
          <CheckCircle2 className="w-5 h-5 text-secondary shrink-0" />
          <span className="text-body-md font-body-md text-sm">{toastMessage}</span>
          <button
            onClick={() => setToastMessage(null)}
            className="text-inverse-on-surface/60 hover:text-inverse-on-surface p-1 rounded transition-colors ml-auto shrink-0"
            aria-label="Close notification"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}
    </ToastContext.Provider>
  );
}
