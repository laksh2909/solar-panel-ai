"use client";

import React, { createContext, useContext, useState, useCallback } from "react";

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
        <div className="fixed bottom-6 right-6 bg-inverse-surface text-inverse-on-surface px-space-md py-space-sm rounded-lg shadow-xl flex items-center gap-3 z-50 border border-outline-variant/20 animate-fade-in">
          <span className="material-symbols-outlined text-secondary text-base">
            check_circle
          </span>
          <span className="font-body-md text-body-md">{toastMessage}</span>
          <button
            onClick={() => setToastMessage(null)}
            className="text-inverse-on-surface/60 hover:text-inverse-on-surface text-xs ml-2"
          >
            ✕
          </button>
        </div>
      )}
    </ToastContext.Provider>
  );
}
