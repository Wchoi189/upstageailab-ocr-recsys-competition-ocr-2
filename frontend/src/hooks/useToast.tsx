import { useState, useCallback } from "react";
import type React from "react";
import { Toast, type ToastType } from "../components/ui/Toast";

/**
 * Toast item with unique ID
 */
interface ToastItem {
  id: string;
  message: string;
  type: ToastType;
  duration?: number;
}

/**
 * Hook for managing toast notifications
 *
 * Provides methods to show/hide toast notifications
 * and a component to render active toasts
 */
export function useToast(): {
  showToast: (message: string, type?: ToastType, duration?: number) => void;
  ToastContainer: React.FC;
} {
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  const showToast = useCallback(
    (message: string, type: ToastType = "info", duration?: number): void => {
      const id = `toast-${Date.now()}-${Math.random()}`;
      setToasts((prev) => [...prev, { id, message, type, duration }]);
    },
    []
  );

  const removeToast = useCallback((id: string): void => {
    setToasts((prev) => prev.filter((toast) => toast.id !== id));
  }, []);

  const ToastContainer: React.FC = () => (
    <div>
      {toasts.map((toast, index) => (
        <div
          key={toast.id}
          style={{
            position: "fixed",
            top: `${1 + index * 5}rem`,
            right: "1rem",
            zIndex: 9999,
          }}
        >
          <Toast
            message={toast.message}
            type={toast.type}
            duration={toast.duration}
            onClose={() => removeToast(toast.id)}
          />
        </div>
      ))}
    </div>
  );

  return { showToast, ToastContainer };
}
