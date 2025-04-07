'use client';

import { useEffect, useState } from 'react';

interface ToastProps {
  message: string;
  type?: 'success' | 'error';
}

export function Toast({ message, type = 'success' }: ToastProps) {
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    const timeout = setTimeout(() => setVisible(false), 3000);
    return () => clearTimeout(timeout);
  }, []);

  if (!visible) return null;

  return (
    <div className={`fixed top-5 right-5 z-50 px-4 py-2 rounded-md text-white shadow-lg
      ${type === 'success' ? 'bg-green-600' : 'bg-red-600'}`}>
      {message}
    </div>
  );
}
