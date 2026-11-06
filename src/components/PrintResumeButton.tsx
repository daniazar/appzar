'use client';

import React from 'react';
import { Printer } from 'lucide-react';

export default function PrintResumeButton() {
  return (
    <button
      type="button"
      onClick={() => window.print()}
      className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-[#5e60ce] to-[#4ea8de] px-4 py-2 text-xs sm:text-sm font-semibold text-white shadow-lg shadow-[#5e60ce]/25 hover:brightness-110 transition-all"
    >
      <Printer className="h-4 w-4" />
      <span>Print / Save as PDF</span>
    </button>
  );
}
