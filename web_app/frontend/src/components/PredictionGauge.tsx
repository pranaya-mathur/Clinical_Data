"use client";

import { motion } from "framer-motion";

interface GaugeProps {
  value: number; // 0 to 1
  size?: number;
}

export default function PredictionGauge({ value, size = 200 }: GaugeProps) {
  const percentage = Math.round(value * 100);
  const radius = size / 2 - 10;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (value * circumference);

  const getColor = (v: number) => {
    if (v < 0.3) return "#10b981"; // Green
    if (v < 0.6) return "#f59e0b"; // Yellow
    return "#ef4444"; // Red
  };

  const color = getColor(value);

  return (
    <div className="relative flex flex-col items-center justify-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="transform -rotate-90">
        {/* Background Circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="currentColor"
          strokeWidth="10"
          fill="transparent"
          className="text-slate-800"
        />
        {/* Progress Circle */}
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke={color}
          strokeWidth="10"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 1.5, ease: "easeOut" }}
          strokeLinecap="round"
          fill="transparent"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
        <span className="text-4xl font-bold tracking-tighter" style={{ color }}>
          {percentage}%
        </span>
        <span className="text-[10px] uppercase font-semibold text-slate-500 tracking-widest mt-1">
          Plaque Probability
        </span>
      </div>
    </div>
  );
}
