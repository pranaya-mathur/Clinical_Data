"use client";

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceArea } from 'recharts';

interface Highlight {
  start: number;
  end: number;
  color: string;
  label: string;
}

interface CurveChartProps {
  title: string;
  data: number[];
  color: string;
  highlights?: Highlight[];
}

export default function CurveChart({ title, data, color, highlights = [] }: CurveChartProps) {
  // Downsample for performance: show every 4th point
  const chartData = data
    .filter((_, i) => i % 4 === 0)
    .map((val, i) => ({ time: i * 4, intensity: parseFloat(val.toFixed(4)) }));

  if (chartData.length === 0) {
    return (
      <div className="rounded-xl p-4 flex items-center justify-center h-[320px]"
        style={{ backgroundColor: "rgba(15,23,42,0.5)", border: "1px solid #1e293b" }}>
        <p className="text-sm" style={{ color: "#64748b" }}>No data yet — run prediction first.</p>
      </div>
    );
  }

  return (
    <div className="rounded-xl p-4"
      style={{ backgroundColor: "rgba(15,23,42,0.5)", border: "1px solid #1e293b" }}>
      <h3 className="text-[11px] font-bold uppercase tracking-wider mb-4" style={{ color: "#64748b" }}>{title}</h3>
      <div style={{ height: 280, width: "100%" }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 10, right: 10, bottom: 25, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
            <XAxis
              dataKey="time"
              stroke="#334155"
              fontSize={11}
              tick={{ fill: "#64748b" }}
              label={{ value: "Migration Time", position: "insideBottom", offset: -15, fill: "#64748b", fontSize: 11 }}
            />
            <YAxis
              stroke="#334155"
              fontSize={11}
              tick={{ fill: "#64748b" }}
              label={{ value: "Intensity", angle: -90, position: "insideLeft", offset: 15, fill: "#64748b", fontSize: 11 }}
            />
            <Tooltip
              contentStyle={{ backgroundColor: "#0f172a", border: "1px solid #1e293b", borderRadius: "8px", fontSize: 12 }}
              labelStyle={{ color: "#94a3b8" }}
              itemStyle={{ color }}
            />
            {highlights.map((h, i) => (
              <ReferenceArea
                key={i}
                x1={h.start}
                x2={h.end}
                fill={h.color}
                strokeOpacity={0.3}
                label={{ value: h.label, position: "top", fill: "#94a3b8", fontSize: 10 }}
              />
            ))}
            <Line
              type="monotone"
              dataKey="intensity"
              stroke={color}
              strokeWidth={2}
              dot={false}
              animationDuration={1200}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
