"use client";

import { useState, useRef } from "react";
import { Upload, Activity, FileText, Download, ShieldCheck, Info, RefreshCw } from "lucide-react";
import CurveChart from "@/components/CurveChart";
import PredictionGauge from "@/components/PredictionGauge";
import { motion, AnimatePresence } from "framer-motion";
import axios from "axios";
import { jsPDF } from "jspdf";
import html2canvas from "html2canvas";

const METADATA_DEFAULTS = {
  age: 55, sex_male: 1, bmi: 27.5, diabetes: 0, hypertension: 1, smoking: 0,
  hdl2b_percent: 0.25, hdl2a_percent: 0.15, hdl3a_percent: 0.2, hdl3b_percent: 0.2,
  hdl3c_percent: 0.2, sdldl_percent: 0.35, total_hdl: 45, total_ldl: 120, hdl_ldl_ratio: 0.37
};

export default function LipospecApp() {
  const [hdlFile, setHdlFile] = useState<File | null>(null);
  const [ldlFile, setLdlFile] = useState<File | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [metadata, setMetadata] = useState(METADATA_DEFAULTS);
  const [isLoadingSample, setIsLoadingSample] = useState(false);
  const reportRef = useRef<HTMLDivElement>(null);

  const loadSample = async () => {
    setIsPredicting(true);
    try {
      const response = await axios.get("http://localhost:8000/samples");
      const sample = response.data[0]; 
      
      // Update metadata with sample data + is_sample flag
      const sampleMeta = { ...sample.metadata, is_sample: true };
      setMetadata(sampleMeta);
      
      // Simulate "files" for UI but we won't actually need them for the API
      setHdlFile(new File([""], "sample_hdl.npy"));
      setLdlFile(new File([""], "sample_ldl.npy"));
      
      // Automatically trigger a "real" prediction call to the backend 
      // but with the is_sample flag so the backend knows what to do
      console.log("🧪 Requesting demo prediction with metadata:", sampleMeta);
      const formData = new FormData();
      formData.append("metadata", JSON.stringify(sampleMeta));
      
      const predResponse = await axios.post("http://localhost:8000/predict", formData);
      console.log("✅ Demo prediction successful:", predResponse.data);
      setResult(predResponse.data);

    } catch (error) {
      console.error("Error loading sample:", error);
      alert("Failed to load sample patient. Ensure backend is running.");
    } finally {
      setIsLoadingSample(false);
      setIsPredicting(false);
    }
  };

  const handlePredict = async () => {
    if (!hdlFile || !ldlFile) {
      alert("Please upload both HDL and LDL curves or use a sample.");
      return;
    }
    setIsPredicting(true);
    const formData = new FormData();
    // Only append files if they aren't the sample placeholders
    if (hdlFile.size > 0) formData.append("hdl_file", hdlFile);
    if (ldlFile.size > 0) formData.append("ldl_file", ldlFile);
    
    formData.append("metadata", JSON.stringify(metadata));

    try {
      const response = await axios.post("http://localhost:8000/predict", formData);
      setResult(response.data);
    } catch {
      alert("Prediction failed. Ensure the backend is running and files are valid .npy curves.");
    } finally {
      setIsPredicting(false);
    }
  };

  const downloadPDF = async () => {
    if (!reportRef.current) return;
    const canvas = await html2canvas(reportRef.current);
    const imgData = canvas.toDataURL("image/png");
    const pdf = new jsPDF("p", "mm", "a4");
    const pdfWidth = pdf.internal.pageSize.getWidth();
    const pdfHeight = (canvas.height * pdfWidth) / canvas.width;
    pdf.addImage(imgData, "PNG", 0, 0, pdfWidth, pdfHeight);
    pdf.save(`LipoSpec_Report_${Date.now()}.pdf`);
  };

  const probability = result ? (result.probability as number) : 0;
  const riskLevel = probability > 0.6 ? "High" : probability > 0.3 ? "Moderate" : "Low";
  const riskColor = probability > 0.6 ? "#ef4444" : probability > 0.3 ? "#f59e0b" : "#10b981";

  return (
    <div className="flex min-h-screen medical-gradient" style={{ backgroundColor: "#0f172a", color: "#e2e8f0" }}>
      {/* Sidebar */}
      <aside
        className="fixed left-0 top-0 h-full w-72 hidden lg:flex flex-col p-6 z-50 glass"
        style={{ borderRight: "1px solid #1e293b" }}
      >
        <div className="flex items-center gap-3 mb-10">
          <div className="w-10 h-10 rounded-xl flex items-center justify-center shadow-lg"
            style={{ backgroundColor: "#0ea5e9" }}>
            <Activity className="text-white w-5 h-5" />
          </div>
          <div>
            <h1 className="text-lg font-bold leading-none tracking-tight text-white">LipoSpec</h1>
            <p className="text-[10px] uppercase font-bold tracking-widest mt-1" style={{ color: "#64748b" }}>Plaque Predictor</p>
          </div>
        </div>

        <div className="space-y-6 flex-1 overflow-y-auto">
          <div>
            <p className="text-[10px] uppercase font-bold tracking-widest mb-4" style={{ color: "#64748b" }}>Patient Data</p>
            <div className="space-y-4">
              {[
                { label: "Age", key: "age", type: "number" },
                { label: "BMI", key: "bmi", type: "number", step: "0.1" },
                { label: "Total HDL (mg/dL)", key: "total_hdl", type: "number" },
                { label: "Total LDL (mg/dL)", key: "total_ldl", type: "number" },
              ].map((field) => (
                <div key={field.key} className="space-y-1.5">
                  <label className="text-[11px] font-semibold ml-1" style={{ color: "#94a3b8" }}>{field.label}</label>
                  <input
                    type={field.type}
                    step={field.step}
                    className="w-full rounded-lg px-3 py-2 text-sm outline-none transition-all text-white"
                    style={{ backgroundColor: "rgba(15,23,42,0.8)", border: "1px solid #1e293b" }}
                    value={(metadata as Record<string, number>)[field.key]}
                    onChange={(e) => setMetadata({ ...metadata, [field.key]: parseFloat(e.target.value) })}
                  />
                </div>
              ))}
              <div className="space-y-1.5">
                <label className="text-[11px] font-semibold ml-1" style={{ color: "#94a3b8" }}>Gender</label>
                <select
                  className="w-full rounded-lg px-3 py-2 text-sm outline-none transition-all text-white"
                  style={{ backgroundColor: "rgba(15,23,42,0.8)", border: "1px solid #1e293b" }}
                  value={metadata.sex_male}
                  onChange={(e) => setMetadata({ ...metadata, sex_male: parseInt(e.target.value) })}
                >
                  <option value={1}>Male</option>
                  <option value={0}>Female</option>
                </select>
              </div>
            </div>
          </div>

          <div>
            <p className="text-[10px] uppercase font-bold tracking-widest mb-4" style={{ color: "#64748b" }}>Comorbidities</p>
            <div className="space-y-2">
              {[
                { label: "Diabetes", key: "diabetes" },
                { label: "Hypertension", key: "hypertension" },
                { label: "Smoking", key: "smoking" },
              ].map((item) => (
                <label
                  key={item.key}
                  className="flex items-center justify-between p-3 rounded-lg cursor-pointer transition-all"
                  style={{ border: "1px solid #1e293b", backgroundColor: "rgba(15,23,42,0.4)" }}
                >
                  <span className="text-xs font-medium text-white">{item.label}</span>
                  <input
                    type="checkbox"
                    className="accent-sky-500"
                    checked={(metadata as Record<string, number>)[item.key] === 1}
                    onChange={(e) => setMetadata({ ...metadata, [item.key]: e.target.checked ? 1 : 0 })}
                  />
                </label>
              ))}
            </div>
          </div>

          <button
            onClick={loadSample}
            disabled={isLoadingSample}
            className="w-full mt-4 flex items-center justify-center gap-2 py-3 rounded-xl text-xs font-bold transition-all"
            style={{ backgroundColor: "rgba(14,165,233,0.1)", border: "1px solid rgba(14,165,233,0.3)", color: "#0ea5e9" }}
          >
            {isLoadingSample ? <RefreshCw className="animate-spin" size={14} /> : <Info size={14} />}
            LOAD SAMPLE PATIENT
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="lg:pl-72 flex-1 p-6 lg:p-10 max-w-7xl mx-auto w-full">

        {/* Header */}
        <header className="flex flex-col md:flex-row md:items-center justify-between gap-6 mb-10">
          <div>
            <div className="flex items-center gap-2 mb-2" style={{ color: "#0ea5e9" }}>
              <ShieldCheck size={14} />
              <span className="text-[10px] font-bold uppercase tracking-[0.2em]">Verified Clinical AI</span>
            </div>
            <h2 className="text-3xl font-bold tracking-tight text-white">Diagnostic Dashboard</h2>
            <p className="mt-2 text-sm" style={{ color: "#64748b" }}>
              Upload HDL &amp; LDL electropherogram curves for multimodal plaque risk assessment.
            </p>
          </div>

          <button
            onClick={handlePredict}
            disabled={isPredicting || !hdlFile || !ldlFile}
            className="flex items-center gap-3 px-8 py-4 rounded-2xl font-bold text-white transition-all cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
            style={{ backgroundColor: "#0ea5e9", boxShadow: "0 0 30px rgba(14,165,233,0.3)" }}
          >
            {isPredicting ? <RefreshCw className="animate-spin" size={20} /> : <Activity size={20} />}
            <span>{isPredicting ? "Processing Samples..." : "Predict Plaque Risk"}</span>
          </button>
        </header>

        {/* Upload + Charts */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 mb-10">
          {/* HDL */}
          <div>
            {!hdlFile ? (
              <div
                className="h-72 rounded-2xl flex flex-col items-center justify-center cursor-pointer transition-all group"
                style={{ border: "2px dashed #1e293b" }}
                onClick={() => document.getElementById("hdl-input")?.click()}
              >
                <div className="w-16 h-16 rounded-full flex items-center justify-center mb-4 group-hover:scale-110 transition-transform"
                  style={{ backgroundColor: "#0f172a" }}>
                  <Upload style={{ color: "#0ea5e9" }} />
                </div>
                <p className="font-bold text-sm text-white">Upload HDL Curve</p>
                <p className="text-[10px] mt-1 uppercase tracking-widest" style={{ color: "#64748b" }}>Supports .npy files</p>
                <input id="hdl-input" type="file" className="hidden" accept=".npy"
                  onChange={(e) => setHdlFile(e.target.files?.[0] || null)} />
              </div>
            ) : (
              <div className="relative group">
                <CurveChart
                  title="HDL Electropherogram"
                  data={result ? (result.hdl_curve as number[]) : []}
                  color="#0ea5e9"
                  highlights={result?.highlights ? (result.highlights as { hdl: unknown[] }).hdl as { start: number; end: number; color: string; label: string }[] : []}
                />
                <button
                  onClick={() => { setHdlFile(null); setResult(null); }}
                  className="absolute top-4 right-4 p-1.5 rounded-lg opacity-0 group-hover:opacity-100 transition-all text-white"
                  style={{ backgroundColor: "rgba(15,23,42,0.8)" }}
                >
                  <RefreshCw size={14} />
                </button>
              </div>
            )}
          </div>

          {/* LDL */}
          <div>
            {!ldlFile ? (
              <div
                className="h-72 rounded-2xl flex flex-col items-center justify-center cursor-pointer transition-all group"
                style={{ border: "2px dashed #1e293b" }}
                onClick={() => document.getElementById("ldl-input")?.click()}
              >
                <div className="w-16 h-16 rounded-full flex items-center justify-center mb-4 group-hover:scale-110 transition-transform"
                  style={{ backgroundColor: "#0f172a" }}>
                  <Upload style={{ color: "#14b8a6" }} />
                </div>
                <p className="font-bold text-sm text-white">Upload LDL Curve</p>
                <p className="text-[10px] mt-1 uppercase tracking-widest" style={{ color: "#64748b" }}>Supports .npy files</p>
                <input id="ldl-input" type="file" className="hidden" accept=".npy"
                  onChange={(e) => setLdlFile(e.target.files?.[0] || null)} />
              </div>
            ) : (
              <div className="relative group">
                <CurveChart
                  title="LDL Electropherogram"
                  data={result ? (result.ldl_curve as number[]) : []}
                  color="#14b8a6"
                  highlights={result?.highlights ? (result.highlights as { ldl: unknown[] }).ldl as { start: number; end: number; color: string; label: string }[] : []}
                />
                <button
                  onClick={() => { setLdlFile(null); setResult(null); }}
                  className="absolute top-4 right-4 p-1.5 rounded-lg opacity-0 group-hover:opacity-100 transition-all text-white"
                  style={{ backgroundColor: "rgba(15,23,42,0.8)" }}
                >
                  <RefreshCw size={14} />
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Results */}
        <AnimatePresence>
          {result && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.5, ease: "easeOut" }}
              ref={reportRef}
              className="space-y-6"
            >
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Gauge */}
                <div className="rounded-3xl p-8 flex flex-col items-center justify-center glass"
                  style={{ border: "1px solid #1e293b" }}>
                  <PredictionGauge value={probability} size={220} />
                  <div className="mt-6 text-center">
                    <div className="text-[10px] font-bold uppercase tracking-widest mb-1" style={{ color: "#64748b" }}>Risk Level</div>
                    <div className="text-2xl font-bold" style={{ color: riskColor }}>{riskLevel}</div>
                  </div>
                  <div className="mt-4 text-center">
                    <div className="text-[10px] font-bold uppercase tracking-widest mb-1" style={{ color: "#64748b" }}>XGB Score</div>
                    <div className="text-sm font-mono text-white">{String(result.xgb_score)}</div>
                    <div className="text-[10px] font-bold uppercase tracking-widest mt-2 mb-1" style={{ color: "#64748b" }}>CNN Score</div>
                    <div className="text-sm font-mono text-white">{String(result.cnn_score)}</div>
                  </div>
                </div>

                {/* Clinical Summary */}
                <div className="lg:col-span-2 rounded-3xl p-8 glass" style={{ border: "1px solid #1e293b" }}>
                  <div className="flex items-center justify-between mb-8">
                    <h3 className="text-xl font-bold tracking-tight text-white">Clinical Summary</h3>
                    <button
                      onClick={downloadPDF}
                      className="flex items-center gap-2 text-xs font-bold transition-colors"
                      style={{ color: "#0ea5e9" }}
                    >
                      <Download size={14} />
                      DOWNLOAD PDF REPORT
                    </button>
                  </div>

                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-6 mb-8">
                    {[
                      { label: "Age", value: metadata.age },
                      { label: "BMI", value: metadata.bmi },
                      { label: "Total HDL", value: `${metadata.total_hdl} mg/dL` },
                      { label: "Total LDL", value: `${metadata.total_ldl} mg/dL` },
                      { label: "HDL/LDL Ratio", value: metadata.hdl_ldl_ratio.toFixed(2) },
                      { label: "sdLDL %", value: `${(metadata.sdldl_percent * 100).toFixed(1)}%` },
                      { label: "HDL-2b %", value: `${(metadata.hdl2b_percent * 100).toFixed(1)}%` },
                      { label: "HDL-2a %", value: `${(metadata.hdl2a_percent * 100).toFixed(1)}%` },
                      { label: "HDL-3a %", value: `${(metadata.hdl3a_percent * 100).toFixed(1)}%` },
                    ].map((stat, i) => (
                      <div key={i} className="space-y-1">
                        <p className="text-[10px] font-bold uppercase tracking-wider" style={{ color: "#64748b" }}>{stat.label}</p>
                        <p className="text-lg font-bold text-white">{stat.value}</p>
                      </div>
                    ))}
                  </div>

                  <div className="p-4 rounded-xl flex gap-3"
                    style={{ backgroundColor: "rgba(14,165,233,0.08)", border: "1px solid rgba(14,165,233,0.2)" }}>
                    <Info className="shrink-0 mt-0.5" style={{ color: "#0ea5e9" }} size={18} />
                    <p className="text-sm leading-relaxed" style={{ color: "#cbd5e1" }}>
                      <span className="font-bold" style={{ color: "#0ea5e9" }}>Clinical Interpretation: </span>
                      {result.clinical_insight ? String(result.clinical_insight) : (
                        <>
                          The multimodal ensemble predicts a{" "}
                          <span style={{ color: riskColor, fontWeight: 700 }}>{riskLevel}</span> risk of clinically significant plaque
                          ({(probability * 100).toFixed(1)}% probability). Analysis integrates {metadata.age}-year-old patient
                          metadata with high-resolution spectral electropherogram data.
                        </>
                      )}
                    </p>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Empty state */}
        {!result && !isPredicting && (
          <div className="flex flex-col items-center justify-center py-20 text-center">
            <div className="w-20 h-20 rounded-full flex items-center justify-center mb-6"
              style={{ backgroundColor: "#0f172a", border: "1px solid #1e293b" }}>
              <FileText style={{ color: "#334155" }} size={40} />
            </div>
            <h3 className="text-xl font-bold text-white mb-2">Awaiting Sample Input</h3>
            <p className="text-sm max-w-sm" style={{ color: "#64748b" }}>
              Upload HDL and LDL electropherogram <code className="text-xs bg-slate-800 px-1.5 py-0.5 rounded">.npy</code> files
              and fill in the patient metadata on the left to begin analysis.
            </p>
          </div>
        )}
      </main>
    </div>
  );
}
