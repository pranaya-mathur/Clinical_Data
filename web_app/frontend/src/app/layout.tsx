import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "LipoSpec Plaque Predictor",
  description: "AI-powered cardiovascular plaque risk assessment using LipoSpec spectral analysis",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full" style={{ backgroundColor: "#0f172a" }}>
      <body className={`${inter.className} min-h-full antialiased`} style={{ backgroundColor: "#0f172a", color: "#e2e8f0" }}>
        {children}
      </body>
    </html>
  );
}
