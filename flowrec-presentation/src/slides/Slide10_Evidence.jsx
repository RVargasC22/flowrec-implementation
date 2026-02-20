import React from 'react';
import { motion } from 'framer-motion';
import Slide from '../components/Slide';

export default function Slide10_Evidence() {
  return (
    <Slide title="Evidencia: Reproducible y Verificable">
      <div className="flex flex-col h-full gap-5">

        {/* Tabla benchmark */}
        <div className="flex-1 rounded-xl overflow-hidden bg-black/30 border border-white/10">
          <img
            src="/benchmarks/tabla.png"
            alt="Tabla resumen benchmark FlowRec vs MinT vs BottomUp"
            className="w-full h-full object-contain"
          />
        </div>

        {/* Fuentes */}
        <motion.div
          className="grid grid-cols-3 gap-4"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <div className="bg-surface/30 border border-white/10 rounded-lg p-4">
            <div className="text-xs text-accent font-mono mb-1">M5 · Walmart</div>
            <p className="text-sm text-white/60">datasetsforecast.m5 · 500 series reales · 1,969 pasos temporales</p>
          </div>
          <div className="bg-surface/30 border border-white/10 rounded-lg p-4">
            <div className="text-xs text-accent-secondary font-mono mb-1">TourismLarge</div>
            <p className="text-sm text-white/60">HierarchicalData('TourismLarge') · 555 nodos · 304 series base · 228 meses</p>
          </div>
          <div className="bg-surface/30 border border-white/10 rounded-lg p-4">
            <div className="text-xs text-accent-tertiary font-mono mb-1">Traffic · SF Bay Area</div>
            <p className="text-sm text-white/60">HierarchicalData('Traffic') · 207 nodos · 200 series base · 366 días</p>
          </div>
        </motion.div>

        <motion.div
          className="text-center text-xs text-white/30 font-mono"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
        >
          Reproducir: <span className="text-accent/60">python flowrec_benchmark_real.py</span> · Datos públicos vía datasetsforecast
        </motion.div>

      </div>
    </Slide>
  );
}
