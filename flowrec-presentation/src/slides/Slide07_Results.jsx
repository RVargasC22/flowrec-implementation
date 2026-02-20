import React from 'react';
import Slide from '../components/Slide';
import { motion } from 'framer-motion';

// Resultados verificados — flowrec_benchmark_real.py (2025)
// Datasets REALES: M5 (500 series Walmart), TourismLarge (555 nodos), Traffic SF (207 nodos)
const RESULTS = {
  m5:      { speedup: '10,355×', rmse: '-1.04%', lat: '167 µs' },
  tourism: { speedup: '1,342×',  rmse: '-8.22%', lat: '281 µs' },
  traffic: { speedup: '12,749×', rmse: '-6.85%', lat: '15 µs'  },
};

export default function Slide09_Results() {
  return (
    <Slide title="Resultados Experimentales — Datos Reales">
      <div className="flex flex-col h-full gap-4">

        {/* KPI Row */}
        <div className="grid grid-cols-3 gap-4">
          {/* M5 */}
          <motion.div
            className="bg-surface/30 rounded-xl p-4 border border-accent/20 text-center"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            <h3 className="text-base font-bold mb-0.5">M5 (Walmart)</h3>
            <p className="text-xs opacity-40 mb-2">500 series · 1,969 pasos</p>
            <div className="text-4xl font-mono text-accent font-bold">{RESULTS.m5.speedup}</div>
            <div className="text-xs opacity-50 mt-1">Speedup vs MinT-OLS</div>
            <div className="flex gap-4 justify-center text-xs mt-2">
              <span className="text-accent/80">RMSE <strong>{RESULTS.m5.rmse}</strong></span>
              <span className="opacity-40">lat {RESULTS.m5.lat}</span>
            </div>
          </motion.div>

          {/* Tourism */}
          <motion.div
            className="bg-surface/30 rounded-xl p-4 border border-accent-secondary/20 text-center"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <h3 className="text-base font-bold mb-0.5">Tourism Large</h3>
            <p className="text-xs opacity-40 mb-2">555 nodos · 304 base · 228 meses</p>
            <div className="text-4xl font-mono text-accent-secondary font-bold">{RESULTS.tourism.rmse}</div>
            <div className="text-xs opacity-50 mt-1">RMSE vs Base</div>
            <div className="flex gap-4 justify-center text-xs mt-2">
              <span className="text-accent-secondary/80">speedup <strong>{RESULTS.tourism.speedup}</strong></span>
              <span className="opacity-40">lat {RESULTS.tourism.lat}</span>
            </div>
          </motion.div>

          {/* Traffic */}
          <motion.div
            className="bg-surface/30 rounded-xl p-4 border border-accent-tertiary/20 text-center"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <h3 className="text-base font-bold mb-0.5">Traffic (SF)</h3>
            <p className="text-xs opacity-40 mb-2">207 nodos · 200 base · 366 días</p>
            <div className="text-4xl font-mono text-accent-tertiary font-bold">{RESULTS.traffic.lat}</div>
            <div className="text-xs opacity-50 mt-1">Latencia FlowRec</div>
            <div className="flex gap-4 justify-center text-xs mt-2">
              <span className="text-accent-tertiary/80">speedup <strong>{RESULTS.traffic.speedup}</strong></span>
              <span className="opacity-40">RMSE {RESULTS.traffic.rmse}</span>
            </div>
          </motion.div>
        </div>

        {/* Charts Row */}
        <div className="grid grid-cols-2 gap-4 flex-1">
          <motion.div
            className="rounded-xl overflow-hidden bg-black/30 border border-white/10 flex flex-col"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4 }}
          >
            <div className="px-4 pt-3 pb-1 text-xs text-white/40 font-mono">RMSE comparativo — FlowRec vs MinT-OLS vs BottomUp</div>
            <img
              src="/benchmarks/rmse.png"
              alt="Comparativa de RMSE en los 3 datasets"
              className="flex-1 w-full object-contain"
            />
          </motion.div>

          <motion.div
            className="rounded-xl overflow-hidden bg-black/30 border border-white/10 flex flex-col"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
          >
            <div className="px-4 pt-3 pb-1 text-xs text-white/40 font-mono">Latencia y Speedup — escala logarítmica</div>
            <img
              src="/benchmarks/speedup.png"
              alt="Análisis de latencia y speedup"
              className="flex-1 w-full object-contain"
            />
          </motion.div>
        </div>

      </div>
    </Slide>
  );
}
