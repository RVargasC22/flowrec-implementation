import React from 'react';
import { motion } from 'framer-motion';
import Slide from '../components/Slide';

export default function Slide08_DynT10T11() {
  return (
    <Slide title="Garantías Dinámicas: Resiliencia y Control">
      <div className="grid grid-cols-2 gap-6 h-full">

        {/* T10 — Disrupciones */}
        <motion.div
          className="bg-surface/30 rounded-xl border border-orange-400/20 p-5 flex flex-col gap-4"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
        >
          <div className="flex items-center gap-3">
            <span className="text-xs font-mono bg-orange-400/20 text-orange-400 px-2 py-1 rounded">Teorema 10</span>
            <h3 className="text-xl font-bold">Resiliencia ante Disrupciones</h3>
          </div>

          <p className="text-sm text-white/60 leading-relaxed">
            Cuando un nodo falla, FlowRec redistribuye coherentemente. El error de recuperación tiene <strong className="text-white">cota teórica garantizada</strong>.
          </p>

          <div className="flex-1 rounded-lg overflow-hidden bg-black/30 border border-white/5">
            <img
              src="/demo/t10.png"
              alt="T10 - Comparación de estrategias de disrupción"
              className="w-full h-full object-contain"
            />
          </div>

          <div className="flex gap-4">
            <div className="flex-1 bg-orange-400/10 border border-orange-400/20 rounded-lg p-3 text-center">
              <div className="text-2xl font-mono font-bold text-orange-400">4 / 4</div>
              <div className="text-xs text-white/50 mt-1">estrategias verificadas ✓</div>
            </div>
            <div className="flex-1 bg-orange-400/10 border border-orange-400/20 rounded-lg p-3 text-center">
              <div className="text-2xl font-mono font-bold text-orange-400">&lt; cota</div>
              <div className="text-xs text-white/50 mt-1">error real ≤ teórico</div>
            </div>
          </div>
        </motion.div>

        {/* T11 — ε-Aproximación */}
        <motion.div
          className="bg-surface/30 rounded-xl border border-purple-400/20 p-5 flex flex-col gap-4"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="flex items-center gap-3">
            <span className="text-xs font-mono bg-purple-400/20 text-purple-400 px-2 py-1 rounded">Teorema 11</span>
            <h3 className="text-xl font-bold">ε-Aproximación Controlada</h3>
          </div>

          <p className="text-sm text-white/60 leading-relaxed">
            Tú eliges el trade-off entre <strong className="text-white">precisión y velocidad</strong>. ε grande = más rápido; ε pequeño = más exacto.
          </p>

          <div className="flex-1 rounded-lg overflow-hidden bg-black/30 border border-white/5">
            <img
              src="/demo/t11.png"
              alt="T11 - Trade-off epsilon vs iteraciones"
              className="w-full h-full object-contain"
            />
          </div>

          <div className="flex gap-4">
            <div className="flex-1 bg-purple-400/10 border border-purple-400/20 rounded-lg p-3 text-center">
              <div className="text-2xl font-mono font-bold text-purple-400">1 iter.</div>
              <div className="text-xs text-white/50 mt-1">con ε = 0.05</div>
            </div>
            <div className="flex-1 bg-purple-400/10 border border-purple-400/20 rounded-lg p-3 text-center">
              <div className="text-2xl font-mono font-bold text-purple-400">3 iter.</div>
              <div className="text-xs text-white/50 mt-1">con ε = 0.001</div>
            </div>
          </div>
        </motion.div>

      </div>
    </Slide>
  );
}
