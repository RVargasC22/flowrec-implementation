import React from 'react';
import { motion } from 'framer-motion';
import Slide from '../components/Slide';

export default function Slide07_DynT8T9() {
  return (
    <Slide title="Garantías Dinámicas: Escalabilidad y Mejora">
      <div className="grid grid-cols-2 gap-6 h-full">

        {/* T8 — Expansión */}
        <motion.div
          className="bg-surface/30 rounded-xl border border-blue-400/20 p-5 flex flex-col gap-4"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
        >
          <div className="flex items-center gap-3">
            <span className="text-xs font-mono bg-blue-400/20 text-blue-400 px-2 py-1 rounded">Teorema 8</span>
            <h3 className="text-xl font-bold">Expansión Dinámica de Red</h3>
          </div>

          <p className="text-sm text-white/60 leading-relaxed">
            Agregar un nodo nuevo a la jerarquía sólo requiere recalcular el <strong className="text-white">camino ancestral</strong> — no toda la red.
          </p>

          {/* Imagen real del demo */}
          <div className="flex-1 rounded-lg overflow-hidden bg-black/30 border border-white/5">
            <img
              src="/demo/t8.png"
              alt="T8 - Expansión de red: antes y después"
              className="w-full h-full object-contain"
            />
          </div>

          <div className="flex gap-4">
            <div className="flex-1 bg-blue-400/10 border border-blue-400/20 rounded-lg p-3 text-center">
              <div className="text-2xl font-mono font-bold text-blue-400">86%</div>
              <div className="text-xs text-white/50 mt-1">nodos sin recalcular</div>
            </div>
            <div className="flex-1 bg-blue-400/10 border border-blue-400/20 rounded-lg p-3 text-center">
              <div className="text-2xl font-mono font-bold text-blue-400">O(k)</div>
              <div className="text-xs text-white/50 mt-1">costo por nodo nuevo</div>
            </div>
          </div>
        </motion.div>

        {/* T9 — Monotonicidad */}
        <motion.div
          className="bg-surface/30 rounded-xl border border-green-400/20 p-5 flex flex-col gap-4"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="flex items-center gap-3">
            <span className="text-xs font-mono bg-green-400/20 text-green-400 px-2 py-1 rounded">Teorema 9</span>
            <h3 className="text-xl font-bold">Monotonicidad</h3>
          </div>

          <p className="text-sm text-white/60 leading-relaxed">
            Si los pronósticos base mejoran, la reconciliación <strong className="text-white">siempre mejora</strong>. Garantía válida para toda la jerarquía.
          </p>

          {/* Imagen real del demo */}
          <div className="flex-1 rounded-lg overflow-hidden bg-black/30 border border-white/5">
            <img
              src="/demo/t9.png"
              alt="T9 - Trayectorias de mejora y monotonicidad"
              className="w-full h-full object-contain"
            />
          </div>

          <div className="flex gap-4">
            <div className="flex-1 bg-green-400/10 border border-green-400/20 rounded-lg p-3 text-center">
              <div className="text-2xl font-mono font-bold text-green-400">9 / 9</div>
              <div className="text-xs text-white/50 mt-1">series verificadas ✓</div>
            </div>
            <div className="flex-1 bg-green-400/10 border border-green-400/20 rounded-lg p-3 text-center">
              <div className="text-2xl font-mono font-bold text-green-400">100%</div>
              <div className="text-xs text-white/50 mt-1">monotonicidad global</div>
            </div>
          </div>
        </motion.div>

      </div>
    </Slide>
  );
}
