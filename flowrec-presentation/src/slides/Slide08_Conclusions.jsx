import React from 'react';
import { motion } from 'framer-motion';
import { Check, Zap, Layers, ExternalLink } from 'lucide-react';
import Slide from '../components/Slide';

const STATS = [
  { value: '10,355×', label: 'Speedup', sub: 'M5 · Walmart', color: 'text-accent' },
  { value: '−8.22%', label: 'RMSE',    sub: 'TourismLarge',  color: 'text-accent-secondary' },
  { value: '15 µs',  label: 'Latencia',sub: 'Traffic SF',    color: 'text-accent-tertiary' },
];

export default function Slide11_Conclusions() {
  return (
    <Slide title="Conclusiones">
      <div className="flex flex-col h-full gap-5">

        {/* ── A: Strip de números clave ── */}
        <motion.div
          className="grid grid-cols-3 gap-4"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          {STATS.map((s, i) => (
            <motion.div
              key={s.label}
              className="bg-surface/20 border border-white/10 rounded-xl py-3 px-5 flex items-center gap-4"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.15 + i * 0.1 }}
            >
              <div className={`text-3xl font-mono font-bold ${s.color}`}>{s.value}</div>
              <div>
                <div className="text-sm font-semibold text-white/80">{s.label}</div>
                <div className="text-xs text-white/40">{s.sub}</div>
              </div>
            </motion.div>
          ))}
        </motion.div>

        {/* ── B: 3 tarjetas ── */}
        <div className="grid grid-cols-3 gap-5 flex-1">
          <motion.div
            className="bg-surface/30 p-6 rounded-2xl border border-green-500/30 flex flex-col items-center text-center gap-4 group hover:bg-green-500/10 transition-colors"
            initial={{ y: 40, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.35 }}
          >
            <div className="w-16 h-16 rounded-full bg-green-500/20 flex items-center justify-center text-green-400 group-hover:scale-110 transition-transform">
              <Check size={36} strokeWidth={3} />
            </div>
            <h3 className="text-2xl font-bold text-white">Validado</h3>
            <p className="text-white/60 text-sm">T8–T11 probados empíricamente. 9/9 series monótonas. 4/4 estrategias de disrupción verificadas.</p>
          </motion.div>

          <motion.div
            className="bg-surface/30 p-6 rounded-2xl border border-accent/30 flex flex-col items-center text-center gap-4 group hover:bg-accent/10 transition-colors"
            initial={{ y: 40, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.5 }}
          >
            <div className="w-16 h-16 rounded-full bg-accent/20 flex items-center justify-center text-accent group-hover:scale-110 transition-transform">
              <Zap size={36} strokeWidth={3} />
            </div>
            <h3 className="text-2xl font-bold text-white">hasta 12,749×</h3>
            <p className="text-white/60 text-sm">O(n log n) vs O(n³) de MinT-OLS. Verificado en M5, TourismLarge y Traffic SF.</p>
          </motion.div>

          <motion.div
            className="bg-surface/30 p-6 rounded-2xl border border-accent-tertiary/30 flex flex-col items-center text-center gap-4 group hover:bg-accent-tertiary/10 transition-colors"
            initial={{ y: 40, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.65 }}
          >
            <div className="w-16 h-16 rounded-full bg-accent-tertiary/20 flex items-center justify-center text-accent-tertiary group-hover:scale-110 transition-transform">
              <Layers size={36} strokeWidth={3} />
            </div>
            <h3 className="text-2xl font-bold text-white">Modular</h3>
            <p className="text-white/60 text-sm">Arquitectura DAG flexible. Soporta cualquier topología jerárquica. Nodos añadibles en O(k).</p>
          </motion.div>
        </div>

        {/* ── C: Tagline + referencias ── */}
        <motion.div
          className="flex items-center justify-between px-2"
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1 }}
        >
          <p className="text-white/40 text-lg">FlowRec está listo para producción.</p>
          <div className="flex gap-4 text-xs text-white/30">
            <a
              href="https://arxiv.org/abs/2505.03955"
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-1.5 hover:text-accent transition-colors"
            >
              <ExternalLink size={12} />
              arXiv:2505.03955
            </a>
            <span className="text-white/10">·</span>
            <a
              href="https://github.com/RVargasC22/flowrec-implementation"
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-1.5 hover:text-accent transition-colors"
            >
              <ExternalLink size={12} />
              github.com/RVargasC22
            </a>
          </div>
        </motion.div>

      </div>
    </Slide>
  );
}
