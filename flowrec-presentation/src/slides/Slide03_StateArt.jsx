import React from 'react';
import { motion } from 'framer-motion';
import { XCircle, CheckCircle } from 'lucide-react';
import Slide from '../components/Slide';

const METHODS = [
  {
    name: 'BottomUp',
    formula: 'ỹ = S · ŷ_base',
    pros: 'Rápido O(n)',
    items: [
      { text: 'Impreciso en niveles superiores', ok: false },
      { text: 'No minimiza error global', ok: false },
      { text: 'Solo árboles jerárquicos', ok: false },
    ],
    color: 'border-yellow-500/30 bg-yellow-500/5',
    tag: 'text-yellow-400',
  },
  {
    name: 'MinT-OLS',
    year: '2019',
    formula: 'ỹ = S(SᵀW⁻¹S)⁻¹SᵀW⁻¹ŷ',
    pros: 'Óptimo bajo L2',
    items: [
      { text: 'Complejidad O(n³) — inescalable', ok: false },
      { text: 'Solo árboles rígidos', ok: false },
      { text: 'Estático: recálculo total', ok: false },
    ],
    color: 'border-red-500/30 bg-red-500/5',
    tag: 'text-red-400',
  },
  {
    name: 'FlowRec',
    year: '2025',
    formula: 'ỹ = P · ŷ  [proyección ortogonal]',
    pros: 'Óptimo + Rápido + Dinámico',
    items: [
      { text: 'O(n log n) — explota sparsity', ok: true },
      { text: 'DAG arbitrario (ciclos, multi-padre)', ok: true },
      { text: 'Dinámico: O(k) por nodo nuevo', ok: true },
    ],
    color: 'border-accent/40 bg-accent/5',
    tag: 'text-accent',
    isOurs: true,
  },
];

// Complejidad escalada visualmente para n=1000 (valores relativos)
const COMPLEXITY = [
  { label: 'BottomUp',  complexity: 'O(n)',       bar: 8,   color: '#f59e0b', textColor: 'text-yellow-400' },
  { label: 'MinT-OLS',  complexity: 'O(n³)',      bar: 100, color: '#f87171', textColor: 'text-red-400' },
  { label: 'FlowRec',   complexity: 'O(n log n)', bar: 11,  color: '#00f2ea', textColor: 'text-accent' },
];

export default function Slide03_StateArt() {
  return (
    <Slide title="Estado del Arte: ¿Por qué Fallan?">
      <div className="flex flex-col h-full gap-4">

        {/* ── Tarjetas (fila superior) ── */}
        <div className="grid grid-cols-3 gap-4 flex-1">
          {METHODS.map((m, idx) => (
            <motion.div
              key={m.name}
              className={`rounded-xl border p-4 flex flex-col gap-3 ${m.color}`}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.15 }}
            >
              <div className="flex items-baseline gap-2">
                <h3 className={`text-xl font-bold ${m.tag}`}>{m.name}</h3>
                {m.year && <span className="text-xs text-white/40">({m.year})</span>}
              </div>

              <div className="bg-black/40 px-3 py-1.5 rounded font-mono text-xs text-white/70 border-l-2 border-white/20">
                {m.formula}
              </div>

              <div className={`text-xs font-semibold ${m.tag}`}>{m.pros}</div>

              <ul className="space-y-2 flex-1">
                {m.items.map((item, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm">
                    {item.ok
                      ? <CheckCircle size={14} className="text-accent mt-0.5 shrink-0" />
                      : <XCircle size={14} className="text-red-400 mt-0.5 shrink-0" />
                    }
                    <span className={item.ok ? 'text-white/80' : 'text-white/50'}>{item.text}</span>
                  </li>
                ))}
              </ul>

              {m.isOurs && (
                <div className="pt-2 border-t border-accent/20 text-center text-xs text-accent font-bold tracking-widest uppercase">
                  ★ Nuestra Propuesta
                </div>
              )}
            </motion.div>
          ))}
        </div>

        {/* ── Gráfica comparativa de complejidad (fila inferior) ── */}
        <motion.div
          className="bg-surface/30 border border-white/10 rounded-xl px-6 py-4 flex flex-col gap-3"
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
        >
          <p className="text-xs text-white/40 font-mono uppercase tracking-widest">
            Comparativa de complejidad computacional — para n = 1,000 nodos (escala logarítmica)
          </p>
          <div className="flex flex-col gap-2.5">
            {COMPLEXITY.map((c, i) => (
              <div key={c.label} className="flex items-center gap-4">
                <span className={`w-24 text-sm font-semibold shrink-0 ${c.textColor}`}>{c.label}</span>
                <div className="flex-1 h-7 bg-white/5 rounded-lg overflow-hidden relative">
                  <motion.div
                    className="h-full rounded-lg"
                    style={{ backgroundColor: c.color + '55', borderRight: `3px solid ${c.color}` }}
                    initial={{ width: 0 }}
                    animate={{ width: `${c.bar}%` }}
                    transition={{ delay: 0.8 + i * 0.15, duration: 0.7, ease: 'easeOut' }}
                  />
                  <span
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-xs font-mono"
                    style={{ color: c.color }}
                  >
                    {c.complexity}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </motion.div>

      </div>
    </Slide>
  );
}
