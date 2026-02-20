import React from 'react';
import { motion } from 'framer-motion';
import { ShieldCheck, Zap, Target, TrendingDown } from 'lucide-react';
import Slide from '../components/Slide';

const THEOREMS = [
  { 
    id: 'T1', 
    title: 'Coherencia Exacta', 
    desc: 'La salida ỹ satisface S·ỹ_base = ỹ_agg para TODOS los nodos del DAG. Error de suma = 0.',
    icon: ShieldCheck,
    color: 'text-blue-400',
    border: 'border-blue-400/20',
    detail: '∑ hijos = padre · garantizado',
  },
  { 
    id: 'T3', 
    title: 'Complejidad O(n log n)', 
    desc: 'Explota la dispersidad de S. Frente a la inversión densa O(n³) de MinT.',
    icon: TrendingDown,
    color: 'text-yellow-400',
    border: 'border-yellow-400/20',
    detail: 'vs O(n³) de MinT-OLS',
  },
  { 
    id: 'T5', 
    title: 'Optimalidad L2', 
    desc: 'P es la proyección ortogonal al subespacio de coherencia. Minimiza ||ỹ − ŷ||² globalmente.',
    icon: Target,
    color: 'text-green-400',
    border: 'border-green-400/20',
    detail: 'mínimo cuadrados óptimo',
  },
  { 
    id: 'T7', 
    title: 'Convergencia Global', 
    desc: 'El algoritmo converge al mínimo global bajo pérdida L2 y cualquier topología DAG.',
    icon: Zap,
    color: 'text-purple-400',
    border: 'border-purple-400/20',
    detail: 'sin mínimos locales',
  },
];

export default function Slide06_Theorems() {
  return (
    <Slide title="Garantías Teóricas (T1, T3, T5, T7)">
      <div className="grid grid-cols-2 gap-6 h-full">
        {THEOREMS.map((theorem, idx) => {
          const Icon = theorem.icon;
          return (
            <motion.div 
              key={theorem.id}
              className={`bg-surface/40 p-6 rounded-xl border ${theorem.border} hover:bg-surface/60 transition-colors group relative overflow-hidden`}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.15 }}
            >
              <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity">
                 <Icon size={100} />
              </div>
              
              <div className="flex items-center gap-4 mb-3">
                <div className={`p-3 rounded-lg bg-black/30 ${theorem.color}`}>
                   <Icon size={28} />
                </div>
                <div>
                   <span className="text-xs font-mono opacity-40 block">Teorema {theorem.id}</span>
                   <h3 className="text-xl font-bold">{theorem.title}</h3>
                </div>
              </div>
              
              <p className="text-base text-white/70 leading-relaxed z-10 relative mb-3">
                {theorem.desc}
              </p>

              <div className={`text-xs font-mono ${theorem.color} opacity-70 bg-black/30 px-3 py-1 rounded inline-block`}>
                {theorem.detail}
              </div>
            </motion.div>
          );
        })}
      </div>
    </Slide>
  );
}
