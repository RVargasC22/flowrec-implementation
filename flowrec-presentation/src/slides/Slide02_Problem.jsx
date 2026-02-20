import React from 'react';
import { motion } from 'framer-motion';
import { AlertTriangle, TrendingUp } from 'lucide-react';
import Slide from '../components/Slide';

const listVariants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.25 } },
};
const itemVariants = {
  hidden: { x: -30, opacity: 0 },
  visible: { x: 0, opacity: 1, transition: { duration: 0.5 } },
};

export default function Slide02_Problem() {
  return (
    <Slide title="El Problema: Incoherencia Jerárquica">
      <div className="flex h-full gap-10">

        {/* ── Left: Text ── */}
        <div className="flex-1 flex flex-col justify-center">
          <motion.ul
            className="space-y-7 text-xl"
            initial="hidden"
            animate="visible"
            variants={listVariants}
          >
            <motion.li className="flex items-start gap-4" variants={itemVariants}>
              <TrendingUp className="text-accent mt-1 shrink-0" size={30} />
              <span>
                Los modelos de ML optimizan cada serie{' '}
                <strong className="text-white">localmente</strong>{' '}
                — sin ver el todo.
              </span>
            </motion.li>
            <motion.li className="flex items-start gap-4" variants={itemVariants}>
              <AlertTriangle className="text-accent-secondary mt-1 shrink-0" size={30} />
              <span>
                Resultado:{' '}
                <code className="bg-surface px-2 py-1 rounded text-accent-secondary text-lg">
                  Sum(Hijos) ≠ Padre
                </code>
              </span>
            </motion.li>
            <motion.li className="flex items-start gap-4" variants={itemVariants}>
              <span className="w-2.5 h-2.5 rounded-full bg-accent mt-2.5 shrink-0" />
              <span>
                Sistemas financieros requieren{' '}
                <span className="text-accent font-bold">coherencia exacta</span>.
              </span>
            </motion.li>
            <motion.li className="flex items-start gap-4" variants={itemVariants}>
              <span className="w-2.5 h-2.5 rounded-full bg-accent mt-2.5 shrink-0" />
              <span>Métodos manuales (Pro-rateo) destruyen la precisión.</span>
            </motion.li>
          </motion.ul>

          {/* Impact callout */}
          <motion.div
            className="mt-8 border-l-4 border-amber-400 bg-amber-400/10 rounded-r-xl px-5 py-4"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 1.2, duration: 0.6 }}
          >
            <p className="text-amber-300 text-lg leading-snug">
              <span className="font-bold">En Amazon</span>, esto ocurre con
              millones de SKUs simultáneos.{' '}
              <span className="font-bold">Un error de +100 se multiplica
              en toda la jerarquía.</span>
            </p>
          </motion.div>
        </div>

        {/* ── Right: Tree + Error box stacked ── */}
        <div className="flex-1 flex flex-col items-center justify-center gap-6">

          {/* SVG: solo el árbol, viewBox compacto */}
          <svg viewBox="0 0 300 180" className="w-full max-w-sm">
            {/* Edges */}
            <motion.path d="M150 40 L70 130" stroke="#555" strokeWidth="4"
              initial={{ pathLength: 0 }} animate={{ pathLength: 1 }}
              transition={{ duration: 0.7 }} />
            <motion.path d="M150 40 L230 130" stroke="#555" strokeWidth="4"
              initial={{ pathLength: 0 }} animate={{ pathLength: 1 }}
              transition={{ duration: 0.7, delay: 0.1 }} />

            {/* Parent */}
            <motion.g initial={{ scale: 0, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.15, type: 'spring' }}>
              <circle cx="150" cy="40" r="35" fill="#1e212b" stroke="#ff0055" strokeWidth="3.5" className="animate-pulse" />
              <text x="150" y="46" textAnchor="middle" fill="#fff" fontSize="16" fontWeight="bold">1000</text>
            </motion.g>

            {/* Child 400 */}
            <motion.g initial={{ scale: 0, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.5, type: 'spring' }}>
              <circle cx="70" cy="130" r="30" fill="#1e212b" stroke="#555" strokeWidth="2.5" />
              <text x="70" y="136" textAnchor="middle" fill="#fff" fontSize="14">400</text>
            </motion.g>

            {/* Child 700 */}
            <motion.g initial={{ scale: 0, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.65, type: 'spring' }}>
              <circle cx="230" cy="130" r="30" fill="#1e212b" stroke="#555" strokeWidth="2.5" />
              <text x="230" y="136" textAnchor="middle" fill="#fff" fontSize="14">700</text>
            </motion.g>
          </svg>

          {/* Error box: HTML independiente — nunca solapa el árbol */}
          <motion.div
            className="w-full max-w-sm rounded-xl border border-red-500/40 bg-red-500/10 px-5 py-4"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1, duration: 0.6 }}
          >
            <p className="text-red-400 font-bold text-base mb-1">⚠ Incoherencia detectada</p>
            <p className="text-red-400 text-sm">Suma hijos: 400 + 700 = <strong>1100</strong></p>
            <p className="text-red-400 text-sm">Diferencia con padre: <strong>+100 ≠ 0</strong></p>
          </motion.div>

        </div>
      </div>
    </Slide>
  );
}
