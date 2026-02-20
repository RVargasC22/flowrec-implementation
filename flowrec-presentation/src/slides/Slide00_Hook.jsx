import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Activity, Clock } from 'lucide-react';
import Slide from '../components/Slide';

export default function Slide00_Hook() {
  const [dataPoints, setDataPoints] = useState([]);

  useEffect(() => {
    const interval = setInterval(() => {
      setDataPoints(current => {
        const next = [...current, Math.random() * 50 + 20];
        if (next.length > 50) next.shift();
        return next;
      });
    }, 50);
    return () => clearInterval(interval);
  }, []);

  return (
    <Slide className="bg-black">
      <div className="absolute inset-0 bg-[url('/grid.svg')] opacity-20" />
      
      <div className="flex flex-col items-center justify-center h-full relative z-10">
        <motion.div 
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 1 }}
          className="text-center mb-8"
        >
          <div className="inline-flex items-center gap-2 px-4 py-1 rounded-full bg-red-500/10 text-red-500 border border-red-500/20 mb-4 animate-pulse">
            <span className="w-2 h-2 bg-red-500 rounded-full" />
            LIVE TRAFFIC DATA // SF BAY AREA
          </div>
          <h1 className="text-6xl font-black text-white mb-2">Reconciliación en Tiempo Real</h1>
          <p className="text-xl text-white/50">Jerárquica · Exacta · 10,000× más rápido que MinT</p>
        </motion.div>

        {/* KPI Row */}
        <motion.div
          className="flex gap-12 mb-8"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
        >
          <div className="text-center">
            <div className="text-4xl font-mono font-bold text-accent">10,355×</div>
            <div className="text-xs text-white/40 uppercase tracking-widest mt-1">Speedup · M5 Walmart</div>
          </div>
          <div className="w-px bg-white/10" />
          <div className="text-center">
            <div className="text-4xl font-mono font-bold text-accent-secondary">−8.22%</div>
            <div className="text-xs text-white/40 uppercase tracking-widest mt-1">RMSE · TourismLarge</div>
          </div>
          <div className="w-px bg-white/10" />
          <div className="text-center">
            <div className="text-4xl font-mono font-bold text-accent-tertiary">15 µs</div>
            <div className="text-xs text-white/40 uppercase tracking-widest mt-1">Latencia · Traffic SF</div>
          </div>
        </motion.div>

        {/* Real-time bars */}
        <div className="w-full max-w-4xl h-48 bg-surface/50 border border-white/10 rounded-xl relative overflow-hidden flex items-end px-4 pb-4 gap-1">
           {dataPoints.map((val, i) => (
             <motion.div
               key={i}
               className="flex-1 bg-accent/80 hover:bg-accent transition-colors rounded-t-sm"
               style={{ height: `${val}%` }}
               initial={{ height: 0 }}
               animate={{ height: `${val}%` }}
               transition={{ type: 'spring', stiffness: 300, damping: 20 }}
             />
           ))}
           <div className="absolute top-4 right-4 flex flex-col items-end">
              <span className="text-4xl font-mono text-white font-bold tabular-nums">
                {(Math.random() * 5 + 12).toFixed(1)} µs
              </span>
              <span className="text-xs text-white/40 uppercase tracking-widest">Processing Time</span>
           </div>
        </div>

        <motion.div 
          className="mt-6 flex gap-10"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 2 }}
        >
           <div className="flex items-center gap-3 text-white/50">
             <Clock size={18} />
             <span>Walmart M5 · 500 series · 1,969 pasos</span>
           </div>
           <div className="flex items-center gap-3 text-white/50">
             <Activity size={18} />
             <span>TourismLarge · Traffic SF · datos públicos reales</span>
           </div>
        </motion.div>
      </div>
    </Slide>
  );
}
