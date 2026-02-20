import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Server, Grid, ArrowRight, CheckCircle } from 'lucide-react';
import Slide from '../components/Slide';

const STEPS = [
  { id: 'input', label: 'Raw Data', icon: Server, color: 'text-white' },
  { id: 'base', label: 'Base Forecast', icon: Grid, color: 'text-accent-tertiary' },
  { id: 'proj', label: 'Projection', icon: ArrowRight, color: 'text-accent' },
  { id: 'output', label: 'Reconciled', icon: CheckCircle, color: 'text-accent-secondary' }
];

export default function Slide05_Pipeline() {
  const [activeStep, setActiveStep] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveStep((prev) => (prev + 1) % STEPS.length);
    }, 2500);
    return () => clearInterval(interval);
  }, []);

  return (
    <Slide title="Pipeline: Flujo de Optimización">
      <div className="flex flex-col items-center justify-center h-full gap-12">
        {/* Animated Pipeline Path */}
        <div className="flex items-center gap-8 relative">
           {STEPS.map((step, index) => {
             const Icon = step.icon;
             const isActive = index === activeStep;
             const isPast = index < activeStep;

             return (
               <React.Fragment key={step.id}>
                 {/* Step Node */}
                 <div className="flex flex-col items-center gap-4 relative z-10">
                   <motion.div 
                     className={`w-24 h-24 rounded-full flex items-center justify-center border-4 ${isActive ? 'border-accent shadow-[0_0_30px_rgba(0,242,234,0.4)] bg-surface' : 'border-white/10 bg-black/40'}`}
                     animate={{ scale: isActive ? 1.1 : 1 }}
                   >
                     <Icon size={40} className={isActive ? step.color : 'text-white/20'} />
                   </motion.div>
                   <span className={`text-lg font-mono ${isActive ? 'text-white' : 'text-white/30'}`}>{step.label}</span>
                 </div>

                 {/* Connector (if not last) */}
                 {index < STEPS.length - 1 && (
                   <div className="w-32 h-2 bg-white/10 relative overflow-hidden rounded-full">
                     <motion.div 
                       className="absolute inset-0 bg-accent"
                       initial={{ x: '-100%' }}
                       animate={{ x: isPast ? '100%' : isActive ? '0%' : '-100%' }}
                       transition={{ duration: isActive ? 1.5 : 0, ease: 'linear' }}
                     />
                   </div>
                 )}
               </React.Fragment>
             );
           })}
        </div>

        {/* Detailed Explanation Area */}
        <div className="w-full max-w-4xl h-48 bg-surface/50 border border-white/5 rounded-xl p-8 relative overflow-hidden">
           <AnimatePresence mode="wait">
             <motion.div
               key={activeStep}
               initial={{ opacity: 0, y: 10 }}
               animate={{ opacity: 1, y: 0 }}
               exit={{ opacity: 0, y: -10 }}
               className="absolute inset-0 p-8 flex flex-col justify-center items-center text-center"
             >
                {activeStep === 0 && (
                  <p className="text-2xl">Ingesta de millones de series temporales crudas (S3/Redshift).</p>
                )}
                {activeStep === 1 && (
                  <p className="text-2xl">Generación de pronósticos base (ŷ) independientes. <br/><span className="text-accent-tertiary">Incoherentes.</span></p>
                )}
                {activeStep === 2 && (
                  <div className="space-y-2">
                    <p className="text-2xl font-bold text-accent">Proyección Ortogonal <em>P</em></p>
                     <p className="text-lg opacity-70">Ajuste de mínimos cuadrados generalizados en O(n log n).</p>
                  </div>
                )}
                {activeStep === 3 && (
                  <div className="space-y-2">
                     <p className="text-2xl font-bold text-accent-secondary">Salida Coherente ỹ</p>
                    <p className="text-lg opacity-70">Garantía de suma exacta en toda la jerarquía.</p>
                  </div>
                )}
             </motion.div>
           </AnimatePresence>
        </div>
      </div>
    </Slide>
  );
}
