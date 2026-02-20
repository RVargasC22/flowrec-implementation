import React, { useState, useEffect } from 'react';
import { AnimatePresence } from 'framer-motion';

// Slides Imports
import Slide00_Hook       from './slides/Slide00_Hook';
import Slide01_Title      from './slides/Slide01_Title';
import Slide02_Problem    from './slides/Slide02_Problem';
import Slide03_StateArt   from './slides/Slide03_StateArt';
import Slide04_FlowRecNet from './slides/Slide04_FlowRecNet';
import Slide05_Pipeline   from './slides/Slide05_Pipeline';
import Slide06_Theorems   from './slides/Slide06_Theorems';
import Slide07_DynT8T9   from './slides/Slide07_DynT8T9';
import Slide08_DynT10T11  from './slides/Slide08_DynT10T11';
import Slide09_Results    from './slides/Slide07_Results';      // archivo reutilizado
import Slide10_Evidence   from './slides/Slide10_Evidence';
import Slide11_Conclusions from './slides/Slide08_Conclusions'; // archivo reutilizado
import Slide12_End        from './slides/Slide09_End';

const SLIDES = [
  { id: 'hook',        component: Slide00_Hook,        label: '0 · Hook' },
  { id: 'title',       component: Slide01_Title,        label: '1 · Título' },
  { id: 'problem',     component: Slide02_Problem,      label: '2 · Problema' },
  { id: 'soa',         component: Slide03_StateArt,     label: '3 · Estado del Arte' },
  { id: 'arch',        component: Slide04_FlowRecNet,   label: '4 · Arquitectura DAG' },
  { id: 'pipeline',    component: Slide05_Pipeline,     label: '5 · Pipeline' },
  { id: 'theorems',    component: Slide06_Theorems,     label: '6 · Garantías T1–T7' },
  { id: 'dyn-t8-t9',  component: Slide07_DynT8T9,     label: '7 · Dinámico T8+T9' },
  { id: 'dyn-t10-t11', component: Slide08_DynT10T11,   label: '8 · Dinámico T10+T11' },
  { id: 'results',     component: Slide09_Results,      label: '9 · Resultados' },
  { id: 'evidence',    component: Slide10_Evidence,     label: '10 · Evidencia' },
  { id: 'conclusions', component: Slide11_Conclusions,  label: '11 · Conclusiones' },
  { id: 'end',         component: Slide12_End,          label: '12 · Q&A' },
];

export default function App() {
  const [current, setCurrent] = useState(0);

  useEffect(() => {
    const handleKey = (e) => {
      if (e.key === 'ArrowRight' || e.key === 'PageDown') setCurrent(c => Math.min(c + 1, SLIDES.length - 1));
      if (e.key === 'ArrowLeft'  || e.key === 'PageUp')   setCurrent(c => Math.max(c - 1, 0));
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, []);

  const SlideComponent = SLIDES[current].component;

  return (
    <div className="fixed inset-0 bg-background overflow-hidden flex flex-col select-none">
      {/* Slide Area */}
      <div className="flex-1 relative">
        <AnimatePresence mode="wait">
          <SlideComponent key={current} />
        </AnimatePresence>
      </div>

      {/* Nav Bar */}
      <div className="flex items-center justify-between px-8 py-2 bg-black/60 border-t border-white/5 z-50">
        <button
          onClick={() => setCurrent(c => Math.max(c - 1, 0))}
          disabled={current === 0}
          className="px-4 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 disabled:opacity-20 transition-colors text-sm"
        >
          ← Anterior
        </button>

        {/* Dot navigation */}
        <div className="flex gap-2 items-center">
          {SLIDES.map((s, i) => (
            <button
              key={s.id}
              onClick={() => setCurrent(i)}
              title={s.label}
              className={`rounded-full transition-all duration-300 ${
                i === current
                  ? 'w-6 h-2.5 bg-accent'
                  : 'w-2.5 h-2.5 bg-white/20 hover:bg-white/40'
              }`}
            />
          ))}
        </div>

        <div className="flex items-center gap-6">
          <span className="text-xs text-white/30 font-mono">{SLIDES[current].label}</span>
          <button
            onClick={() => setCurrent(c => Math.min(c + 1, SLIDES.length - 1))}
            disabled={current === SLIDES.length - 1}
            className="px-4 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 disabled:opacity-20 transition-colors text-sm"
          >
            Siguiente →
          </button>
        </div>
      </div>
    </div>
  );
}
