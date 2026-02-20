import React from 'react';
import { motion } from 'framer-motion';
import Slide from '../components/Slide';

export default function Slide09_End() {
  return (
    <Slide className="items-center justify-center bg-black">
        <motion.div 
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 1 }}
          className="text-center"
        >
             <h1 className="text-9xl font-black text-transparent bg-clip-text bg-gradient-to-br from-white via-accent-secondary to-accent mb-8">
                 Gracias
             </h1>
             <p className="text-2xl text-white/50 mb-12">¿Preguntas?</p>
             
             <div className="flex justify-center gap-8">
                <a href="https://github.com/RVargasC22/flowrec-implementation" target="_blank" className="px-6 py-3 rounded-full bg-surface border border-white/10 hover:border-accent hover:text-accent transition-colors flex items-center gap-3">
                   <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path></svg>
                   github.com/RVargasC22
                </a>
             </div>
        </motion.div>
    </Slide>
  );
}
