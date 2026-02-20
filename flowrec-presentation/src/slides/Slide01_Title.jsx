import React from 'react';
import { motion } from 'framer-motion';
import Slide from '../components/Slide';

export default function Slide01_Title() {
  return (
    <Slide className="justify-center items-center text-center">
      <div className="relative w-64 h-64 mb-12">
        {/* Animated Network Logo */}
        <motion.svg 
          viewBox="0 0 200 200" 
          className="w-full h-full drop-shadow-[0_0_20px_rgba(0,242,234,0.3)]"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1 }}
        >
          {/* Ring 1 */}
          <motion.circle cx="100" cy="100" r="80" fill="none" stroke="#00f2ea" strokeWidth="2" strokeDasharray="10 5" opacity="0.5"
            animate={{ rotate: 360 }}
            transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
          />
          {/* Ring 2 */}
          <motion.circle cx="100" cy="100" r="60" fill="none" stroke="#ff0055" strokeWidth="2" strokeDasharray="5 5" opacity="0.7"
             animate={{ rotate: -360 }}
             transition={{ duration: 15, repeat: Infinity, ease: "linear" }}
          />
          
          {/* Main Nodes */}
          <motion.circle cx="100" cy="100" r="10" fill="#00f2ea" 
              animate={{ scale: [1, 1.2, 1] }} transition={{ duration: 2, repeat: Infinity }}
          />
          
          {/* Edges */}
          <path d="M100 100 L160 100" stroke="#00f2ea" strokeWidth="2" />
          <path d="M100 100 L58 142" stroke="#00f2ea" strokeWidth="2" />
          <path d="M100 100 L58 58" stroke="#00f2ea" strokeWidth="2" />

          {/* Child Nodes */}
          <motion.circle cx="160" cy="100" r="5" fill="#fff" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }} />
          <motion.circle cx="58" cy="142" r="5" fill="#fff" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.7 }} />
          <motion.circle cx="58" cy="58" r="5" fill="#fff" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.9 }} />
        </motion.svg>
      </div>

      <motion.h1 
        className="text-7xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white via-accent to-accent-secondary mb-4"
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.2 }}
      >
        FlowRec
      </motion.h1>

      <motion.h2 
        className="text-2xl font-light text-text/70 tracking-widest uppercase"
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.4 }}
      >
        Hierarchical Forecast Reconciliation on Networks
      </motion.h2>

      <motion.p 
        className="mt-12 text-accent font-medium"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1 }}
      >
        Amazon Science • Mayo 2025
      </motion.p>
    </Slide>
  );
}
