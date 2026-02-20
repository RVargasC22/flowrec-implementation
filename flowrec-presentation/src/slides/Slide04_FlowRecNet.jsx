import React from 'react';
import { motion } from 'framer-motion';
import { Network, Zap, Layers } from 'lucide-react';
import Slide from '../components/Slide';

export default function Slide04_FlowRecNet() {
  return (
    <Slide title="Solución: Grafo Generalizado (DAG)">
      <div className="flex gap-12 h-full">
        {/* Viz Column - DAG Network */}
        <div className="flex-1 bg-black/20 rounded-xl relative overflow-hidden flex items-center justify-center">
            <svg viewBox="0 0 500 300" className="w-full">
                <defs>
                    <marker id="arrow" markerWidth="10" markerHeight="10" refX="20" refY="3" orient="auto" markerUnits="strokeWidth">
                        <path d="M0,0 L0,6 L9,3 z" fill="#555" />
                    </marker>
                </defs>
                
                {/* Edges */}
                <motion.path d="M50 150 L150 50" stroke="#555" strokeWidth="2" markerEnd="url(#arrow)" 
                   initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1 }} />
                <motion.path d="M50 150 L150 250" stroke="#555" strokeWidth="2" markerEnd="url(#arrow)" 
                   initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1, delay: 0.2 }} />
                
                <motion.path d="M150 50 L300 100" stroke="#555" strokeWidth="2" markerEnd="url(#arrow)" 
                   initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1, delay: 0.4 }} />
                <motion.path d="M150 250 L300 100" stroke="#00f2ea" strokeWidth="2" strokeDasharray="5,5" markerEnd="url(#arrow)"
                   initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 1, delay: 1 }} />
                 <text x="210" y="200" fill="#00f2ea" fontSize="12">Cross-Learning</text>

                {/* Nodes */}
                <circle cx="50" cy="150" r="20" fill="#333" stroke="#fff"/>
                <circle cx="150" cy="50" r="15" fill="#333" stroke="#fff"/>
                <circle cx="150" cy="250" r="15" fill="#333" stroke="#fff"/>
                <circle cx="300" cy="100" r="15" fill="#333" stroke="#fff"/>
            </svg>
            
            <div className="absolute inset-0 bg-gradient-to-t from-background to-transparent pointer-events-none" />
        </div>

        {/* Text Column */}
        <div className="flex-1 space-y-8 flex flex-col justify-center">
            <motion.div 
              className="p-6 bg-surface border border-accent/20 rounded-lg"
              initial={{ x: 20, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ delay: 0.5 }}
            >
                <div className="flex items-center gap-4 mb-2">
                    <Network className="text-accent" />
                    <h3 className="text-xl font-bold">Topología Arbitraria</h3>
                </div>
                <p className="text-text/70">Soporta ciclos, múltiples padres y conexiones cruzadas (DAG).</p>
            </motion.div>

            <motion.div 
              className="p-6 bg-surface border border-accent/20 rounded-lg"
              initial={{ x: 20, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ delay: 0.7 }}
            >
                <div className="flex items-center gap-4 mb-2">
                    <Layers className="text-accent-tertiary" />
                    <h3 className="text-xl font-bold">Jerarquía Dinámica</h3>
                </div>
                <p className="text-text/70">Nodos pueden agregarse/eliminarse sin reentrenar todo el modelo.</p>
            </motion.div>

            <motion.div 
              className="p-6 bg-surface border border-accent/20 rounded-lg"
              initial={{ x: 20, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ delay: 0.9 }}
            >
                <div className="flex items-center gap-4 mb-2">
                    <Zap className="text-accent-secondary" />
                    <h3 className="text-xl font-bold">Eficiencia O(n log n)</h3>
                </div>
                <p className="text-text/70">Explota la dispersión (sparsity) de la matriz S.</p>
            </motion.div>
        </div>
      </div>
    </Slide>
  );
}
