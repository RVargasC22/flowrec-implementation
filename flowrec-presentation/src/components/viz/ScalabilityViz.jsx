import React from 'react';
import { motion } from 'framer-motion';

export default function ScalabilityViz() {
  // Generate random particles
  const particles = Array.from({ length: 150 }).map((_, i) => ({
    id: i,
    x: Math.random() * 100,
    y: Math.random() * 100,
    size: Math.random() * 3 + 1,
    duration: Math.random() * 20 + 10,
    delay: Math.random() * 5
  }));

  return (
    <div className="w-full h-full relative overflow-hidden bg-black/40 rounded-lg">
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-10">
        <div className="text-center">
           <motion.div 
             className="text-6xl font-black text-white/5"
             animate={{ scale: [1, 1.1, 1], opacity: [0.05, 0.1, 0.05] }}
             transition={{ duration: 4, repeat: Infinity }}
           >
             40M+
           </motion.div>
        </div>
      </div>

      {particles.map((p) => (
        <motion.div
          key={p.id}
          className="absolute rounded-full bg-accent"
          style={{ 
            left: `${p.x}%`, 
            top: `${p.y}%`, 
            width: p.size, 
            height: p.size,
            opacity: Math.random() * 0.5 + 0.1
          }}
          animate={{
            y: [0, -100],
            opacity: [0, 0.8, 0]
          }}
          transition={{
            duration: p.duration,
            repeat: Infinity,
            delay: p.delay,
            ease: "linear"
          }}
        />
      ))}
      
      {/* Grid overlay */}
      <div className="absolute inset-0 bg-[url('/grid.svg')] opacity-10" />
    </div>
  );
}
