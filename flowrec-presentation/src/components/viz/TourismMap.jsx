import React from 'react';
import { motion } from 'framer-motion';

const ZONES = [
  { id: 'NSW', name: 'New South Wales', path: 'M350,150 L400,150 L400,200 L350,200 Z', cx: 375, cy: 175, val: 98.2 },
  { id: 'VIC', name: 'Victoria', path: 'M320,200 L380,200 L360,240 L300,240 Z', cx: 340, cy: 220, val: 94.5 },
  { id: 'QLD', name: 'Queensland', path: 'M350,50 L420,50 L420,150 L350,150 Z', cx: 385, cy: 100, val: 97.8 },
  { id: 'WA', name: 'Western Australia', path: 'M50,100 L200,100 L200,250 L50,250 Z', cx: 125, cy: 175, val: 96.3 },
  { id: 'SA', name: 'South Australia', path: 'M200,150 L320,150 L300,240 L200,240 Z', cx: 250, cy: 195, val: 95.1 },
  { id: 'NT', name: 'Northern Territory', path: 'M200,50 L350,50 L350,150 L200,150 Z', cx: 275, cy: 100, val: 93.9 },
  { id: 'TAS', name: 'Tasmania', path: 'M340,260 L380,260 L370,290 L350,290 Z', cx: 360, cy: 275, val: 99.1 },
];

// Simplified geometric map for demo purposes (SVG paths are symbolic)
export default function TourismMap() {
  return (
    <svg viewBox="0 0 500 350" className="w-full h-full filter drop-shadow-lg">
       {/* Background */}
       <path d="M50,50 Q250,20 450,50 T 450,250 T 250,300 T 50,250 Z" fill="#1e212b" stroke="#333" strokeWidth="2" opacity="0.5" />
       
       {ZONES.map((zone, i) => (
         <motion.g 
           key={zone.id}
           initial={{ opacity: 0, scale: 0.8 }}
           animate={{ opacity: 1, scale: 1 }}
           transition={{ delay: i * 0.1 }}
           whileHover={{ scale: 1.05 }}
           className="cursor-pointer group"
         >
            <path d={zone.path} fill="rgba(30, 41, 59, 0.8)" stroke="#555" strokeWidth="1" className="group-hover:fill-accent-secondary/20 group-hover:stroke-accent-secondary transition-colors" />
            <circle cx={zone.cx} cy={zone.cy} r="4" fill="#00f2ea" className="animate-pulse" />
            
            {/* Tooltip on Hover */}
            <g className="opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
               <rect x={zone.cx - 40} y={zone.cy - 35} width="80" height="30" rx="4" fill="#000" stroke="#00f2ea" />
               <text x={zone.cx} y={zone.cy - 15} textAnchor="middle" fill="#fff" fontSize="10">{zone.name}</text>
               <text x={zone.cx} y={zone.cy - 5} textAnchor="middle" fill="#00f2ea" fontSize="8">Acc: {zone.val}%</text>
            </g>
         </motion.g>
       ))}
    </svg>
  );
}
