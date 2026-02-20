import { motion } from 'framer-motion';
import clsx from 'clsx';

const slideVariants = {
  hidden: { opacity: 0, scale: 0.95 },
  visible: { opacity: 1, scale: 1, transition: { duration: 0.5 } },
  exit: { opacity: 0, scale: 1.05, transition: { duration: 0.3 } }
};

export default function Slide({ title, children, className }) {
  return (
    <motion.div 
      className={clsx(
        "w-full h-full flex flex-col p-12 relative overflow-hidden bg-gradient-to-br from-background via-[#161b28] to-background shadow-[0_0_80px_rgba(0,242,234,0.1)] rounded-lg border border-white/5",
        className
      )}
      initial="hidden"
      animate="visible"
      exit="exit"
      variants={slideVariants}
    >
      {/* Header */}
      {title && (
        <header className="mb-8 border-b-2 border-accent/30 pb-4">
          <h2 className="text-4xl font-bold text-accent drop-shadow-[0_0_10px_rgba(0,242,234,0.3)]">
            {title}
          </h2>
        </header>
      )}

      {/* Content */}
      <div className="flex-1 w-full h-full relative z-10">
        {children}
      </div>

      {/* Background Decor */}
      <div className="absolute inset-0 pointer-events-none opacity-10 z-0">
          <svg width="100%" height="100%">
            <pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">
              <path d="M 50 0 L 0 0 0 50" fill="none" stroke="white" strokeWidth="0.5"/>
            </pattern>
            <rect width="100%" height="100%" fill="url(#grid)" />
          </svg>
      </div>
    </motion.div>
  );
}
