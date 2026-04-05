'use client';

export default function TypingIndicator() {
  return (
    <div className="flex items-center gap-1 py-2">
      <style>{`
        @keyframes pulse-dots {
          0%, 60%, 100% {
            opacity: 0.3;
          }
          30% {
            opacity: 1;
          }
        }
        .dot-1 {
          animation: pulse-dots 1.2s infinite;
        }
        .dot-2 {
          animation: pulse-dots 1.2s infinite 0.2s;
        }
        .dot-3 {
          animation: pulse-dots 1.2s infinite 0.4s;
        }
      `}</style>
      <span className="dot-1 text-base leading-none">.</span>
      <span className="dot-2 text-base leading-none">.</span>
      <span className="dot-3 text-base leading-none">.</span>
    </div>
  );
}
