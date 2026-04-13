export default function AgentProgress({ steps = [] }) {
  if (!steps.length) {
    return (
      <div className="flex items-center gap-3 py-1">
        <div className="flex gap-1">
          {[0, 1, 2].map(i => (
            <span
              key={i}
              className="bounce-dot block w-2 h-2 rounded-full"
              style={{ background: '#6366f1', animationDelay: `${i * 0.16}s` }}
            />
          ))}
        </div>
        <span className="text-sm text-slate-500">Analyzing documents…</span>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {steps.map((step, i) => {
        const isActive = i === steps.length - 1;
        return (
          <div key={i} className="flex items-center gap-3">
            <div className="flex-shrink-0 w-5 flex justify-center">
              {isActive ? (
                <div className="flex gap-0.5">
                  {[0, 1, 2].map(j => (
                    <span
                      key={j}
                      className="bounce-dot block w-1.5 h-1.5 rounded-full"
                      style={{ background: '#4f46e5', animationDelay: `${j * 0.16}s` }}
                    />
                  ))}
                </div>
              ) : (
                <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                  <circle cx="7" cy="7" r="6" stroke="#cbd5e1" strokeWidth="1.5" />
                  <path d="M4.5 7L6.5 9L9.5 5.5" stroke="#cbd5e1" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              )}
            </div>
            <span
              className="text-sm leading-snug"
              style={{ color: isActive ? '#0f172a' : '#94a3b8' }}
            >
              {step.processMessage}
            </span>
          </div>
        );
      })}
    </div>
  );
}
