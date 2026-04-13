import { FileText, ExternalLink } from 'lucide-react';

export default function CitationList({ citations = [], onCitationClick }) {
  if (!citations.length) return null;

  return (
    <div className="mt-5 pt-4 border-t border-slate-200">
      <p className="text-xs font-semibold uppercase tracking-widest mb-3 text-slate-400">
        Sources
      </p>
      <div className="flex flex-wrap gap-2">
        {citations.map((cit, i) => (
          <button
            key={i}
            onClick={() => onCitationClick && onCitationClick({ citation: cit })}
            className="group inline-flex items-center gap-2 text-xs font-medium px-3 py-2 rounded-lg transition bg-slate-50 border border-slate-200 text-slate-700 hover:bg-indigo-50 hover:border-indigo-200 hover:text-indigo-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-300 focus-visible:ring-offset-2 focus-visible:ring-offset-white"
            title={`${cit.fileName} - Page ${cit.PageNumber}`}
          >
            <FileText size={12} className="flex-shrink-0 opacity-60" />
            <span className="max-w-[150px] truncate">{cit.fileName}</span>
            {cit.PageNumber && (
              <span
                className="px-1.5 py-0.5 rounded-md font-semibold bg-indigo-100 text-indigo-700"
                style={{ fontSize: '10px' }}
              >
                p.{cit.PageNumber}
              </span>
            )}
            <ExternalLink size={10} className="opacity-40 flex-shrink-0" />
          </button>
        ))}
      </div>
    </div>
  );
}
