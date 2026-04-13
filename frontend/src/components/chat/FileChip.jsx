import { FileText, X } from 'lucide-react';

export function FileChip({ file, onRemove, onClick }) {
  return (
    <div
      className="inline-flex items-center gap-1.5 px-3 py-2 rounded-lg cursor-pointer group transition max-w-[240px] bg-white/70 border border-slate-200 hover:border-slate-300 hover:bg-white shadow-sm"
      onClick={() => onClick && onClick(file)}
    >
      <FileText size={11} className="text-indigo-600 flex-shrink-0" />
      <span className="text-xs font-medium truncate flex-1 text-slate-700">
        {file.fileName || file.name}
      </span>
      {onRemove && (
        <button
          onClick={e => { e.stopPropagation(); onRemove(file.fileId); }}
          className="flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity text-slate-400 hover:text-rose-600"
        >
          <X size={10} />
        </button>
      )}
    </div>
  );
}

export function FileChipBar({ files = [], onRemove, onOpen }) {
  if (!files.length) return null;
  return (
    <div className="flex gap-2 flex-wrap px-3 pb-2">
      {files.map(f => (
        <FileChip key={f.fileId} file={f} onRemove={onRemove} onClick={onOpen} />
      ))}
    </div>
  );
}
