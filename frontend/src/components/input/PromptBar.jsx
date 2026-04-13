import { useRef, useEffect } from 'react';
import { Paperclip, X, Loader2, AlertCircle, CheckCircle2 } from 'lucide-react';

export default function PromptBar({
  value,
  onChange,
  onSend,
  disabled,
  attachedFiles = [],
  uploadingFiles = [],
  onFileUpload,
  onFileRemove,
}) {
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 140) + 'px';
  }, [value]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!disabled && (value.trim() || attachedFiles.length > 0)) onSend(value);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files?.length) {
      onFileUpload(e.target.files);
      e.target.value = '';
    }
  };

  const canSend = !disabled && (value.trim() || attachedFiles.length > 0);

  return (
    <div className="flex-shrink-0 px-6 md:px-10 pb-5 pt-4 bg-white border-t border-slate-200">
      <div className="rounded-xl overflow-hidden bg-white border border-slate-200 shadow-sm">
        {(attachedFiles.length > 0 || uploadingFiles.length > 0) && (
          <div className="px-4 pt-3 pb-2 flex flex-wrap gap-2 border-b border-slate-100">
            {attachedFiles.map((f) => (
              <span
                key={f.fileId}
                className={[
                  'inline-flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-lg border',
                  f.status === 'failed'
                    ? 'bg-rose-50 border-rose-200 text-rose-700'
                    : f.status === 'ready'
                      ? 'bg-emerald-50 border-emerald-200 text-emerald-700'
                      : 'bg-indigo-50 border-indigo-200 text-indigo-700',
                ].join(' ')}
              >
                {f.status === 'processing' && <Loader2 size={10} className="animate-spin" />}
                {f.status === 'ready' && <CheckCircle2 size={10} />}
                {f.status === 'failed' && <AlertCircle size={10} />}
                <span className="truncate max-w-[160px]">{f.fileName}</span>
                <button
                  onClick={() => onFileRemove(f.fileId)}
                  className="text-inherit/70 hover:text-rose-700 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-300 focus-visible:ring-offset-2 focus-visible:ring-offset-white rounded-md"
                >
                  <X size={10} />
                </button>
              </span>
            ))}
            {uploadingFiles.map((f) => (
              <span
                key={f.fileId}
                className="inline-flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg bg-slate-50 text-slate-500 border border-slate-200"
              >
                <Loader2 size={10} className="animate-spin" />
                <span className="truncate max-w-[160px]">{f.fileName}</span>
              </span>
            ))}
          </div>
        )}

        <div
          className="flex items-end gap-2 px-4 py-3 mx-2 mb-2 rounded-xl bg-slate-50 border border-slate-200 focus-within:border-indigo-300 focus-within:ring-2 focus-within:ring-indigo-100 transition"
        >
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={disabled}
            className="flex-shrink-0 w-10 h-10 rounded-lg flex items-center justify-center transition-colors disabled:opacity-40 text-slate-400 hover:text-indigo-600 hover:bg-indigo-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-300 focus-visible:ring-offset-2 focus-visible:ring-offset-white"
          >
            <Paperclip size={18} />
          </button>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf,.docx,.doc,.txt"
            className="hidden"
            onChange={handleFileChange}
          />

          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={disabled}
            rows={1}
            placeholder="Ask about this contract..."
            className="flex-1 resize-none bg-transparent text-[15px] focus:outline-none disabled:opacity-50 leading-relaxed py-2.5 text-slate-900 placeholder:text-slate-400"
            style={{ minHeight: 38, maxHeight: 140 }}
          />

          <button
            type="button"
            onClick={() => onSend(value)}
            disabled={!canSend}
            className={[
              'flex-shrink-0 w-10 h-10 rounded-lg flex items-center justify-center transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-300 focus-visible:ring-offset-2 focus-visible:ring-offset-white',
              canSend ? 'shadow-sm' : '',
            ].join(' ')}
            style={canSend ? { background: 'var(--primary)' } : { background: '#e2e8f0' }}
          >
            {disabled ? (
              <Loader2 size={15} className="animate-spin text-indigo-200" />
            ) : (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={canSend ? 'white' : '#64748b'} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="22" y1="2" x2="11" y2="13" />
                <polygon points="22 2 15 22 11 13 2 9 22 2" />
              </svg>
            )}
          </button>
        </div>
      </div>

      <p className="text-center text-xs mt-2 text-slate-400">
        Enter to send - Shift+Enter for new line
      </p>
    </div>
  );
}
