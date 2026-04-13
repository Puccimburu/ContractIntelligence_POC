import { useState, useCallback } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import { ChevronLeft, ChevronRight, X, ZoomIn, ZoomOut, FileText } from 'lucide-react';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';

pdfjs.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

function ToolbarBtn({ onClick, disabled, title, children, danger }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={[
        'w-8 h-8 rounded-xl flex items-center justify-center transition disabled:opacity-30',
        danger
          ? 'text-slate-500 hover:text-rose-600 hover:bg-rose-50'
          : 'text-slate-500 hover:text-indigo-700 hover:bg-indigo-50',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-300 focus-visible:ring-offset-2 focus-visible:ring-offset-white',
      ].join(' ')}
    >
      {children}
    </button>
  );
}

export default function PdfViewer({
  url,
  fileName,
  currentPage,
  numPages,
  setCurrentPage,
  setNumPages,
  onClose,
}) {
  const [scale, setScale] = useState(1.0);
  const [error, setError] = useState(null);

  const onDocumentLoadSuccess = useCallback(({ numPages: n }) => {
    setNumPages(n);
    setError(null);
  }, [setNumPages]);

  return (
    <div className="flex flex-col h-full bg-[var(--panel-muted)]">
      <div className="flex items-center justify-between px-3 flex-shrink-0 h-12 bg-white/80 backdrop-blur border-b border-slate-200">
        <div className="flex items-center gap-2 min-w-0">
          <FileText size={13} className="text-indigo-600 flex-shrink-0" />
          <span className="text-xs font-semibold truncate max-w-[220px] text-slate-700">
            {fileName}
          </span>
        </div>

        <div className="flex items-center gap-0.5">
          <ToolbarBtn onClick={() => setScale(s => Math.max(0.5, s - 0.15))} title="Zoom out">
            <ZoomOut size={14} />
          </ToolbarBtn>
          <span className="text-xs w-11 text-center tabular-nums font-medium text-slate-500">
            {Math.round(scale * 100)}%
          </span>
          <ToolbarBtn onClick={() => setScale(s => Math.min(3, s + 0.15))} title="Zoom in">
            <ZoomIn size={14} />
          </ToolbarBtn>

          <div className="w-px h-4 bg-slate-200 mx-1.5" />

          <ToolbarBtn
            onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
            disabled={currentPage <= 1}
            title="Previous page"
          >
            <ChevronLeft size={14} />
          </ToolbarBtn>
          <span className="text-xs tabular-nums font-medium text-slate-600" style={{ minWidth: 52, textAlign: 'center' }}>
            {currentPage} / {numPages || '—'}
          </span>
          <ToolbarBtn
            onClick={() => setCurrentPage(p => Math.min(numPages || p, p + 1))}
            disabled={Boolean(numPages) && currentPage >= numPages}
            title="Next page"
          >
            <ChevronRight size={14} />
          </ToolbarBtn>

          <div className="w-px h-4 bg-slate-200 mx-1.5" />

          <ToolbarBtn onClick={onClose} danger title="Close">
            <X size={14} />
          </ToolbarBtn>
        </div>
      </div>

      <div className="flex-1 overflow-auto thin-scroll flex justify-center py-8 px-4 bg-slate-100">
        {error ? (
          <div className="text-sm p-8 text-rose-600">{error}</div>
        ) : (
          <Document
            file={url}
            onLoadSuccess={onDocumentLoadSuccess}
            onLoadError={e => setError('Failed to load PDF: ' + e.message)}
            loading={
              <div className="flex items-center justify-center p-8 text-sm text-slate-500">
                Loading…
              </div>
            }
          >
            <Page
              pageNumber={currentPage}
              scale={scale}
              renderAnnotationLayer
              renderTextLayer
              className="rounded-md shadow-lg ring-1 ring-black/5"
            />
          </Document>
        )}
      </div>
    </div>
  );
}
