import { useState, useCallback } from 'react';
import { Panel, Group as PanelGroup, Separator as PanelResizeHandle } from 'react-resizable-panels';
import { Plus, Share2 } from 'lucide-react';
import MessageList from './chat/MessageList';
import PromptBar from './input/PromptBar';
import PdfViewer from './pdf/PdfViewer';
import GraphView from './graph/GraphView';
import useConversation from '../hooks/useConversation';
import useFileUpload from '../hooks/useFileUpload';
import usePdfViewer from '../hooks/usePdfViewer';
import api from '../api/api';

export default function Layout() {
  const {
    chatHistory, isLoading, inputValue, setInputValue,
    conversationId, messageId, sendMessage, startNewConversation,
  } = useConversation();

  const {
    attachedFiles, uploadingFiles, isUploading,
    handleFileUpload, removeFile, clearAllFiles,
  } = useFileUpload();

  const {
    isPdfOpen, pdfUrl, fileName: pdfFileName, currentPage, numPages,
    setCurrentPage, setNumPages, openPdf, closePdf, highlightCitationText,
  } = usePdfViewer();

  const [fileUrlCache, setFileUrlCache] = useState({});
  const [isGraphOpen, setIsGraphOpen] = useState(false);

  const handleNewChat = useCallback(() => {
    try {
      Object.values(fileUrlCache).forEach((url) => {
        if (typeof url === 'string') URL.revokeObjectURL(url);
      });
    } catch {
      // ignore
    }
    setFileUrlCache({});
    clearAllFiles();
    closePdf();
    setIsGraphOpen(false);
    startNewConversation();
  }, [fileUrlCache, clearAllFiles, closePdf, startNewConversation]);

  const handleCitationClick = useCallback(async ({ citation }) => {
    if (!citation?.fileId) return;
    const page = parseInt(citation.PageNumber, 10) || 1;
    if (fileUrlCache[citation.fileId]) {
      openPdf(fileUrlCache[citation.fileId], citation.fileName, page);
      return;
    }
    try {
      const res = await api.get(`/ci/file/${citation.fileId}`, { responseType: 'blob' });
      const url = URL.createObjectURL(res.data);
      setFileUrlCache(prev => ({ ...prev, [citation.fileId]: url }));
      openPdf(url, citation.fileName, page);
    } catch {
      // silent
    }
  }, [fileUrlCache, openPdf]);

  const handleSend = useCallback((msg) => {
    sendMessage(msg, attachedFiles, 'ask', clearAllFiles);
  }, [sendMessage, attachedFiles, clearAllFiles]);

  const handleFileUploadWithConversation = useCallback((files) => {
    handleFileUpload(files, conversationId);
  }, [handleFileUpload, conversationId]);

  return (
    <div className="flex flex-col h-screen app-bg">
      <header className="flex items-center justify-between px-5 md:px-8 flex-shrink-0 h-16 md:h-[72px] bg-white border-b border-slate-200 shadow-sm">
        <div className="flex items-center gap-3">
          <div
            className="w-10 h-10 rounded-xl flex items-center justify-center shadow-sm"
            style={{ background: 'linear-gradient(135deg, var(--primary) 0%, #2563eb 100%)' }}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="3" y="8" width="18" height="12" rx="2" />
              <path d="M9 8V6a3 3 0 0 1 6 0v2" />
              <circle cx="9" cy="14" r="1" fill="white" stroke="none" />
              <circle cx="15" cy="14" r="1" fill="white" stroke="none" />
            </svg>
          </div>
          <div>
            <p className="text-[15px] font-semibold leading-none text-slate-900">Contract Intelligence</p>
            <p className="text-xs leading-none mt-1 text-slate-500">Legal Desk</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {conversationId && (
            <button
              onClick={() => { setIsGraphOpen(v => !v); }}
              className={`flex items-center gap-2 text-sm font-semibold px-4 py-2.5 rounded-lg transition-colors border shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400 focus-visible:ring-offset-2 focus-visible:ring-offset-white ${isGraphOpen ? 'bg-indigo-600 text-white border-indigo-600 hover:bg-indigo-700' : 'bg-white hover:bg-slate-50 text-indigo-700 border-slate-200'}`}
              title="Toggle knowledge graph"
            >
              <Share2 size={16} />
              Graph
            </button>
          )}
          <button
            onClick={handleNewChat}
            className="flex items-center gap-2 text-sm font-semibold px-4 py-2.5 rounded-lg transition-colors bg-white hover:bg-slate-50 text-indigo-700 border border-slate-200 shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400 focus-visible:ring-offset-2 focus-visible:ring-offset-white"
            title="Start a new chat"
          >
            <Plus size={16} />
            New chat
          </button>
        </div>
      </header>

      <div className="flex-1 flex justify-center overflow-hidden px-4 py-4 md:px-10 md:py-8">
        <div
          className="h-full w-full max-w-[1440px] rounded-xl overflow-hidden app-shell"
        >
          <PanelGroup direction="horizontal" className="w-full h-full">
          <Panel minSize={35} defaultSize={isPdfOpen ? 55 : 100} className="flex flex-col min-w-0">
            <MessageList
              chatHistory={chatHistory}
              isLoading={isLoading}
              conversationId={conversationId}
              messageId={messageId}
              onFollowUp={handleSend}
              onCitationClick={handleCitationClick}
              onHighlight={highlightCitationText}
            />
            <PromptBar
              value={inputValue}
              onChange={setInputValue}
              onSend={handleSend}
              disabled={isLoading}
              attachedFiles={attachedFiles}
              uploadingFiles={uploadingFiles}
              isUploading={isUploading}
              onFileUpload={handleFileUploadWithConversation}
              onFileRemove={removeFile}
            />
          </Panel>

          {isPdfOpen && (
            <>
              <PanelResizeHandle className="w-px cursor-col-resize bg-slate-200" />
              <Panel minSize={30} defaultSize={45} className="min-w-0">
                <PdfViewer
                  url={pdfUrl}
                  fileName={pdfFileName}
                  currentPage={currentPage}
                  numPages={numPages}
                  setCurrentPage={setCurrentPage}
                  setNumPages={setNumPages}
                  onClose={closePdf}
                />
              </Panel>
            </>
          )}

          {isGraphOpen && (
            <>
              <PanelResizeHandle className="w-px cursor-col-resize bg-slate-200" />
              <Panel minSize={30} defaultSize={isPdfOpen ? 35 : 45} className="min-w-0">
                <GraphView
                  conversationId={conversationId}
                  onClose={() => setIsGraphOpen(false)}
                />
              </Panel>
            </>
          )}
        </PanelGroup>
        </div>
      </div>
    </div>
  );
}
