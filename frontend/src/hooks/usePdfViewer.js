import { useState, useCallback } from 'react';

export default function usePdfViewer() {
  const [isPdfOpen, setIsPdfOpen] = useState(false);
  const [pdfUrl, setPdfUrl] = useState('');
  const [fileName, setFileName] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [numPages, setNumPages] = useState(null);

  const openPdf = useCallback((url, name, page = 1) => {
    setPdfUrl(url);
    setFileName(name || '');
    setCurrentPage(page);
    setNumPages(null);
    setIsPdfOpen(true);
  }, []);

  const closePdf = useCallback(() => {
    setIsPdfOpen(false);
    setPdfUrl('');
  }, []);

  // Called when user clicks a citation badge
  const onCitationClick = useCallback(({ citation }) => {
    if (!citation) return;
    const page = parseInt(citation.PageNumber, 10) || 1;
    if (citation._url) {
      openPdf(citation._url, citation.fileName, page);
    } else if (isPdfOpen) {
      setCurrentPage(page);
    }
  }, [isPdfOpen, openPdf]);

  // Called when user clicks citation text (highlight)
  const highlightCitationText = useCallback(({ pageNumber }) => {
    const page = parseInt(pageNumber, 10) || 1;
    if (isPdfOpen) setCurrentPage(page);
  }, [isPdfOpen]);

  return {
    isPdfOpen,
    pdfUrl,
    fileName,
    currentPage,
    numPages,
    setCurrentPage,
    setNumPages,
    openPdf,
    closePdf,
    onCitationClick,
    highlightCitationText,
  };
}
