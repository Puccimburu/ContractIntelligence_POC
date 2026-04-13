import { useState, useCallback, useRef } from 'react';
import { uploadFile, getFileStatus } from '../api/conversationService';

const MAX_FILES = 5;
const POLL_INTERVAL_MS = 5000;
const POLL_MAX_ATTEMPTS = 120; // 10 minutes max — covers scanned PDFs with OCR

export default function useFileUpload() {
  const [attachedFiles, setAttachedFiles] = useState([]);
  const [uploadingFiles, setUploadingFiles] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  // Track the conversation established by the first upload so that subsequent
  // file picker sessions (or multi-file batches) all land in the same conversation.
  const activeConvIdRef = useRef(null);

  /**
   * Poll until processing_status === "ready" or "failed".
   * Returns the final status string.
   */
  const _pollUntilReady = useCallback(async (fileId) => {
    for (let i = 0; i < POLL_MAX_ATTEMPTS; i++) {
      await new Promise(r => setTimeout(r, POLL_INTERVAL_MS));
      try {
        const status = await getFileStatus(fileId);
        if (status.processing_status === 'ready') return 'ready';
        if (status.processing_status === 'failed') return 'failed';
      } catch (_) {
        // transient network error — keep polling
      }
    }
    return 'timeout';
  }, []);

  const uploadSingle = useCallback(async (file, currentConversationId) => {
    const tempId = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    setUploadingFiles(prev => [...prev, { fileId: tempId, fileName: file.name }]);

    try {
      // Upload to Python backend — passes conversationId so all files land in same scope
      const data = await uploadFile(file, currentConversationId);
      const uploaded = {
        fileId: data.fileId,
        fileName: data.fileName,
        conversationId: data.conversationId,
        status: 'processing',
      };

      // Optimistically add to attached list while processing runs in background
      setAttachedFiles(prev => [...prev, uploaded]);

      // Poll in background — update status chip when ready
      _pollUntilReady(data.fileId).then(finalStatus => {
        setAttachedFiles(prev =>
          prev.map(f => f.fileId === data.fileId ? { ...f, status: finalStatus } : f)
        );
      });

      return uploaded;
    } catch (err) {
      console.error('Upload failed:', err);
      return null;
    } finally {
      setUploadingFiles(prev => prev.filter(f => f.fileId !== tempId));
    }
  }, [_pollUntilReady]);

  const handleFileUpload = useCallback(async (files, currentConversationId) => {
    const arr = Array.from(files);
    // Read current length via setter to avoid closing over stale state
    let tooMany = false;
    setAttachedFiles(prev => {
      if (prev.length + arr.length > MAX_FILES) tooMany = true;
      return prev;
    });
    if (tooMany) {
      alert(`Maximum ${MAX_FILES} files allowed`);
      return;
    }
    setIsUploading(true);
    // Resolve the conversation to use: prefer the caller-supplied id, then the
    // one we established on a previous upload in this session, then null (backend
    // will create a new one for the first file and we capture it below).
    let activeConvId = currentConversationId || activeConvIdRef.current || null;
    for (const file of arr) {
      const uploaded = await uploadSingle(file, activeConvId);
      if (uploaded && !activeConvId) {
        // First file created a new conversation — reuse it for all remaining files.
        activeConvId = uploaded.conversationId;
        activeConvIdRef.current = activeConvId;
      }
    }
    setIsUploading(false);
  }, [uploadSingle]);

  const removeFile = useCallback((fileId) => {
    setAttachedFiles(prev => prev.filter(f => f.fileId !== fileId));
  }, []);

  const clearAllFiles = useCallback(() => {
    setAttachedFiles([]);
    setUploadingFiles([]);
    activeConvIdRef.current = null;  // Reset so a new chat starts a fresh conversation
  }, []);

  return {
    attachedFiles,
    uploadingFiles,
    isUploading,
    handleFileUpload,
    removeFile,
    clearAllFiles,
  };
}
