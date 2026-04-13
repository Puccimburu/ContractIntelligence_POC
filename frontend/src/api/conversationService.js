import api from './api';

/**
 * Send a query to the Python RAG pipeline.
 * Returns { answer, citations }
 */
export const sendQuery = async (conversationId, query) => {
  const response = await api.post('/ci/query', { conversationId, query });
  return response.data;
};

/**
 * Upload a file directly to the Python backend.
 * Returns { fileId, fileName, conversationId, status }
 */
export const uploadFile = async (file, conversationId = null) => {
  const form = new FormData();
  form.append('file', file);
  if (conversationId) form.append('conversationId', conversationId);

  const response = await api.post('/ci/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

/**
 * Poll processing status for a file.
 * Returns { fileId, processing_status, sections_embedded }
 */
export const getFileStatus = async (fileId) => {
  const response = await api.get(`/ci/status/${fileId}`);
  return response.data;
};

/**
 * List all files in a conversation (for state restore on refresh).
 */
export const getConversationFiles = async (conversationId) => {
  const response = await api.get(`/ci/conversations/${conversationId}/files`);
  return response.data;
};
