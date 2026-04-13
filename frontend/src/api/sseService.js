const BASE = import.meta.env.VITE_API_BASE_URL;

/**
 * Stream AI response for a given aiMessageId.
 * Fires onUpdate(data) on 'ai-response-update' events.
 * Calls setIsSending(false) when done.
 */
export const connectToMessageStream = (aiMessageId, setIsSending, onUpdate) => {
  if (!aiMessageId) return;

  const url = `${BASE}/desk/messages/stream/${aiMessageId}`;
  const es = new EventSource(url, { withCredentials: true });

  const timeout = setTimeout(() => {
    es.close();
    setIsSending(false);
  }, 780000); // 13 minutes — covers 10-min processing wait + RAG + LLM

  es.onopen = () => clearTimeout(timeout);

  es.addEventListener('initial-state', (e) => {
    try { onUpdate(JSON.parse(e.data), 'initial'); } catch (_) {}
  });

  es.addEventListener('ai-response-update', (e) => {
    try {
      const data = JSON.parse(e.data);
      onUpdate(data, 'update');
      if (data.isComplete) { es.close(); setIsSending(false); }
    } catch (_) {}
  });

  es.onerror = () => { es.close(); setIsSending(false); };

  return { close: () => { clearTimeout(timeout); es.close(); } };
};

/**
 * Stream agent progress messages for a conversationId + messageId pair.
 */
export const connectToAgentMessageStream = (conversationId, messageId, onUpdate) => {
  if (!conversationId || !messageId) return;

  const url = `${BASE}/messages/stream/${conversationId}/${messageId}`;
  const es = new EventSource(url, { withCredentials: true });

  const timeout = setTimeout(() => es.close(), 20000);
  es.onopen = () => clearTimeout(timeout);

  es.addEventListener('initial-state', (e) => {
    try { onUpdate(JSON.parse(e.data), 'initial'); } catch (_) {}
  });

  es.addEventListener('agent-update', (e) => {
    try { onUpdate(JSON.parse(e.data), 'update'); } catch (_) {}
  });

  es.onerror = () => es.close();

  return { close: () => { clearTimeout(timeout); es.close(); } };
};
