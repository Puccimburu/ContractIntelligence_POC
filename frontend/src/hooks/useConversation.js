import { useState, useCallback, useRef } from 'react';
import api from '../api/api';
import { connectToMessageStream } from '../api/sseService';

export default function useConversation() {
  const [chatHistory, setChatHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [inputValue, setInputValue] = useState('');
  const [conversationId, setConversationId] = useState(null);
  const [messageId, setMessageId] = useState(null);
  const [aiMessageId, setAiMessageId] = useState(null);

  // Keep a ref to the active SSE connection so we can close it on new send
  const sseRef = useRef(null);

  const sendMessage = useCallback(async (userMessage, attachedFiles = [], _featureId, clearFiles) => {
    if (!userMessage?.trim() && attachedFiles.length === 0) return;
    if (isLoading) return;

    const activeConversationId =
      conversationId ||
      attachedFiles.find(f => f.conversationId)?.conversationId ||
      null;

    if (!activeConversationId) {
      alert('Please upload at least one contract file before asking a question.');
      return;
    }

    if (!conversationId) setConversationId(activeConversationId);

    // Close any previous SSE stream
    if (sseRef.current) {
      sseRef.current.close();
      sseRef.current = null;
    }

    setIsLoading(true);
    setInputValue('');
    if (clearFiles) clearFiles();

    // Append user message optimistically
    const userMsg = {
      id: Date.now(),
      role: 'user',
      content: [{ type: 'text', content: userMessage }],
      getText: () => userMessage,
      attachments: attachedFiles,
    };
    setChatHistory(prev => [...prev, userMsg]);

    // AI placeholder while streaming
    const tempAiId = `temp-${Date.now()}`;
    setChatHistory(prev => [
      ...prev,
      { id: tempAiId, role: 'ai', content: [], citations: [], followUpQuestions: [] },
    ]);

    try {
      // Fire-and-forget: creates messages in MongoDB and kicks off RAG
      const { data } = await api.post('/desk/conversation', {
        conversationId: activeConversationId,
        query: userMessage,
      });

      const { messageId: msgId, aiMessageId: aiMsgId } = data;
      setMessageId(msgId);
      setAiMessageId(aiMsgId);

      // Open SSE stream for the AI response
      const sse = connectToMessageStream(aiMsgId, setIsLoading, (payload, eventType) => {
        if (eventType === 'initial' && payload.status === 'Processed') {
          // Already done (e.g. fast response or reconnect)
          setChatHistory(prev =>
            prev.map(m =>
              m.id === tempAiId
                ? {
                    ...m,
                    content: payload.content || [],
                    citations: payload.citations || [],
                    followUpQuestions: payload.followUpQuestions || [],
                  }
                : m,
            ),
          );
          setIsLoading(false);
        } else if (eventType === 'update' && payload.isComplete) {
          setChatHistory(prev =>
            prev.map(m =>
              m.id === tempAiId
                ? {
                    ...m,
                    content: payload.content || [],
                    citations: payload.citations || [],
                    followUpQuestions: payload.followUpQuestions || [],
                  }
                : m,
            ),
          );
          setIsLoading(false);
        }
      });

      sseRef.current = sse;
    } catch (err) {
      console.error('Desk conversation failed:', err);
      setChatHistory(prev =>
        prev.map(m =>
          m.id === tempAiId
            ? {
                ...m,
                content: 'An error occurred while processing your query. Please try again.',
                citations: [],
                followUpQuestions: [],
              }
            : m,
        ),
      );
      setIsLoading(false);
    }
  }, [conversationId, isLoading]);

  const startNewConversation = useCallback(() => {
    if (sseRef.current) {
      sseRef.current.close();
      sseRef.current = null;
    }
    setChatHistory([]);
    setConversationId(null);
    setMessageId(null);
    setAiMessageId(null);
    setIsLoading(false);
    setInputValue('');
  }, []);

  return {
    chatHistory,
    isLoading,
    inputValue,
    setInputValue,
    conversationId,
    messageId,
    aiMessageId,
    sendMessage,
    startNewConversation,
  };
}
