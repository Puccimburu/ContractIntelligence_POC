import { useEffect, useRef, useState } from 'react';
import { connectToAgentMessageStream } from '../api/sseService';

export default function useAgentMessages({ conversationId, messageId, isLoading }) {
  const [agentMessages, setAgentMessages] = useState([]);
  const esRef = useRef(null);

  useEffect(() => {
    if (!isLoading || !conversationId || !messageId) {
      setAgentMessages([]);
      return;
    }

    if (esRef.current) { esRef.current.close(); }

    const es = connectToAgentMessageStream(conversationId, messageId, (messages, type) => {
      if (type === 'initial') {
        setAgentMessages(Array.isArray(messages) ? messages : []);
      } else {
        setAgentMessages(prev => {
          const incoming = Array.isArray(messages) ? messages : [messages];
          const existing = new Set(prev.map(m => m.processMessage + m.agentName));
          const newOnes = incoming.filter(m => !existing.has(m.processMessage + m.agentName));
          return [...prev, ...newOnes];
        });
      }
    });

    esRef.current = es;
    return () => { if (esRef.current) { esRef.current.close(); } };
  }, [conversationId, messageId, isLoading]);

  return { agentMessages };
}
