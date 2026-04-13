import { useEffect, useRef } from 'react';
import BotResponse from './BotResponse';
import AgentProgress from './AgentProgress';
import FollowUpQuestions from './FollowUpQuestions';
import { FileChip } from './FileChip';
import useAgentMessages from '../../hooks/useAgentMessages';
import { ListChecks, AlertTriangle, DoorOpen, Scale } from 'lucide-react';

const STARTER_PROMPTS = [
  { Icon: ListChecks, title: 'Key obligations', text: 'Summarise key obligations and rights' },
  { Icon: AlertTriangle, title: 'SLA penalties', text: 'What are the SLA penalties?' },
  { Icon: DoorOpen, title: 'Termination', text: 'Find termination and exit clauses' },
  { Icon: Scale, title: 'Liability caps', text: 'Identify liability caps and exclusions' },
];

function BotAvatar() {
  return (
    <div
      className="flex-shrink-0 w-10 h-10 rounded-xl flex items-center justify-center shadow-sm ring-1 ring-black/5"
      style={{ background: 'linear-gradient(135deg, var(--primary) 0%, #2563eb 100%)' }}
    >
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="8" width="18" height="12" rx="2" />
        <path d="M9 8V6a3 3 0 0 1 6 0v2" />
        <circle cx="9" cy="14" r="1" fill="white" stroke="none" />
        <circle cx="15" cy="14" r="1" fill="white" stroke="none" />
      </svg>
    </div>
  );
}

export default function MessageList({
  chatHistory,
  isLoading,
  conversationId,
  messageId,
  onFollowUp,
  onCitationClick,
  onHighlight,
}) {
  const bottomRef = useRef(null);
  const { agentMessages } = useAgentMessages({ conversationId, messageId, isLoading });

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory]);

  if (!chatHistory.length && !isLoading) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center px-6 py-16 bg-white">
        <div className="w-full max-w-3xl">
          <div className="flex flex-col items-center text-center">
            <div
              className="w-16 h-16 rounded-2xl flex items-center justify-center mb-4 shadow-md ring-1 ring-black/5"
              style={{ background: 'linear-gradient(135deg, var(--primary) 0%, #2563eb 100%)' }}
            >
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <rect x="3" y="8" width="18" height="12" rx="2" />
                <path d="M9 8V6a3 3 0 0 1 6 0v2" />
                <circle cx="9" cy="14" r="1" fill="white" stroke="none" />
                <circle cx="15" cy="14" r="1" fill="white" stroke="none" />
              </svg>
            </div>
            <h2 className="text-xl md:text-2xl font-semibold text-slate-900 tracking-tight">
              Ask anything about your contracts
            </h2>
            <p className="text-sm md:text-[15px] mt-2 max-w-xl text-slate-600 leading-relaxed">
              Upload a contract, then ask for obligations, risks, SLA penalties, termination, and citations back to the source.
            </p>
          </div>

          <div className="mt-9 grid grid-cols-1 sm:grid-cols-2 gap-3">
            {STARTER_PROMPTS.map(({ Icon, title, text }, i) => (
              <button
                key={i}
                onClick={() => onFollowUp && onFollowUp(text)}
                className="group flex items-start gap-3 text-left p-5 rounded-xl bg-slate-50 border border-slate-200 hover:bg-white hover:border-slate-300 transition shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400 focus-visible:ring-offset-2 focus-visible:ring-offset-white"
              >
                <div className="w-9 h-9 rounded-xl flex items-center justify-center bg-indigo-50 text-indigo-700 border border-indigo-100">
                  <Icon size={18} />
                </div>
                <div className="min-w-0">
                  <div className="text-sm font-semibold text-slate-900">{title}</div>
                  <div className="text-sm text-slate-600 mt-0.5">{text}</div>
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>
    );
  }

  const pairs = [];
  for (let i = 0; i < chatHistory.length; i++) {
    if (chatHistory[i].role === 'user') {
      pairs.push({ user: chatHistory[i], ai: chatHistory[i + 1] || null, idx: i });
    }
  }

  return (
    <div className="flex-1 overflow-y-auto thin-scroll bg-slate-50">
      <div className="px-6 md:px-10 py-8 space-y-7 w-full max-w-4xl mx-auto">
        {pairs.map(({ user, ai, idx }) => {
          const isLast = idx === chatHistory.length - 1 || idx === chatHistory.length - 2;
          const showLoading = isLoading && isLast && (!ai || !ai.content?.length);
          const userText = user.content?.[0]?.content || user.getText?.() || '';

          return (
            <div key={user.id || idx} className="space-y-3">
              <div className="flex justify-end">
                <div className="max-w-[85%] space-y-2">
                  <div
                    className="px-5 py-4 rounded-xl text-sm leading-relaxed text-white shadow-sm"
                    style={{ background: 'var(--primary)' }}
                  >
                    {userText}
                  </div>
                  {user.attachments?.length > 0 && (
                    <div className="flex gap-2 flex-wrap justify-end">
                      {user.attachments.map(att => (
                        <FileChip
                          key={att.fileId}
                          file={att}
                          onClick={f => onCitationClick?.({ citation: { fileId: f.fileId, fileName: f.fileName } })}
                        />
                      ))}
                    </div>
                  )}
                </div>
              </div>

              <div className="flex gap-2.5 items-start">
                <BotAvatar />
                <div
                  className="flex-1 min-w-0 px-5 py-4 rounded-xl bg-white border border-slate-200 shadow-sm"
                >
                  {showLoading ? (
                    <AgentProgress steps={agentMessages} />
                  ) : ai?.role === 'ai' && ai.content?.length > 0 ? (
                    <>
                      <BotResponse
                        content={ai.content}
                        citations={ai.citations || []}
                        onCitationClick={onCitationClick}
                        onHighlight={onHighlight}
                      />
                      {isLast && (
                        <FollowUpQuestions
                          questions={ai.followUpQuestions || []}
                          onSelect={onFollowUp}
                          disabled={isLoading}
                        />
                      )}
                    </>
                  ) : null}
                </div>
              </div>
            </div>
          );
        })}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
