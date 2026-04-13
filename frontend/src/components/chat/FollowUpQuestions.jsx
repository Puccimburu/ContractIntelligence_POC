export default function FollowUpQuestions({ questions = [], onSelect, disabled }) {
  if (!questions.length) return null;

  return (
    <div className="mt-4 pt-3.5 border-t border-slate-200">
      <p className="text-xs font-medium mb-2.5 text-slate-500">
        Suggested questions
      </p>
      <div className="flex flex-col gap-2">
        {questions.map((q, i) => (
          <button
            key={i}
            disabled={disabled}
            onClick={() => onSelect && onSelect(q)}
            className="text-sm px-4 py-3 rounded-lg text-left transition disabled:opacity-40 disabled:cursor-not-allowed bg-white border border-slate-200 text-slate-800 hover:bg-indigo-50 hover:border-indigo-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-300 focus-visible:ring-offset-2 focus-visible:ring-offset-white"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}
