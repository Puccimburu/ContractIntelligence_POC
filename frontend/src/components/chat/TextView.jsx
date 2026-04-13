import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkBreaks from 'remark-breaks';

export default function TextView({ content }) {
  if (!content) return null;
  return (
    <div className="md-body">
      <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]}>
        {typeof content === 'string' ? content : String(content)}
      </ReactMarkdown>
    </div>
  );
}
