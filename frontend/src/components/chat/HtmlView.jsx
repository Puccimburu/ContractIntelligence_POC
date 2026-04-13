import { useEffect, useRef } from 'react';
import DOMPurify from 'dompurify';

export default function HtmlView({ content }) {
  const ref = useRef(null);

  useEffect(() => {
    if (!ref.current) return;
    ref.current.srcdoc = DOMPurify.sanitize(
      typeof content === 'string' ? content : String(content)
    );
  }, [content]);

  return (
    <iframe
      ref={ref}
      title="html-content"
      sandbox="allow-same-origin"
      className="w-full rounded-xl border border-gray-200"
      style={{ minHeight: 200 }}
    />
  );
}
