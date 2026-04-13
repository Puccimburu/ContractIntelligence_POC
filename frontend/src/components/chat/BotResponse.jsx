import TextView from './TextView';
import TableView from './TableView';
import ChartView from './ChartView';
import HtmlView from './HtmlView';
import CitationList from './CitationList';

export default function BotResponse({ content, citations = [], onCitationClick, onHighlight }) {
  let parsed;
  try {
    if (typeof content === 'string' && content.trim().startsWith('[')) {
      parsed = JSON.parse(content);
    } else if (Array.isArray(content)) {
      parsed = content;
    }
  } catch (_) {}

  if (!Array.isArray(parsed)) {
    return <TextView content={typeof content === 'string' ? content : JSON.stringify(content)} />;
  }

  return (
    <div className="flex flex-col gap-4">
      {parsed.map((item, i) => {
        switch (item?.type) {
          case 'text':  return typeof item.content === 'string' ? <TextView  key={i} content={item.content} /> : null;
          case 'table': return <TableView key={i} headers={item.headers} rows={item.rows} />;
          case 'chart': return <ChartView key={i} chartType={item.chartType} data={item.data} />;
          case 'html':  return typeof item.content === 'string' ? <HtmlView  key={i} content={item.content} /> : null;
          default:      return typeof item?.content === 'string' ? <TextView key={i} content={item.content} /> : null;
        }
      })}

      {citations.length > 0 && (
        <CitationList
          citations={citations}
          onCitationClick={onCitationClick}
          onHighlight={onHighlight}
        />
      )}
    </div>
  );
}
