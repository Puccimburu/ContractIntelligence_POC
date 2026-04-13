export default function TableView({ headers = [], rows = [] }) {
  if (!headers.length && !rows.length) return null;

  return (
    <div className="overflow-x-auto rounded-xl my-3 border border-slate-200 bg-white">
      <table className="min-w-full text-sm">
        {headers.length > 0 && (
          <thead>
            <tr className="bg-slate-50 border-b border-slate-200">
              {headers.map((h, i) => (
                <th
                  key={i}
                  className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide whitespace-nowrap text-slate-500"
                >
                  {String(h ?? '')}
                </th>
              ))}
            </tr>
          </thead>
        )}
        <tbody>
          {rows.map((row, ri) => (
            <tr
              key={ri}
              className={[
                ri > 0 ? 'border-t border-slate-100' : '',
                'hover:bg-slate-50 transition-colors',
              ].join(' ')}
            >
              {(Array.isArray(row) ? row : Object.values(row)).map((cell, ci) => (
                <td key={ci} className="px-4 py-3 text-slate-700">
                  {String(cell ?? '')}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
