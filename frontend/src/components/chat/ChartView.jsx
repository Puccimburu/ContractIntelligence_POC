import {
  Chart as ChartJS, CategoryScale, LinearScale, BarElement, LineElement,
  PointElement, ArcElement, RadialLinearScale, Title, Tooltip, Legend,
} from 'chart.js';
import { Bar, Line, Pie, Doughnut, Radar } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale, LinearScale, BarElement, LineElement, PointElement,
  ArcElement, RadialLinearScale, Title, Tooltip, Legend
);

const CHART_COMPONENTS = { bar: Bar, line: Line, pie: Pie, doughnut: Doughnut, radar: Radar };
const CARTESIAN_TYPES = new Set(['bar', 'line']);

export default function ChartView({ chartType = 'bar', data }) {
  if (!data) return null;

  const Component = CHART_COMPONENTS[chartType] || Bar;

  const options = {
    responsive: true,
    maintainAspectRatio: true,
    plugins: {
      legend: { labels: { color: '#6b7280', font: { size: 12 } } },
      tooltip: {
        backgroundColor: '#111827',
        padding: 10,
        cornerRadius: 8,
        titleColor: '#f9fafb',
        bodyColor: '#d1d5db',
      },
    },
    ...(CARTESIAN_TYPES.has(chartType) ? {
      scales: {
        x: { ticks: { color: '#9ca3af', font: { size: 11 } }, grid: { color: '#f3f4f6' } },
        y: { ticks: { color: '#9ca3af', font: { size: 11 } }, grid: { color: '#f3f4f6' } },
      },
    } : {}),
  };

  return (
    <div className="rounded-xl border border-gray-200 p-4 bg-white my-2" style={{ maxHeight: 380 }}>
      <Component data={data} options={options} />
    </div>
  );
}
