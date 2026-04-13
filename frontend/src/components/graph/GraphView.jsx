/**
 * GraphView — Contract Knowledge Graph Visualisation
 *
 * Shows two views:
 *   Overview  – Documents, persons, orgs + high-level edges (MASTER_OF, NOVATES, etc.)
 *   Detail    – Section-level graph for a single document (click a doc node to drill in)
 *
 * Uses React Flow (@xyflow/react) + Dagre for automatic layout.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  useReactFlow,
  ReactFlowProvider,
  Handle,
  Position,
  MarkerType,
  Panel,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import dagre from '@dagrejs/dagre';
import api from '../../api/api';

// ---------------------------------------------------------------------------
// Dagre auto-layout
// ---------------------------------------------------------------------------

const NODE_WIDTH = 180;
const NODE_HEIGHT = 56;

function applyDagreLayout(nodes, edges, direction = 'LR') {
  const g = new dagre.graphlib.Graph();
  g.setGraph({ rankdir: direction, ranksep: 80, nodesep: 40 });
  g.setDefaultEdgeLabel(() => ({}));

  nodes.forEach((n) => {
    const w = n.data?.nodeType === 'section' ? 160 : NODE_WIDTH;
    const h = n.data?.nodeType === 'section' ? 48 : NODE_HEIGHT;
    g.setNode(n.id, { width: w, height: h });
  });
  edges.forEach((e) => g.setEdge(e.source, e.target));

  dagre.layout(g);

  return nodes.map((n) => {
    const pos = g.node(n.id);
    return { ...n, position: { x: pos.x - (pos.width || NODE_WIDTH) / 2, y: pos.y - (pos.height || NODE_HEIGHT) / 2 } };
  });
}

// ---------------------------------------------------------------------------
// Custom node — ContractNode
// ---------------------------------------------------------------------------

const ROLE_LABELS = {
  master_agreement: 'Master',
  transaction: 'Work Order',
  modification: 'Novation',
  termination: 'Termination',
  standalone: 'Standalone',
};

function ContractNode({ data, selected }) {
  const nt = data.nodeType;
  const isDoc = nt === 'document';
  const isPerson = nt === 'person';
  const isOrg = nt === 'organization';
  const isSection = nt === 'section';

  const baseStyle = {
    background: data.color || '#E5E7EB',
    border: selected ? '2px solid #1D4ED8' : '1.5px solid rgba(0,0,0,0.12)',
    borderRadius: isPerson || isOrg ? '9999px' : isSection ? '6px' : '10px',
    minWidth: isSection ? 140 : 160,
    maxWidth: isSection ? 200 : 220,
    padding: isSection ? '6px 10px' : '8px 14px',
    boxShadow: selected ? '0 0 0 3px rgba(29,78,216,0.2)' : '0 2px 8px rgba(0,0,0,0.10)',
    cursor: isDoc ? 'pointer' : 'default',
    transition: 'box-shadow 0.15s, border 0.15s',
  };

  const role = (data.properties?.functionalRole) || '';
  const roleLabel = ROLE_LABELS[role] || '';

  return (
    <div style={baseStyle}>
      <Handle type="target" position={Position.Left} style={{ opacity: 0 }} />
      <Handle type="source" position={Position.Right} style={{ opacity: 0 }} />

      {/* Badge */}
      {isDoc && roleLabel && (
        <div style={{
          fontSize: 9, fontWeight: 700, letterSpacing: '0.05em',
          textTransform: 'uppercase', opacity: 0.65,
          color: '#1e293b', marginBottom: 2,
        }}>
          {roleLabel}
        </div>
      )}
      {(isPerson || isOrg) && (
        <div style={{ fontSize: 9, fontWeight: 700, textTransform: 'uppercase', opacity: 0.55, marginBottom: 2, color: '#fff' }}>
          {isPerson ? 'Person' : 'Organisation'}
        </div>
      )}
      {isSection && data.properties?.clauseType && data.properties.clauseType !== 'other' && (
        <div style={{ fontSize: 9, fontWeight: 600, textTransform: 'uppercase', opacity: 0.55, marginBottom: 1 }}>
          {data.properties.clauseType.replace(/_/g, ' ')}
        </div>
      )}

      {/* Label */}
      <div style={{
        fontSize: isSection ? 11 : 12,
        fontWeight: 600,
        lineHeight: 1.3,
        color: (isPerson || isOrg) ? '#fff' : '#0f172a',
        wordBreak: 'break-word',
      }}>
        {data.label}
      </div>

      {/* Risk badge */}
      {isSection && data.properties?.riskLevel >= 4 && (
        <div style={{ fontSize: 9, color: '#DC2626', fontWeight: 700, marginTop: 2 }}>
          ⚠ HIGH RISK
        </div>
      )}

      {/* Aliases tooltip hint */}
      {data.aliases?.length > 1 && (
        <div style={{ fontSize: 9, opacity: 0.5, marginTop: 2 }} title={data.aliases.join(', ')}>
          +{data.aliases.length - 1} alias{data.aliases.length > 2 ? 'es' : ''}
        </div>
      )}
    </div>
  );
}

const nodeTypes = { contractNode: ContractNode };

// ---------------------------------------------------------------------------
// Edge legend
// ---------------------------------------------------------------------------

const EDGE_LEGEND = [
  { type: 'MASTER_OF',   color: '#3B82F6', label: 'Master Of' },
  { type: 'NOVATES',     color: '#F59E0B', label: 'Novates' },
  { type: 'TERMINATES',  color: '#EF4444', label: 'Terminates' },
  { type: 'RENEWS',      color: '#10B981', label: 'Renews' },
  { type: 'PARTY_TO',    color: '#A78BFA', label: 'Party To' },
  { type: 'CONDITIONS',  color: '#F59E0B', label: 'Conditions' },
  { type: 'SUPERSEDES',  color: '#EF4444', label: 'Supersedes' },
  { type: 'EXCEPTIONS',  color: '#F97316', label: 'Exceptions' },
  { type: 'OBLIGATES',   color: '#8B5CF6', label: 'Obligates' },
  { type: 'DEFINES',     color: '#10B981', label: 'Defines' },
  { type: 'REFERENCES',  color: '#60A5FA', label: 'References' },
];

function Legend({ types }) {
  const visible = EDGE_LEGEND.filter(l => types.has(l.type));
  if (!visible.length) return null;
  return (
    <div style={{
      position: 'absolute', bottom: 12, left: 12, zIndex: 10,
      background: 'rgba(255,255,255,0.95)', borderRadius: 8, padding: '8px 12px',
      boxShadow: '0 2px 8px rgba(0,0,0,0.12)', fontSize: 11,
      display: 'flex', flexDirection: 'column', gap: 4,
    }}>
      <div style={{ fontWeight: 700, fontSize: 10, marginBottom: 2, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
        Relationships
      </div>
      {visible.map(l => (
        <div key={l.type} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <div style={{ width: 20, height: 2, background: l.color, borderRadius: 1 }} />
          <span style={{ color: '#334155' }}>{l.label}</span>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Inner graph renderer
// ---------------------------------------------------------------------------

function GraphCanvas({ nodes: initNodes, edges: initEdges, onNodeClick, direction }) {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const { fitView } = useReactFlow();
  const edgeTypes = useMemo(() => new Set(initEdges.map(e => e.data?.edgeType)), [initEdges]);

  useEffect(() => {
    if (!initNodes.length) return;
    const laidOut = applyDagreLayout(initNodes, initEdges, direction);
    setNodes(laidOut);
    const fmtLabel = (raw) => raw
      ? raw.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
      : '';

    setEdges(initEdges.map(e => ({
      ...e,
      label: fmtLabel(e.label),
      markerEnd: { type: MarkerType.ArrowClosed, width: 14, height: 14, color: e.style?.stroke || '#9CA3AF' },
      labelStyle: { fontSize: 9, fontWeight: 600 },
      labelBgStyle: { fill: 'rgba(255,255,255,0.85)', fillOpacity: 0.85 },
      labelBgPadding: [3, 4],
      labelBgBorderRadius: 4,
    })));
    setTimeout(() => fitView({ padding: 0.15, duration: 300 }), 50);
  }, [initNodes, initEdges, direction]);

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick}
        fitView
        fitViewOptions={{ padding: 0.15 }}
        minZoom={0.15}
        maxZoom={2.5}
        proOptions={{ hideAttribution: true }}
      >
        <Background color="#e2e8f0" gap={18} size={1} />
        <Controls position="top-right" showInteractive={false} />
        <MiniMap
          position="bottom-right"
          style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 6 }}
          nodeColor={(n) => n.data?.color || '#E5E7EB'}
          maskColor="rgba(241,245,249,0.7)"
        />
      </ReactFlow>
      <Legend types={edgeTypes} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Stats bar
// ---------------------------------------------------------------------------

function StatsBar({ stats, view, onBack, docLabel }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 12, padding: '6px 14px',
      background: '#f8fafc', borderBottom: '1px solid #e2e8f0', fontSize: 12, color: '#64748b',
    }}>
      {view === 'detail' && (
        <button
          onClick={onBack}
          style={{ fontSize: 12, color: '#3B82F6', fontWeight: 600, cursor: 'pointer', background: 'none', border: 'none', padding: 0 }}
        >
          ← Overview
        </button>
      )}
      {view === 'detail' && docLabel && (
        <span style={{ fontWeight: 600, color: '#0f172a' }}>{docLabel}</span>
      )}
      {stats && (
        <>
          <span>{stats.visibleNodes ?? stats.totalNodes} nodes</span>
          <span>·</span>
          <span>{stats.visibleEdges ?? stats.totalEdges} relationships</span>
          {stats.totalNodes > (stats.visibleNodes ?? stats.totalNodes) && (
            <>
              <span>·</span>
              <span>{stats.totalNodes} total nodes (section detail per document)</span>
            </>
          )}
        </>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main GraphView (no Router needed)
// ---------------------------------------------------------------------------

export default function GraphView({ conversationId, onClose }) {
  const [view, setView] = useState('overview'); // 'overview' | 'detail'
  const [overviewData, setOverviewData] = useState(null);
  const [detailData, setDetailData] = useState(null);
  const [detailFileId, setDetailFileId] = useState(null);
  const [detailLabel, setDetailLabel] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Load overview graph
  useEffect(() => {
    if (!conversationId) return;
    setLoading(true);
    setError(null);
    api.get(`/ci/conversations/${conversationId}/graph`)
      .then(res => setOverviewData(res.data))
      .catch(err => setError(err.response?.data?.detail || 'Failed to load graph'))
      .finally(() => setLoading(false));
  }, [conversationId]);

  // Load detail graph when a document node is clicked
  const loadDetail = useCallback(async (fileId, label) => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.get(`/ci/files/${fileId}/graph`);
      setDetailData(res.data);
      setDetailFileId(fileId);
      setDetailLabel(label);
      setView('detail');
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load section graph');
    } finally {
      setLoading(false);
    }
  }, []);

  const handleNodeClick = useCallback((_event, node) => {
    if (node.data?.nodeType === 'document' && node.data?.fileId) {
      loadDetail(node.data.fileId, node.data.label);
    }
  }, [loadDetail]);

  const handleBack = useCallback(() => {
    setView('overview');
    setDetailData(null);
    setDetailFileId(null);
  }, []);

  const current = view === 'overview' ? overviewData : detailData;
  const stats = current?.stats || null;
  const direction = view === 'detail' ? 'TB' : 'LR';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', background: '#fff' }}>
      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '10px 16px', borderBottom: '1px solid #e2e8f0',
        background: '#fff', flexShrink: 0,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" strokeWidth="2">
            <circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/>
            <line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/>
            <line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/>
          </svg>
          <span style={{ fontWeight: 700, fontSize: 14, color: '#0f172a' }}>
            Contract Knowledge Graph
          </span>
          <span style={{
            fontSize: 10, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em',
            padding: '2px 8px', borderRadius: 9999,
            background: view === 'overview' ? '#DBEAFE' : '#D1FAE5',
            color: view === 'overview' ? '#1D4ED8' : '#065F46',
          }}>
            {view === 'overview' ? 'Overview' : 'Section Detail'}
          </span>
        </div>
        <button
          onClick={onClose}
          style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#94a3b8', fontSize: 18, lineHeight: 1 }}
          title="Close graph"
        >
          ✕
        </button>
      </div>

      {/* Stats bar */}
      <StatsBar
        stats={stats}
        view={view}
        onBack={handleBack}
        docLabel={detailLabel}
      />

      {/* Hint */}
      {view === 'overview' && !loading && overviewData?.nodes?.length > 0 && (
        <div style={{ padding: '4px 14px', background: '#eff6ff', borderBottom: '1px solid #dbeafe', fontSize: 11, color: '#3730a3' }}>
          Click a document node to explore its clause-level relationships
        </div>
      )}

      {/* Main area */}
      <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
        {loading && (
          <div style={{
            position: 'absolute', inset: 0, display: 'flex', alignItems: 'center',
            justifyContent: 'center', background: 'rgba(255,255,255,0.8)', zIndex: 20, flexDirection: 'column', gap: 12,
          }}>
            <div style={{ width: 32, height: 32, border: '3px solid #e2e8f0', borderTopColor: '#3B82F6', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
            <span style={{ fontSize: 13, color: '#64748b' }}>Building knowledge graph…</span>
          </div>
        )}

        {error && (
          <div style={{ padding: 24, color: '#EF4444', fontSize: 13 }}>
            <strong>Error:</strong> {error}
            <br /><br />
            <span style={{ color: '#64748b' }}>
              Upload and fully process your contracts first, then the graph will be available.
            </span>
          </div>
        )}

        {!loading && !error && current?.nodes?.length === 0 && (
          <div style={{ padding: 32, textAlign: 'center', color: '#94a3b8' }}>
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" style={{ margin: '0 auto 12px' }}>
              <circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/>
              <line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/>
              <line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/>
            </svg>
            <p style={{ fontWeight: 600, color: '#475569' }}>No graph data yet</p>
            <p style={{ fontSize: 12, marginTop: 4 }}>
              Upload contracts and wait for them to finish processing, then reload the graph.
            </p>
          </div>
        )}

        {!loading && !error && current?.nodes?.length > 0 && (
          <ReactFlowProvider>
            <GraphCanvas
              key={view === 'overview' ? 'overview' : detailFileId}
              nodes={current.nodes}
              edges={current.edges}
              onNodeClick={handleNodeClick}
              direction={direction}
            />
          </ReactFlowProvider>
        )}
      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}
