import React, { useState } from 'react';
import { Network, Grid, Box, FileText, Layers, ZoomIn, Info } from 'lucide-react';
import { ContextNode, ContextBundle } from '../types';

// Mock Data representing the "Registry" or "Tracking DB"
const MOCK_BUNDLE: ContextBundle = {
  name: "OCR-2 Feature Context",
  nodes: [
    { id: 'root', label: 'OCR-2 Project', type: 'root', status: 'active', connections: ['mod1', 'mod2'] },
    { id: 'mod1', label: 'Preprocessing Module', type: 'module', status: 'active', connections: ['p1', 'a1'] },
    { id: 'mod2', label: 'Inference Engine', type: 'module', status: 'active', connections: ['p2'] },
    { id: 'p1', label: 'Image Norm Plan', type: 'plan', status: 'active', connections: ['r1'] },
    { id: 'a1', label: 'Norm Audit', type: 'audit', status: 'active', connections: [] },
    { id: 'p2', label: 'Model Load Plan', type: 'plan', status: 'pending', connections: [] },
    { id: 'r1', label: 'Bug Report #402', type: 'report', status: 'archived', connections: [] },
  ]
};

const ContextExplorer: React.FC = () => {
  const [viewMode, setViewMode] = useState<'graph' | 'matrix'>('graph');
  const [selectedNode, setSelectedNode] = useState<ContextNode | null>(null);

  // Simple Graph Rendering Calculation
  const renderGraph = () => {
    // A simplified manual layout for the visualization request
    // In a real app, use d3-force or react-flow
    const positions: Record<string, { x: number; y: number }> = {
        'root': { x: 400, y: 300 },
        'mod1': { x: 250, y: 150 },
        'mod2': { x: 550, y: 150 },
        'p1': { x: 150, y: 100 },
        'a1': { x: 200, y: 50 },
        'p2': { x: 600, y: 100 },
        'r1': { x: 100, y: 150 },
    };

    return (
        <svg className="w-full h-full bg-slate-900 rounded-xl border border-slate-700 cursor-move">
            <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="20" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#475569" />
                </marker>
            </defs>
            {/* Edges */}
            {MOCK_BUNDLE.nodes.map(node => (
                node.connections.map(targetId => {
                    const start = positions[node.id];
                    const end = positions[targetId];
                    if (!start || !end) return null;
                    return (
                        <line 
                            key={`${node.id}-${targetId}`}
                            x1={start.x} y1={start.y}
                            x2={end.x} y2={end.y}
                            stroke="#334155"
                            strokeWidth="2"
                            markerEnd="url(#arrowhead)"
                        />
                    );
                })
            ))}
            {/* Nodes */}
            {MOCK_BUNDLE.nodes.map(node => {
                const pos = positions[node.id];
                if (!pos) return null;
                const isSelected = selectedNode?.id === node.id;
                
                let color = '#3b82f6'; // blue (default)
                if (node.type === 'root') color = '#a855f7'; // purple
                if (node.type === 'plan') color = '#10b981'; // green
                if (node.type === 'audit') color = '#f59e0b'; // amber
                if (node.type === 'report') color = '#ef4444'; // red

                return (
                    <g 
                        key={node.id} 
                        onClick={() => setSelectedNode(node)}
                        className="cursor-pointer transition-opacity hover:opacity-80"
                    >
                        <circle 
                            cx={pos.x} cy={pos.y} 
                            r={node.type === 'root' ? 30 : 20} 
                            fill={color} 
                            stroke={isSelected ? '#fff' : 'none'}
                            strokeWidth="2"
                            className="drop-shadow-lg"
                        />
                        <text 
                            x={pos.x} y={pos.y + (node.type === 'root' ? 45 : 35)} 
                            textAnchor="middle" 
                            fill="#cbd5e1" 
                            className="text-xs font-mono select-none"
                        >
                            {node.label}
                        </text>
                    </g>
                );
            })}
        </svg>
    );
  };

  const renderMatrix = () => {
    return (
        <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
            <table className="w-full text-left text-sm text-slate-300">
                <thead className="bg-slate-900 text-slate-400 uppercase font-mono text-xs">
                    <tr>
                        <th className="px-6 py-3">Module</th>
                        <th className="px-6 py-3">Artifact ID</th>
                        <th className="px-6 py-3">Type</th>
                        <th className="px-6 py-3">Status</th>
                        <th className="px-6 py-3">Coverage</th>
                    </tr>
                </thead>
                <tbody className="divide-y divide-slate-700">
                    {MOCK_BUNDLE.nodes.filter(n => n.type !== 'root').map(node => (
                        <tr key={node.id} className="hover:bg-slate-700/50 transition-colors cursor-pointer" onClick={() => setSelectedNode(node)}>
                            <td className="px-6 py-4 font-medium text-white">{node.type === 'module' ? node.label : '-'}</td>
                            <td className="px-6 py-4 font-mono text-slate-400">{node.id}</td>
                            <td className="px-6 py-4">
                                <span className={`px-2 py-1 rounded text-xs font-semibold
                                    ${node.type === 'plan' ? 'bg-green-500/10 text-green-400' : 
                                      node.type === 'audit' ? 'bg-amber-500/10 text-amber-400' :
                                      node.type === 'report' ? 'bg-red-500/10 text-red-400' : 'bg-blue-500/10 text-blue-400'
                                    }`}>
                                    {node.type.toUpperCase()}
                                </span>
                            </td>
                            <td className="px-6 py-4">{node.status}</td>
                            <td className="px-6 py-4">
                                {node.connections.length > 0 ? (
                                    <span className="flex items-center gap-1 text-green-400 text-xs">
                                        <Layers size={14} /> Linked
                                    </span>
                                ) : (
                                    <span className="flex items-center gap-1 text-slate-500 text-xs">
                                        Orphan
                                    </span>
                                )}
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
  };

  return (
    <div className="h-full flex flex-col">
      <div className="mb-6 flex justify-between items-end">
        <div>
            <h2 className="text-2xl font-bold text-white mb-2">Context Explorer</h2>
            <p className="text-slate-400">Visualize "Context Bundles" and the Tracking DB Traceability Matrix.</p>
        </div>
        <div className="flex bg-slate-800 rounded-lg p-1 border border-slate-700">
            <button 
                onClick={() => setViewMode('graph')}
                className={`flex items-center gap-2 px-3 py-1.5 rounded text-sm transition-all ${viewMode === 'graph' ? 'bg-blue-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'}`}
            >
                <Network size={16} /> Neural Graph
            </button>
            <button 
                onClick={() => setViewMode('matrix')}
                className={`flex items-center gap-2 px-3 py-1.5 rounded text-sm transition-all ${viewMode === 'matrix' ? 'bg-blue-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'}`}
            >
                <Grid size={16} /> Trace Matrix
            </button>
        </div>
      </div>

      <div className="flex-1 min-h-0 flex gap-6">
         {/* Main Visualization */}
         <div className="flex-1 flex flex-col">
            {viewMode === 'graph' ? renderGraph() : renderMatrix()}
         </div>

         {/* Inspector Panel */}
         <div className="w-80 bg-slate-800 border border-slate-700 rounded-xl p-5 flex flex-col">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <Info className="text-blue-400" /> Node Inspector
            </h3>
            
            {selectedNode ? (
                <div className="space-y-4 animate-in fade-in">
                    <div className="p-4 bg-slate-900 rounded-lg border border-slate-700">
                        <div className="text-xs text-slate-500 uppercase font-bold mb-1">Artifact ID</div>
                        <div className="font-mono text-lg text-white">{selectedNode.id}</div>
                    </div>

                    <div>
                        <div className="text-xs text-slate-500 uppercase font-bold mb-1">Name</div>
                        <div className="text-slate-200">{selectedNode.label}</div>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <div className="text-xs text-slate-500 uppercase font-bold mb-1">Type</div>
                            <span className="capitalize text-sm text-blue-300">{selectedNode.type}</span>
                        </div>
                        <div>
                            <div className="text-xs text-slate-500 uppercase font-bold mb-1">Status</div>
                            <span className="capitalize text-sm text-green-300">{selectedNode.status}</span>
                        </div>
                    </div>

                    <div className="pt-4 border-t border-slate-700">
                        <div className="text-xs text-slate-500 uppercase font-bold mb-2">Dependencies</div>
                        {selectedNode.connections.length > 0 ? (
                            <ul className="space-y-1">
                                {selectedNode.connections.map(c => (
                                    <li key={c} className="flex items-center gap-2 text-sm text-slate-400">
                                        <div className="w-1.5 h-1.5 rounded-full bg-slate-500"></div>
                                        {c}
                                    </li>
                                ))}
                            </ul>
                        ) : (
                            <p className="text-sm text-slate-600 italic">No outgoing dependencies.</p>
                        )}
                    </div>
                </div>
            ) : (
                <div className="flex-1 flex flex-col items-center justify-center text-slate-500">
                    <Box size={48} className="mb-4 opacity-20" />
                    <p className="text-sm text-center">Select a node in the graph or matrix to view details.</p>
                </div>
            )}
         </div>
      </div>
    </div>
  );
};

export default ContextExplorer;