
import React, { useState } from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { Database, Layers, GitBranch, Search, Lightbulb } from 'lucide-react';
import { generateArchitectureAdvice } from '../services/aiService';

const mockMetrics = [
  { name: 'Schema Compliance', current: 65, target: 100 },
  { name: 'Branch Integration', current: 40, target: 100 },
  { name: 'Timestamp Accuracy', current: 30, target: 100 },
  { name: 'Index Coverage', current: 20, target: 100 },
];

const StrategyDashboard: React.FC = () => {
    const [advice, setAdvice] = useState<string>("");
    const [loadingAdvice, setLoadingAdvice] = useState(false);

    const askAdvice = async (topic: string) => {
        setLoadingAdvice(true);
        const prompt = `Provide 3 strategic bullet points for implementing a ${topic} in a documentation quality management framework (AgentQMS). Focus on scalability and containerization.`;
        const result = await generateArchitectureAdvice(prompt);
        setAdvice(result);
        setLoadingAdvice(false);
    }

  return (
    <div className="h-full overflow-y-auto pr-2">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-white mb-2">Strategy & Architecture</h2>
        <p className="text-slate-400">Framework health, indexing strategy, and containerization roadmap.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        {/* Metric Cards */}
        <div className="bg-slate-800 p-5 rounded-xl border border-slate-700">
            <div className="flex items-center gap-3 mb-2 text-blue-400">
                <Database size={24} />
                <h3 className="font-semibold text-lg">Indexing Status</h3>
            </div>
            <p className="text-3xl font-bold text-white">Crisis</p>
            <p className="text-sm text-slate-500 mt-1">Chaotic versioning detected. Implementation of dedicated Indexing Service required.</p>
        </div>

        <div className="bg-slate-800 p-5 rounded-xl border border-slate-700">
            <div className="flex items-center gap-3 mb-2 text-purple-400">
                <Layers size={24} />
                <h3 className="font-semibold text-lg">Containerization</h3>
            </div>
            <p className="text-3xl font-bold text-white">Partial</p>
            <p className="text-sm text-slate-500 mt-1">Refactor Active: `toolkit` removed in favor of `agent_tools`.</p>
        </div>

        <div className="bg-slate-800 p-5 rounded-xl border border-slate-700">
            <div className="flex items-center gap-3 mb-2 text-green-400">
                <GitBranch size={24} />
                <h3 className="font-semibold text-lg">Traceability</h3>
            </div>
            <p className="text-3xl font-bold text-white">Improving</p>
            <p className="text-sm text-slate-500 mt-1">New `branch_name` enforcement active. Timestamp accuracy standardized.</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {/* Chart */}
        <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 h-80">
          <h3 className="text-lg font-semibold text-white mb-4">Framework Health Audit</h3>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={mockMetrics} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
              <XAxis type="number" domain={[0, 100]} stroke="#94a3b8" />
              <YAxis dataKey="name" type="category" stroke="#94a3b8" width={120} tick={{fontSize: 12}} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#fff' }} 
                itemStyle={{ color: '#fff' }}
              />
              <Legend />
              <Bar dataKey="current" name="Current State" fill="#3b82f6" radius={[0, 4, 4, 0]} barSize={20} />
              <Bar dataKey="target" name="Target" fill="#1e293b" stroke="#3b82f6" strokeDasharray="4 4" radius={[0, 4, 4, 0]} barSize={20} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* AI Architect */}
        <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 flex flex-col">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <Lightbulb className="text-yellow-400" /> AI Architect Advisor
            </h3>
            <div className="flex gap-2 mb-4">
                <button onClick={() => askAdvice('Professional Indexing System')} className="bg-slate-700 hover:bg-slate-600 px-3 py-1.5 rounded text-xs text-white transition">Indexing</button>
                <button onClick={() => askAdvice('Containerized Design')} className="bg-slate-700 hover:bg-slate-600 px-3 py-1.5 rounded text-xs text-white transition">Containerization</button>
                <button onClick={() => askAdvice('Self-Auditing Framework')} className="bg-slate-700 hover:bg-slate-600 px-3 py-1.5 rounded text-xs text-white transition">Self-Audit</button>
            </div>
            <div className="flex-1 bg-slate-900 rounded p-4 overflow-y-auto border border-slate-700/50">
                {loadingAdvice ? (
                    <div className="text-slate-500 text-sm animate-pulse">Consulting knowledge base...</div>
                ) : advice ? (
                    <div className="prose prose-invert prose-sm">
                        <p className="whitespace-pre-wrap text-slate-300 text-sm">{advice}</p>
                    </div>
                ) : (
                    <div className="text-slate-600 text-sm italic">Select a topic above to receive strategic architectural recommendations.</div>
                )}
            </div>
        </div>
      </div>

       <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <Search className="text-teal-400" /> Recommended Indexing Structure
            </h3>
            <div className="font-mono text-sm text-slate-400 whitespace-pre bg-slate-950 p-4 rounded border border-slate-800 overflow-x-auto">
{`AgentQMS/
├── agent_tools/            # [Active] Scripts, Validators, & Logic
│   ├── validation/
│   └── generation/
├── registry/               # Central Index
│   ├── global_index.json   # Master ledger
│   └── trace_matrix.json   # Relationship mapping
└── modules/                # Project implementations
    ├── ocr-2/              # Specific project context
    │   ├── audits/
    │   └── plans/
    └── vlm/
`}
            </div>
       </div>
    </div>
  );
};

export default StrategyDashboard;
