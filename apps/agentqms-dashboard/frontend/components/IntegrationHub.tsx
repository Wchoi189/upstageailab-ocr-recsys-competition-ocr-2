
import React, { useState, useEffect } from 'react';
import { Terminal, Bot, Copy, CheckCircle, Database, Activity, HardDrive, RefreshCw, Share2 } from 'lucide-react';
import { generateAgentSystemPrompt } from '../services/aiService';
import { DBStatus } from '../types';

const IntegrationHub: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'bootstrap' | 'protocol' | 'database'>('bootstrap');
  const [projectContext, setProjectContext] = useState('');
  const [generatedProtocol, setGeneratedProtocol] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  
  // Mock Database State
  const [dbStatus, setDbStatus] = useState<DBStatus>({
    connected: false,
    version: '0.0.0',
    lastBackup: 'Never',
    recordCount: 0,
    health: 'offline',
    issues: ['Database file not found in .agentqms/']
  });

  const [isInitializingDB, setIsInitializingDB] = useState(false);

  const bootstrapScript = `#!/bin/bash
# AgentQMS Bootstrap Script
# Run this at your project root to initialize the framework structure

echo "Initializing AgentQMS Structure..."

# 1. Create Framework Directory
# 'agent_tools' replaces previous 'toolkit'/'core' modules
mkdir -p AgentQMS/{agent_tools,registry,templates}

# 2. Create Local Configuration (Project Specific)
mkdir -p .agentqms
cat > .agentqms/config.json <<EOF
{
  "project_root": "$(pwd)",
  "active_module": "default",
  "enforce_branch_names": true,
  "timezone": "KST"
}
EOF

# 3. Create Module Directory for Project Artifacts
mkdir -p AgentQMS/modules/main-project/{assessments,plans,audits}

echo "✅ AgentQMS initialized with 'agent_tools' structure."
echo "👉 Use '.agentqms/config.json' for local settings."
`;

  const initializeDB = () => {
    setIsInitializingDB(true);
    // Simulate API/Script latency
    setTimeout(() => {
        setDbStatus({
            connected: true,
            version: '1.2.0',
            lastBackup: 'Just now',
            recordCount: 12,
            health: 'healthy',
            issues: []
        });
        setIsInitializingDB(false);
    }, 1500);
  };

  const handleGenerateProtocol = async () => {
    if (!projectContext.trim()) return;
    setIsGenerating(true);
    try {
      const result = await generateAgentSystemPrompt(projectContext);
      setGeneratedProtocol(result);
    } catch (e) {
      console.error(e);
      alert("Generation failed. Check Settings to ensure API Key is valid.");
    } finally {
      setIsGenerating(false);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    alert("Copied to clipboard!");
  };

  return (
    <div className="h-full flex flex-col overflow-hidden">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-white mb-2">Integration Hub</h2>
        <p className="text-slate-400">Tools to install AgentQMS, enforce adoption, and manage the Tracking Database.</p>
      </div>

      <div className="flex gap-4 mb-6 border-b border-slate-700">
        <button
          onClick={() => setActiveTab('bootstrap')}
          className={`pb-3 px-2 flex items-center gap-2 transition-colors ${
            activeTab === 'bootstrap' 
              ? 'text-blue-400 border-b-2 border-blue-400 font-medium' 
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <Terminal size={18} />
          Bootstrap Installer
        </button>
        <button
          onClick={() => setActiveTab('protocol')}
          className={`pb-3 px-2 flex items-center gap-2 transition-colors ${
            activeTab === 'protocol' 
              ? 'text-purple-400 border-b-2 border-purple-400 font-medium' 
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <Bot size={18} />
          Agent Protocol
        </button>
        <button
          onClick={() => setActiveTab('database')}
          className={`pb-3 px-2 flex items-center gap-2 transition-colors ${
            activeTab === 'database' 
              ? 'text-green-400 border-b-2 border-green-400 font-medium' 
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <Database size={18} />
          Tracking Database
        </button>
      </div>

      <div className="flex-1 overflow-hidden">
        {activeTab === 'bootstrap' && (
          <div className="h-full flex flex-col animate-in fade-in duration-300">
            <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 mb-6">
              <h3 className="text-lg font-semibold text-white mb-2">Quick Start Installation</h3>
              <p className="text-slate-400 text-sm mb-4">
                Run this script in your project root to scaffold the canonical AgentQMS directory structure. 
                It creates the separation between the framework (`AgentQMS/`) and local config (`.agentqms/`).
              </p>
              
              <div className="relative group">
                <div className="absolute top-3 right-3 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button 
                        onClick={() => copyToClipboard(bootstrapScript)}
                        className="bg-slate-700 hover:bg-slate-600 text-white p-1.5 rounded"
                        title="Copy Script"
                    >
                        <Copy size={16} />
                    </button>
                </div>
                <pre className="bg-slate-950 p-4 rounded-lg font-mono text-sm text-green-400 overflow-x-auto border border-slate-900">
                  {bootstrapScript}
                </pre>
              </div>
            </div>

            <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 flex-1 overflow-y-auto">
                <h3 className="text-lg font-semibold text-white mb-4">Directory Map</h3>
                <div className="font-mono text-sm text-slate-300 whitespace-pre">
{`PROJECT_ROOT/
├── .agentqms/                 # [LOCAL] Configuration & temporary state
│   └── config.json            # Active branch, timezone settings
├── AgentQMS/                  # [FRAMEWORK] The immutable framework
│   ├── agent_tools/           # [NEW] Consolidated tools & scripts
│   ├── templates/             # Markdown Templates (Plans, Audits)
│   └── registry/              # Central Artifact Index
└── src/                       # Your source code
`}
                </div>
            </div>
          </div>
        )}

        {activeTab === 'protocol' && (
          <div className="h-full grid grid-cols-1 lg:grid-cols-2 gap-6 animate-in fade-in duration-300">
            <div className="flex flex-col h-full">
              <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 mb-4 flex-1">
                <h3 className="text-lg font-semibold text-white mb-2 flex items-center gap-2">
                    <Share2 className="text-purple-400" size={20} /> Context Injector
                </h3>
                <p className="text-slate-400 text-sm mb-4">
                  Describe your project context. We will generate a "System Prompt" you can paste into 
                  Cursor, Windsurf, or ChatGPT to force them to use AgentQMS.
                </p>
                <textarea 
                    className="w-full h-32 bg-slate-900 border border-slate-700 rounded p-3 text-slate-300 focus:border-purple-500 focus:outline-none resize-none mb-4"
                    placeholder="e.g., This is a Python OCR project using PyTorch. We are strict about code reviews..."
                    value={projectContext}
                    onChange={(e) => setProjectContext(e.target.value)}
                />
                <button
                    onClick={handleGenerateProtocol}
                    disabled={isGenerating || !projectContext}
                    className={`w-full py-3 rounded-lg font-semibold transition-all flex items-center justify-center gap-2 ${
                        isGenerating ? 'bg-slate-700 text-slate-500' : 'bg-purple-600 hover:bg-purple-500 text-white'
                    }`}
                >
                    {isGenerating ? 'Synthesizing Protocol...' : 'Generate System Prompt'}
                </button>
              </div>

              <div className="bg-slate-800/50 p-4 rounded-xl border border-slate-700/50">
                <h4 className="font-semibold text-slate-300 mb-2 text-sm">Why use this?</h4>
                <ul className="text-xs text-slate-400 space-y-2">
                    <li className="flex gap-2"><CheckCircle size={14} className="text-green-500" /> Forces Agents to check schemas before coding.</li>
                    <li className="flex gap-2"><CheckCircle size={14} className="text-green-500" /> Auto-includes branch names in frontmatter.</li>
                    <li className="flex gap-2"><CheckCircle size={14} className="text-green-500" /> Awareness of 'agent_tools' location.</li>
                </ul>
              </div>
            </div>

            <div className="bg-slate-950 rounded-xl border border-slate-800 p-0 flex flex-col h-full overflow-hidden">
                <div className="p-3 border-b border-slate-800 bg-slate-900 flex justify-between items-center">
                    <span className="text-xs font-mono text-purple-300">system_protocol.md</span>
                    <button 
                        onClick={() => copyToClipboard(generatedProtocol)}
                        disabled={!generatedProtocol}
                        className="flex items-center gap-2 text-xs bg-purple-900/50 hover:bg-purple-900 text-purple-200 px-3 py-1.5 rounded transition-colors disabled:opacity-50"
                    >
                        <Copy size={14} /> Copy
                    </button>
                </div>
                <div className="flex-1 overflow-y-auto p-4">
                    {generatedProtocol ? (
                        <pre className="text-xs font-mono text-slate-300 whitespace-pre-wrap">
                            {generatedProtocol}
                        </pre>
                    ) : (
                        <div className="h-full flex flex-col items-center justify-center text-slate-600">
                            <Bot size={48} className="mb-4 opacity-20" />
                            <p className="text-sm">Context waiting to be generated...</p>
                        </div>
                    )}
                </div>
            </div>
          </div>
        )}

        {activeTab === 'database' && (
           <div className="h-full animate-in fade-in duration-300 flex flex-col gap-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                 {/* Status Card */}
                 <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
                    <div className="flex items-center gap-3 mb-4">
                       <Activity className={dbStatus.health === 'healthy' ? 'text-green-400' : 'text-red-400'} size={24} />
                       <h3 className="font-semibold text-white">System Health</h3>
                    </div>
                    <div className="text-3xl font-bold text-white mb-2 capitalize">{dbStatus.health}</div>
                    <div className="text-xs text-slate-500">
                        {dbStatus.connected ? 'Operational' : 'Connection Required'}
                    </div>
                 </div>

                 {/* Version Card */}
                 <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
                    <div className="flex items-center gap-3 mb-4">
                       <HardDrive className="text-blue-400" size={24} />
                       <h3 className="font-semibold text-white">Schema Version</h3>
                    </div>
                    <div className="text-3xl font-bold text-white mb-2">{dbStatus.version}</div>
                    <div className="text-xs text-slate-500">
                        {dbStatus.recordCount} Artifact Records Indexed
                    </div>
                 </div>

                 {/* Action Card */}
                 <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 flex flex-col justify-between">
                    <div className="flex items-center gap-3 mb-4">
                       <Database className="text-purple-400" size={24} />
                       <h3 className="font-semibold text-white">Control</h3>
                    </div>
                    {dbStatus.connected ? (
                        <button 
                            className="w-full bg-slate-700 hover:bg-slate-600 text-slate-200 py-2 rounded flex items-center justify-center gap-2 transition-colors"
                            onClick={() => {
                                setDbStatus(prev => ({...prev, connected: false, health: 'offline', recordCount: 0}))
                            }}
                        >
                            Disconnect
                        </button>
                    ) : (
                        <button 
                            onClick={initializeDB}
                            disabled={isInitializingDB}
                            className="w-full bg-green-600 hover:bg-green-500 text-white py-2 rounded flex items-center justify-center gap-2 transition-colors disabled:opacity-50"
                        >
                            {isInitializingDB ? <RefreshCw className="animate-spin" size={16}/> : <CheckCircle size={16} />}
                            Initialize Database
                        </button>
                    )}
                 </div>
              </div>

              {/* Console / Log Area */}
              <div className="flex-1 bg-slate-950 rounded-xl border border-slate-800 p-4 font-mono text-sm overflow-hidden flex flex-col">
                  <div className="text-slate-500 mb-2 text-xs uppercase font-bold tracking-wider">System Log</div>
                  <div className="flex-1 overflow-y-auto space-y-2">
                     {!dbStatus.connected && <div className="text-yellow-500">[WARN] Tracking DB connection not established.</div>}
                     {!dbStatus.connected && <div className="text-slate-400">[INFO] Waiting for initialization...</div>}
                     {dbStatus.connected && (
                         <>
                            <div className="text-green-500">[SUCCESS] Connection established to .agentqms/tracking.db</div>
                            <div className="text-slate-300">[INFO] Schema verification passed (v1.2.0)</div>
                            <div className="text-slate-300">[INFO] Indexed {dbStatus.recordCount} artifacts from /AgentQMS/registry</div>
                            <div className="text-blue-400">[READY] Monitoring active.</div>
                         </>
                     )}
                     {dbStatus.issues.map((issue, idx) => (
                         <div key={idx} className="text-red-400">[ERROR] {issue}</div>
                     ))}
                  </div>
              </div>
           </div>
        )}
      </div>
    </div>
  );
};

export default IntegrationHub;
