
import React, { useState } from 'react';
import { ShieldCheck, AlertTriangle, CheckCircle, Loader2, ArrowRight, Terminal, Wand2 } from 'lucide-react';
import { auditDocumentation } from '../services/aiService';
import { bridgeService } from '../services/bridgeService';
import { AuditResponse, AuditToolConfig } from '../types';
import TrackingStatusComponent from './TrackingStatus';

// Mock Tool Definitions based on 'AgentQMS/agent_tools/audit/*.py'
const AUDIT_TOOLS: AuditToolConfig[] = [
    {
        id: 'validate_frontmatter',
        name: 'Frontmatter Validator',
        description: 'Checks for mandatory fields (branch_name, timestamp) in YAML header.',
        command: 'python AgentQMS/agent_tools/audit/validate_frontmatter.py',
        scriptPath: 'agent_tools/audit/validate_frontmatter.py',
        args: [{ name: 'Target File', flag: '--file', type: 'text' }]
    },
    {
        id: 'check_links',
        name: 'Dead Link Checker',
        description: 'Scans markdown files for broken internal and external links.',
        command: 'python AgentQMS/agent_tools/audit/check_links.py',
        scriptPath: 'agent_tools/audit/check_links.py',
        args: [{ name: 'Directory', flag: '--dir', type: 'text' }]
    },
    {
        id: 'structure_audit',
        name: 'Structural Integrity',
        description: 'Verifies that the folder structure matches the .agentqms/config.json rules.',
        command: 'python AgentQMS/agent_tools/audit/structure_check.py',
        scriptPath: 'agent_tools/audit/structure_check.py',
        args: []
    }
];

const FrameworkAuditor: React.FC = () => {
  const [mode, setMode] = useState<'ai' | 'tool'>('tool');

  // AI State
  const [inputContent, setInputContent] = useState('');
  const [isAuditing, setIsAuditing] = useState(false);
  const [result, setResult] = useState<AuditResponse | null>(null);

  // Tool State
  const [selectedTool, setSelectedTool] = useState<AuditToolConfig>(AUDIT_TOOLS[0]);
  const [toolArgs, setToolArgs] = useState<Record<string, string>>({});
  const [generatedCommand, setGeneratedCommand] = useState('');

  const handleAiAudit = async () => {
    if (!inputContent.trim()) return;
    setIsAuditing(true);
    setResult(null);

    try {
      const data = await auditDocumentation(inputContent, 'Generic');
      setResult(data);
    } catch (e) {
      console.error(e);
      setResult({
          score: 0,
          issues: ["Audit Process Failed", e instanceof Error ? e.message : "Unknown Error"],
          recommendations: ["Check Settings > API Key"],
          rawAnalysis: "System encountered an error."
      });
    } finally {
      setIsAuditing(false);
    }
  };

  const updateToolCommand = (tool: AuditToolConfig, args: Record<string, string>) => {
      let cmd = tool.command;
      tool.args.forEach(arg => {
          const val = args[arg.name];
          if (val) cmd += ` ${arg.flag} "${val}"`;
      });
      setGeneratedCommand(cmd);
  };

  const handleArgChange = (name: string, value: string) => {
      const newArgs = { ...toolArgs, [name]: value };
      setToolArgs(newArgs);
      updateToolCommand(selectedTool, newArgs);
  };

  const [toolOutput, setToolOutput] = useState<{tool: string, output: string, error?: string} | null>(null);
  const [runningTool, setRunningTool] = useState<string | null>(null);

  const runValidationTool = async (toolId: string) => {
    setRunningTool(toolId);
    setToolOutput(null);
    try {
      const result = await bridgeService.executeTool(toolId, {});
      setToolOutput({
        tool: toolId,
                output: result.output,
                error: result.success ? undefined : (result.error || 'Tool reported failure'),
      });
    } catch (err) {
      setToolOutput({
        tool: toolId,
        output: '',
        error: err instanceof Error ? err.message : 'Unknown error'
      });
    } finally {
      setRunningTool(null);
    }
  };

  return (
    <div className="h-full flex flex-col">
      <div className="mb-6 flex justify-between items-end">
        <div>
            <h2 className="text-2xl font-bold text-white mb-2">Framework Auditor</h2>
            <p className="text-slate-400">Validate artifacts using AI Intelligence or Local Python Tools.</p>
        </div>
        <div className="flex bg-slate-800 rounded-lg p-1 border border-slate-700">
            <button
                onClick={() => setMode('ai')}
                className={`flex items-center gap-2 px-3 py-1.5 rounded text-sm transition-all ${mode === 'ai' ? 'bg-blue-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'}`}
            >
                <Wand2 size={16} /> AI Analysis
            </button>
            <button
                onClick={() => setMode('tool')}
                className={`flex items-center gap-2 px-3 py-1.5 rounded text-sm transition-all ${mode === 'tool' ? 'bg-blue-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'}`}
            >
                <Terminal size={16} /> Tool Runner
            </button>
        </div>
      </div>

      {/* Quick Actions Bar */}
      <div className="bg-slate-800 p-4 rounded-xl border border-slate-700 mb-6">
        <h3 className="text-sm font-semibold text-slate-300 mb-3">Quick Validation</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <button
            onClick={() => runValidationTool('validate')}
            disabled={runningTool !== null}
            className="flex items-center justify-center gap-2 px-4 py-3 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 text-white rounded-lg transition-colors"
          >
            {runningTool === 'validate' ? <Loader2 className="animate-spin" size={16} /> : <CheckCircle size={16} />}
            Validate All
          </button>
          <button
            onClick={() => runValidationTool('compliance')}
            disabled={runningTool !== null}
            className="flex items-center justify-center gap-2 px-4 py-3 bg-purple-600 hover:bg-purple-500 disabled:bg-slate-700 text-white rounded-lg transition-colors"
          >
            {runningTool === 'compliance' ? <Loader2 className="animate-spin" size={16} /> : <ShieldCheck size={16} />}
            Compliance Check
          </button>
          <button
            onClick={() => runValidationTool('boundary')}
            disabled={runningTool !== null}
            className="flex items-center justify-center gap-2 px-4 py-3 bg-green-600 hover:bg-green-500 disabled:bg-slate-700 text-white rounded-lg transition-colors"
          >
            {runningTool === 'boundary' ? <Loader2 className="animate-spin" size={16} /> : <Terminal size={16} />}
            Boundary Check
          </button>
        </div>
        {toolOutput && (
          <div className="mt-4 bg-slate-900 border border-slate-700 rounded-lg p-4 max-h-60 overflow-y-auto">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold text-slate-400">Output: {toolOutput.tool}</span>
              {toolOutput.error && <AlertTriangle size={14} className="text-red-400" />}
            </div>
            <pre className="text-xs text-slate-300 font-mono whitespace-pre-wrap">
              {toolOutput.error || toolOutput.output || '(No output)'}
            </pre>
          </div>
        )}

        {/* Tracking Status Section */}
        <div className="mt-6 bg-slate-800 p-4 rounded-xl border border-slate-700">
          <TrackingStatusComponent />
        </div>
      </div>

      {mode === 'ai' ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-full min-h-0 animate-in fade-in">
            {/* Input Area */}
            <div className="flex flex-col h-full">
            <textarea
                className="flex-1 w-full bg-slate-900 border border-slate-700 rounded-xl p-4 text-slate-300 font-mono text-sm focus:outline-none focus:border-blue-500 resize-none mb-4"
                placeholder="Paste document content here (including Frontmatter)..."
                value={inputContent}
                onChange={(e) => setInputContent(e.target.value)}
            />
            <button
                onClick={handleAiAudit}
                disabled={isAuditing || !inputContent}
                className={`w-full py-3 rounded-lg flex items-center justify-center gap-2 font-semibold transition-all ${
                isAuditing || !inputContent
                    ? 'bg-slate-700 text-slate-400 cursor-not-allowed'
                    : 'bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-900/20'
                }`}
            >
                {isAuditing ? <Loader2 className="animate-spin" /> : <ShieldCheck />}
                {isAuditing ? 'Auditing...' : 'Run Compliance Audit'}
            </button>
            </div>

            {/* Results Area */}
            <div className="bg-slate-800 rounded-xl border border-slate-700 p-6 overflow-y-auto">
            {!result ? (
                <div className="h-full flex flex-col items-center justify-center text-slate-500 opacity-50">
                <ShieldCheck size={64} className="mb-4" />
                <p>Awaiting input for analysis...</p>
                </div>
            ) : (
                <div className="animate-in slide-in-from-bottom-4 duration-500">
                <div className="flex items-center justify-between mb-6">
                    <h3 className="text-lg font-semibold text-white">Audit Report</h3>
                    <div className={`px-4 py-1.5 rounded-full font-bold text-sm ${
                        result.score >= 90 ? 'bg-green-500/20 text-green-400 border border-green-500/50' :
                        result.score >= 70 ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/50' :
                        'bg-red-500/20 text-red-400 border border-red-500/50'
                    }`}>
                        Score: {result.score}/100
                    </div>
                </div>

                <div className="mb-6 p-4 bg-slate-900/50 rounded-lg border border-slate-700/50">
                    <h4 className="text-sm font-semibold text-slate-300 mb-2">Analysis Summary</h4>
                    <p className="text-slate-400 text-sm">{result.rawAnalysis}</p>
                </div>

                <div className="mb-6">
                    <h4 className="text-sm font-semibold text-red-400 mb-3 flex items-center gap-2">
                        <AlertTriangle size={16} /> Issues Detected
                    </h4>
                    {result.issues.length === 0 ? (
                        <p className="text-slate-500 text-sm italic">No critical issues found.</p>
                    ) : (
                        <ul className="space-y-2">
                            {result.issues.map((issue, idx) => (
                                <li key={idx} className="flex items-start gap-2 text-sm text-slate-300 bg-red-500/5 p-2 rounded">
                                    <span className="mt-1 block min-w-[6px] h-1.5 rounded-full bg-red-500"></span>
                                    {issue}
                                </li>
                            ))}
                        </ul>
                    )}
                </div>

                <div>
                    <h4 className="text-sm font-semibold text-green-400 mb-3 flex items-center gap-2">
                        <CheckCircle size={16} /> Recommended Actions
                    </h4>
                    <ul className="space-y-2">
                        {result.recommendations.map((rec, idx) => (
                            <li key={idx} className="flex items-start gap-2 text-sm text-slate-300 bg-green-500/5 p-2 rounded">
                                <ArrowRight size={14} className="mt-1 text-green-500" />
                                {rec}
                            </li>
                        ))}
                    </ul>
                </div>
                </div>
            )}
            </div>
        </div>
      ) : (
          <div className="h-full flex flex-col animate-in fade-in">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 h-full">
                  {/* Tool Selection */}
                  <div className="bg-slate-800 rounded-xl border border-slate-700 p-4">
                      <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4">Available Tools</h3>
                      <div className="space-y-2">
                          {AUDIT_TOOLS.map(tool => (
                              <button
                                key={tool.id}
                                onClick={() => {
                                    setSelectedTool(tool);
                                    setToolArgs({});
                                    setGeneratedCommand(tool.command);
                                }}
                                className={`w-full text-left p-3 rounded-lg border transition-all ${
                                    selectedTool.id === tool.id
                                    ? 'bg-blue-600/20 border-blue-500 text-white'
                                    : 'bg-slate-900 border-slate-800 text-slate-400 hover:bg-slate-800'
                                }`}
                              >
                                  <div className="font-semibold text-sm">{tool.name}</div>
                                  <div className="text-xs opacity-70 mt-1 truncate">{tool.description}</div>
                              </button>
                          ))}
                      </div>
                  </div>

                  {/* Configuration & Output */}
                  <div className="md:col-span-2 bg-slate-800 rounded-xl border border-slate-700 p-6 flex flex-col">
                      <div className="mb-6">
                          <h3 className="text-lg font-bold text-white mb-2">{selectedTool.name}</h3>
                          <p className="text-slate-400 text-sm">{selectedTool.description}</p>
                      </div>

                      <div className="bg-slate-900 rounded-lg p-4 border border-slate-700 mb-6">
                          <h4 className="text-xs font-semibold text-slate-500 uppercase mb-3">Configuration Arguments</h4>
                          {selectedTool.args.length === 0 ? (
                              <p className="text-sm text-slate-600 italic">No arguments required for this tool.</p>
                          ) : (
                              <div className="space-y-3">
                                  {selectedTool.args.map((arg, idx) => (
                                      <div key={idx}>
                                          <label className="block text-xs font-medium text-slate-300 mb-1">
                                              {arg.name} <span className="text-slate-600">({arg.flag})</span>
                                          </label>
                                          <input
                                              type="text"
                                              className="w-full bg-slate-950 border border-slate-800 rounded px-3 py-2 text-sm text-white focus:border-blue-500 focus:outline-none"
                                              placeholder={`Enter ${arg.name}...`}
                                              onChange={(e) => handleArgChange(arg.name, e.target.value)}
                                          />
                                      </div>
                                  ))}
                              </div>
                          )}
                      </div>

                      <div className="mt-auto">
                          <label className="block text-xs font-semibold text-slate-500 uppercase mb-2">Generated Execution Command</label>
                          <div className="flex gap-2">
                              <code className="flex-1 bg-black rounded-lg p-4 font-mono text-green-400 text-sm border border-slate-800">
                                  {generatedCommand}
                              </code>
                              <button
                                onClick={() => {
                                    navigator.clipboard.writeText(generatedCommand);
                                    alert('Command copied! Run this in your terminal.');
                                }}
                                className="bg-slate-700 hover:bg-slate-600 text-white px-4 rounded-lg flex flex-col items-center justify-center gap-1 transition-colors"
                              >
                                  <Terminal size={18} />
                                  <span className="text-xs">Copy</span>
                              </button>
                          </div>
                          <p className="text-xs text-yellow-500/80 mt-2 flex items-center gap-1">
                              <AlertTriangle size={12} />
                              Browser cannot execute Python directly. Copy command to run locally.
                          </p>
                      </div>
                  </div>
              </div>
          </div>
      )}
    </div>
  );
};

export default FrameworkAuditor;
