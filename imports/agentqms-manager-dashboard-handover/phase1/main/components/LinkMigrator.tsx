
import React, { useState } from 'react';
import { ArrowRightLeft, Sparkles, AlertTriangle, FileCode, Eye, Code, Save, Check, Loader } from 'lucide-react';
import { resolvePath, commitMigration } from '../services/registry';
import { analyzeLinkRelevance } from '../services/aiService';
import { LinkAnalysis } from '../types';

export const LinkMigrator: React.FC = () => {
  const [content, setContent] = useState('');
  const [migratedContent, setMigratedContent] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [isCommitting, setIsCommitting] = useState(false);
  const [commitStatus, setCommitStatus] = useState<{success: boolean; message: string} | null>(null);
  const [analysis, setAnalysis] = useState<LinkAnalysis[]>([]);
  const [aiAnalyzing, setAiAnalyzing] = useState(false);
  const [viewMode, setViewMode] = useState<'RAW' | 'DIFF'>('DIFF');

  // Regex to find markdown links: [text](url)
  const linkRegex = /\[([^\]]+)\]\(([^)]+)\)/g;

  const handleMigration = () => {
    setIsProcessing(true);
    setAnalysis([]);
    setMigratedContent('');
    setCommitStatus(null);
    
    // Simulate processing delay
    setTimeout(() => {
      const newAnalysis: LinkAnalysis[] = [];
      
      const newContent = content.replace(linkRegex, (match, text, url) => {
        // Skip if already UDI or external
        if (url.startsWith('udi://') || url.startsWith('http')) {
          newAnalysis.push({
            original: match,
            type: url.startsWith('udi://') ? 'UDI' : 'EXTERNAL',
            isValid: true,
          });
          return match;
        }

        // Try to resolve path
        // Removing relative prefixes for mock matching
        const cleanUrl = url.replace(/^\.\//, '').replace(/^\.\.\//, '');
        const doc = resolvePath(cleanUrl);

        if (doc) {
          newAnalysis.push({
            original: match,
            type: 'FILE_PATH',
            resolved: doc.udi,
            isValid: true,
            suggestion: doc.contentSnippet // Pass snippet for AI auditing later
          });
          return `[${text}](${doc.udi})`;
        } else {
           newAnalysis.push({
            original: match,
            type: 'FILE_PATH',
            isValid: false,
          });
          return match;
        }
      });

      setMigratedContent(newContent);
      setAnalysis(newAnalysis);
      setIsProcessing(false);
    }, 600);
  };

  const handleCommit = async () => {
    if (!migratedContent) return;
    setIsCommitting(true);
    try {
        // In a real scenario, we'd pass the actual file path we are editing
        const result = await commitMigration('docs/current-file.md', migratedContent);
        setCommitStatus(result);
    } catch (e) {
        setCommitStatus({ success: false, message: 'Failed to commit to bridge.' });
    } finally {
        setIsCommitting(false);
    }
  };

  const runAiAudit = async () => {
    if (analysis.length === 0) return;
    setAiAnalyzing(true);
    
    // Create a copy to update
    const updatedAnalysis = [...analysis];
    
    // Process only valid replacements sequentially for demo purposes
    for (let i = 0; i < updatedAnalysis.length; i++) {
        const item = updatedAnalysis[i];
        if (item.resolved && item.suggestion) {
            const linkTextMatch = item.original.match(/\[([^\]]+)\]/);
            const linkText = linkTextMatch ? linkTextMatch[1] : "Link";
            
            const auditResult = await analyzeLinkRelevance(linkText, item.suggestion);
            updatedAnalysis[i] = { ...item, contextAnalysis: auditResult };
            setAnalysis([...updatedAnalysis]); 
        }
    }
    setAiAnalyzing(false);
  };

  const renderVisualDiff = () => {
    if (!migratedContent) return null;

    // This is a simple visualizer that splits the original content by regex matches
    // effectively reconstructing the document with highlighted changes.
    const parts = [];
    let lastIndex = 0;
    
    // Reset regex state
    const regex = /\[([^\]]+)\]\(([^)]+)\)/g;
    let match;

    while ((match = regex.exec(content)) !== null) {
      // Push text before the match
      if (match.index > lastIndex) {
        parts.push(<span key={`text-${lastIndex}`}>{content.substring(lastIndex, match.index)}</span>);
      }

      const originalMatch = match[0];
      const analysisItem = analysis.find(a => a.original === originalMatch);

      if (analysisItem && analysisItem.resolved) {
        // Render Diff
        parts.push(
            <span key={`diff-${match.index}`} className="inline-flex flex-col align-top mx-1">
                <span className="bg-red-500/20 text-red-400 line-through text-[10px] px-1 rounded-sm">{originalMatch}</span>
                <span className="bg-green-500/20 text-green-400 font-bold px-1 rounded-sm">[{match[1]}]({analysisItem.resolved})</span>
            </span>
        );
      } else {
        // No change
        parts.push(<span key={`same-${match.index}`}>{originalMatch}</span>);
      }
      
      lastIndex = regex.lastIndex;
    }

    // Push remaining text
    if (lastIndex < content.length) {
      parts.push(<span key={`end-${lastIndex}`}>{content.substring(lastIndex)}</span>);
    }

    return (
        <div className="p-4 bg-slate-900 font-mono text-sm text-slate-300 whitespace-pre-wrap leading-relaxed">
            {parts}
        </div>
    );
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-[calc(100vh-250px)]">
      {/* Input Column */}
      <div className="flex flex-col space-y-4">
        <div className="flex justify-between items-center">
            <label className="text-sm font-semibold text-slate-400">Source Markdown (File Paths)</label>
            <button 
                onClick={() => setContent(`Check the [Architecture Doc](docs/architecture/overview.md) and the [API Reference](docs/api/endpoints.md).\n\nAlso see [Invalid Link](docs/missing.md).`)}
                className="text-xs text-blue-400 hover:text-blue-300 underline"
            >
                Load Sample
            </button>
        </div>
        <textarea
          value={content}
          onChange={(e) => setContent(e.target.value)}
          className="flex-1 p-4 bg-slate-900 border border-slate-700 rounded-xl font-mono text-sm text-slate-300 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none placeholder-slate-600"
          placeholder="Paste markdown content here..."
        />
        <button
          onClick={handleMigration}
          disabled={!content || isProcessing}
          className="w-full py-3 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 text-white rounded-xl font-medium flex items-center justify-center transition-colors"
        >
          {isProcessing ? <ArrowRightLeft className="animate-spin mr-2" /> : <ArrowRightLeft className="mr-2" />}
          Analyze & Migrate Links
        </button>
      </div>

      {/* Output Column */}
      <div className="flex flex-col space-y-4">
        <div className="flex justify-between items-center">
            <div className="flex items-center space-x-3">
                <label className="text-sm font-semibold text-slate-400">Output Preview</label>
                {migratedContent && (
                    <div className="flex bg-slate-800 rounded-lg p-0.5 border border-slate-700">
                        <button
                            onClick={() => setViewMode('DIFF')}
                            className={`px-2 py-0.5 text-xs font-medium rounded-md transition-all ${viewMode === 'DIFF' ? 'bg-slate-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'}`}
                        >
                            <span className="flex items-center"><Eye className="w-3 h-3 mr-1"/> Diff</span>
                        </button>
                        <button
                            onClick={() => setViewMode('RAW')}
                            className={`px-2 py-0.5 text-xs font-medium rounded-md transition-all ${viewMode === 'RAW' ? 'bg-slate-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'}`}
                        >
                             <span className="flex items-center"><Code className="w-3 h-3 mr-1"/> Raw</span>
                        </button>
                    </div>
                )}
            </div>
             <div className="flex space-x-2">
                {analysis.some(a => a.resolved) && (
                    <button 
                    onClick={runAiAudit}
                    disabled={aiAnalyzing}
                    className="flex items-center text-xs bg-purple-500/20 text-purple-400 px-3 py-1.5 rounded-lg hover:bg-purple-500/30 border border-purple-500/30 transition-colors disabled:opacity-50 font-medium"
                    >
                        <Sparkles className="w-3 h-3 mr-1" />
                        {aiAnalyzing ? "Auditing..." : "AI Audit"}
                    </button>
                )}
                {migratedContent && (
                     <button 
                     onClick={handleCommit}
                     disabled={isCommitting || commitStatus?.success}
                     className={`flex items-center text-xs px-3 py-1.5 rounded-lg transition-colors font-medium text-white
                        ${commitStatus?.success ? 'bg-green-600' : 'bg-slate-700 hover:bg-slate-600'}
                     `}
                   >
                      {isCommitting ? (
                          <>
                           <Loader className="w-3 h-3 mr-1 animate-spin" />
                           Saving...
                          </>
                      ) : commitStatus?.success ? (
                          <>
                           <Check className="w-3 h-3 mr-1" />
                           Saved
                          </>
                      ) : (
                          <>
                          <Save className="w-3 h-3 mr-1" />
                          Commit to Disk
                          </>
                      )}
                   </button>
                )}
             </div>
        </div>
        
        {commitStatus && (
            <div className={`text-xs px-4 py-2 rounded-lg flex items-center ${commitStatus.success ? 'bg-green-500/10 text-green-400 border border-green-500/20' : 'bg-red-500/10 text-red-400 border border-red-500/20'}`}>
                {commitStatus.success ? <Check className="w-3 h-3 mr-2"/> : <AlertTriangle className="w-3 h-3 mr-2"/>}
                {commitStatus.message}
            </div>
        )}

        <div className="flex-1 border border-slate-700 bg-slate-900 rounded-xl overflow-hidden flex flex-col">
            {migratedContent ? (
                <>
                {viewMode === 'RAW' ? (
                     <textarea
                        readOnly
                        value={migratedContent}
                        className="flex-1 p-4 bg-slate-900 font-mono text-sm text-slate-300 resize-none outline-none"
                    />
                ) : (
                    <div className="flex-1 overflow-y-auto bg-slate-900 border-b border-slate-800">
                        {renderVisualDiff()}
                    </div>
                )}
               
                {/* Analysis Report */}
                <div className="bg-slate-800 border-t border-slate-700 p-4 h-1/3 overflow-y-auto">
                    <h4 className="text-xs font-bold text-slate-500 uppercase mb-2">Change Log</h4>
                    <div className="space-y-2">
                        {analysis.map((item, idx) => (
                            <div key={idx} className="text-xs p-2 rounded border border-slate-700 bg-slate-900/50 flex flex-col gap-1">
                                <div className="flex items-center justify-between">
                                    <code className="bg-slate-800 px-1 rounded text-slate-400 truncate max-w-[150px]">{item.original}</code>
                                    {item.resolved ? (
                                        <span className="flex items-center text-green-400 font-medium">
                                            <ArrowRightLeft className="w-3 h-3 mr-1" /> Replaced
                                        </span>
                                    ) : item.type === 'UDI' ? (
                                        <span className="text-slate-500">Ignored (Already UDI)</span>
                                    ) : (
                                        <span className="flex items-center text-amber-500">
                                            <AlertTriangle className="w-3 h-3 mr-1" /> Unresolved
                                        </span>
                                    )}
                                </div>
                                {item.contextAnalysis && (
                                    <div className="flex items-start gap-2 mt-1 bg-purple-500/10 p-2 rounded border border-purple-500/20">
                                        <Sparkles className="w-3 h-3 text-purple-400 mt-0.5 flex-shrink-0" />
                                        <p className="text-purple-300">{item.contextAnalysis}</p>
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                </div>
                </>
            ) : (
                <div className="flex-1 flex flex-col items-center justify-center text-slate-600">
                    <FileCode className="w-12 h-12 mb-2 opacity-20" />
                    <p>Load content to preview migration</p>
                </div>
            )}
        </div>
      </div>
    </div>
  );
};
