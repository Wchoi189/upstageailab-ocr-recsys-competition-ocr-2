import React, { useState } from 'react';
import { Search, ArrowRight, FileText, AlertCircle } from 'lucide-react';
import { resolveUDI, resolvePath } from '../services/registry';
import { DocEntry } from '../types';

export const LinkResolver: React.FC = () => {
  const [input, setInput] = useState('');
  const [result, setResult] = useState<DocEntry | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleResolve = async () => {
    setError(null);
    setResult(null);

    if (!input) return;

    let found: DocEntry | undefined;
    
    try {
        if (input.startsWith('udi://')) {
        found = await resolveUDI(input);
        } else {
        found = await resolvePath(input);
        }

        if (found) {
        setResult(found);
        } else {
        setError('Resource not found in the current registry.');
        }
    } catch (e) {
        setError('Failed to query registry service.');
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-sm">
        <label className="block text-sm font-medium text-slate-300 mb-2">
          Enter UDI or File Path
        </label>
        <div className="flex space-x-2">
          <div className="relative flex-1">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <Search className="h-5 w-5 text-slate-500" />
            </div>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="e.g., udi://arch-001 or docs/architecture/overview.md"
              className="block w-full bg-slate-900 border border-slate-700 rounded-lg pl-10 pr-3 py-2 text-white focus:ring-blue-500 focus:border-blue-500 sm:text-sm transition-shadow"
              onKeyDown={(e) => e.key === 'Enter' && handleResolve()}
            />
          </div>
          <button
            onClick={handleResolve}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-500 font-medium transition-colors"
          >
            Resolve
          </button>
        </div>
      </div>

      {result && (
        <div className="bg-slate-800 border border-green-500/30 rounded-xl p-6 flex flex-col md:flex-row items-start md:items-center justify-between animate-in fade-in slide-in-from-bottom-2 gap-4">
          <div className="flex items-start space-x-4">
            <div className="p-2 bg-slate-900 rounded-lg border border-slate-700">
               <FileText className="w-8 h-8 text-green-500" />
            </div>
            <div>
              <h4 className="text-lg font-bold text-white">{result.title}</h4>
              <p className="text-slate-400 font-mono text-sm">{result.path}</p>
              <div className="flex items-center space-x-2 mt-2">
                 <span className="px-2 py-0.5 bg-blue-500/20 text-blue-300 text-xs rounded-full border border-blue-500/30">{result.udi}</span>
                 <span className="text-xs text-slate-500">Modified: {result.lastModified}</span>
              </div>
            </div>
          </div>
          <button className="flex items-center text-blue-400 font-medium hover:text-blue-300 text-sm">
            View Source <ArrowRight className="w-4 h-4 ml-1" />
          </button>
        </div>
      )}

      {error && (
        <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-4 flex items-center space-x-3 text-red-400">
          <AlertCircle className="w-5 h-5 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}
      
      <div className="text-xs text-slate-600 mt-4 text-center">
        Try entering <code>udi://arch-001</code> to test resolution.
      </div>
    </div>
  );
};