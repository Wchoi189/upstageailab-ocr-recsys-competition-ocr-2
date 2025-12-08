import React from 'react';
import { Upload, FileText, Zap } from 'lucide-react';

interface InputSectionProps {
  input: string;
  setInput: (val: string) => void;
  onAnalyze: () => void;
  isAnalyzing: boolean;
}

export const InputSection: React.FC<InputSectionProps> = ({ input, setInput, onAnalyze, isAnalyzing }) => {
  return (
    <div className="flex flex-col h-full bg-slate-900 border-r border-slate-800">
      <div className="p-6 border-b border-slate-800 flex justify-between items-center bg-slate-950">
        <div>
          <h2 className="text-xl font-semibold text-white flex items-center gap-2">
            <FileText className="w-5 h-5 text-indigo-400" />
            Session Context
          </h2>
          <p className="text-slate-400 text-sm mt-1">Paste your brain dump, logs, or messy notes here.</p>
        </div>
      </div>
      
      <div className="flex-1 p-4 relative group">
        <textarea
          className="w-full h-full bg-slate-900 text-slate-300 p-4 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-indigo-500/20 placeholder-slate-600 transition-all font-mono text-sm leading-relaxed"
          placeholder="e.g. 'I was working on the login component but got stuck on the auth token refresh logic. I have a 401 error in the console. Here is the error log...'"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          spellCheck={false}
        />
        {input.length === 0 && (
          <div className="absolute inset-0 pointer-events-none flex items-center justify-center opacity-30">
            <div className="text-center">
              <Upload className="w-12 h-12 mx-auto mb-3 text-slate-500" />
              <p className="text-slate-500">Paste content to begin</p>
            </div>
          </div>
        )}
      </div>

      <div className="p-6 border-t border-slate-800 bg-slate-950">
        <button
          onClick={onAnalyze}
          disabled={isAnalyzing || !input.trim()}
          className={`w-full py-4 px-6 rounded-xl font-medium text-white shadow-lg flex items-center justify-center gap-2 transition-all transform active:scale-95
            ${isAnalyzing || !input.trim() 
              ? 'bg-slate-800 text-slate-500 cursor-not-allowed' 
              : 'bg-indigo-600 hover:bg-indigo-500 hover:shadow-indigo-500/20'
            }`}
        >
          {isAnalyzing ? (
            <>
              <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              Analyzing Session...
            </>
          ) : (
            <>
              <Zap className="w-5 h-5 fill-current" />
              Analyze & Recover
            </>
          )}
        </button>
      </div>
    </div>
  );
};