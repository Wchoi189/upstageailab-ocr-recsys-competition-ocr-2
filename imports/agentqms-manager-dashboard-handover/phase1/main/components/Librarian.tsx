
import React, { useState } from 'react';
import { Plus, CheckCircle, RefreshCw, BookOpen } from 'lucide-react';
import { generateUDI } from '../services/registry';

export const Librarian: React.FC = () => {
  const [lastMinted, setLastMinted] = useState<string | null>(null);
  const [isMinting, setIsMinting] = useState(false);

  const handleMint = () => {
    setIsMinting(true);
    setTimeout(() => {
      setLastMinted(generateUDI());
      setIsMinting(false);
    }, 600);
  };

  return (
    <div className="h-full flex flex-col">
       <div className="mb-6">
        <h2 className="text-2xl font-bold text-white mb-2">The Librarian</h2>
        <p className="text-slate-400">Mint new Unique Documentation Identifiers (UDIs) for your knowledge base.</p>
      </div>

      <div className="flex-1 flex flex-col items-center justify-center space-y-8 animate-in fade-in">
        <div className="bg-slate-800 p-10 rounded-2xl shadow-xl border border-slate-700 text-center space-y-6 max-w-xl w-full">
            <div className="w-24 h-24 bg-slate-900 rounded-full flex items-center justify-center mx-auto mb-4 border border-slate-700">
            <BookOpen className="w-10 h-10 text-blue-500" />
            </div>
            
            <div className="space-y-4">
            <h3 className="text-xl font-semibold text-white">Generate UDI</h3>
            <p className="text-slate-400">Create a unique, immutable identifier for a new document.</p>
            
            <button
                onClick={handleMint}
                disabled={isMinting}
                className={`
                w-full py-4 px-6 rounded-xl text-lg font-semibold text-white shadow-md transition-all
                ${isMinting 
                    ? 'bg-slate-700 cursor-not-allowed' 
                    : 'bg-blue-600 hover:bg-blue-500 hover:shadow-lg hover:shadow-blue-900/20 active:scale-95'
                }
                `}
            >
                {isMinting ? (
                <span className="flex items-center justify-center">
                    <RefreshCw className="w-5 h-5 animate-spin mr-2" />
                    Minting...
                </span>
                ) : (
                <span className="flex items-center justify-center">
                    <Plus className="w-5 h-5 mr-2" />
                    Mint New UDI
                </span>
                )}
            </button>
            </div>

            {lastMinted && (
            <div className="mt-8 p-4 bg-green-500/10 border border-green-500/30 rounded-xl animate-in zoom-in duration-300">
                <div className="flex items-center justify-center space-x-2 text-green-400 mb-2">
                <CheckCircle className="w-5 h-5" />
                <span className="font-medium">Success! New ID Generated</span>
                </div>
                <div 
                    onClick={() => navigator.clipboard.writeText(lastMinted)}
                    className="text-2xl font-mono text-white bg-slate-950 py-3 px-6 rounded-lg border border-slate-700 inline-block shadow-inner cursor-pointer hover:border-green-500/50 transition-colors"
                    title="Click to Copy"
                >
                {lastMinted}
                </div>
                <p className="text-xs text-green-500/80 mt-2">Ready to be injected into frontmatter.</p>
            </div>
            )}
        </div>

        <div className="bg-slate-900/50 p-6 rounded-xl border border-slate-800 max-w-xl w-full">
            <h3 className="font-semibold text-slate-300 mb-3 text-sm uppercase tracking-wider">Protocol Notes</h3>
            <ul className="list-disc list-inside text-sm text-slate-400 space-y-2">
            <li>UDIs are immutable once assigned to a specific document version.</li>
            <li>Ensure the backend "Bridge" is active to persist the new UDI to the filesystem.</li>
            <li>Generated UDIs follow the <code>udi://[type]-[hash]</code> format.</li>
            </ul>
        </div>
      </div>
    </div>
  );
};
