import React, { useState } from 'react';
import { Plus, CheckCircle, RefreshCw } from 'lucide-react';
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
    <div className="max-w-2xl mx-auto space-y-8">
      <div className="text-center space-y-2">
        <h2 className="text-3xl font-bold text-gray-900">The Librarian</h2>
        <p className="text-gray-500">Mint new Unique Documentation Identifiers (UDIs) for your knowledge base.</p>
      </div>

      <div className="bg-white p-8 rounded-2xl shadow-sm border border-gray-200 text-center space-y-6">
        <div className="w-20 h-20 bg-indigo-50 rounded-full flex items-center justify-center mx-auto mb-4">
          <Plus className="w-10 h-10 text-indigo-600" />
        </div>
        
        <div className="space-y-4">
          <button
            onClick={handleMint}
            disabled={isMinting}
            className={`
              w-full py-4 px-6 rounded-xl text-lg font-semibold text-white shadow-md transition-all
              ${isMinting 
                ? 'bg-gray-400 cursor-not-allowed' 
                : 'bg-indigo-600 hover:bg-indigo-700 hover:shadow-lg active:scale-95'
              }
            `}
          >
            {isMinting ? (
              <span className="flex items-center justify-center">
                <RefreshCw className="w-5 h-5 animate-spin mr-2" />
                Minting...
              </span>
            ) : (
              "Mint New UDI"
            )}
          </button>
        </div>

        {lastMinted && (
          <div className="mt-8 p-4 bg-green-50 border border-green-100 rounded-xl animate-fade-in-up">
            <div className="flex items-center justify-center space-x-2 text-green-700 mb-2">
              <CheckCircle className="w-5 h-5" />
              <span className="font-medium">Success! New ID Generated</span>
            </div>
            <div className="text-2xl font-mono text-gray-800 bg-white py-3 px-6 rounded-lg border border-green-200 inline-block shadow-inner select-all">
              {lastMinted}
            </div>
            <p className="text-sm text-green-600 mt-2">Ready to be injected into frontmatter.</p>
          </div>
        )}
      </div>

      <div className="bg-gray-50 p-6 rounded-xl border border-gray-200">
        <h3 className="font-semibold text-gray-800 mb-3">Protocol Notes</h3>
        <ul className="list-disc list-inside text-sm text-gray-600 space-y-2">
          <li>UDIs are immutable once assigned to a specific document version.</li>
          <li>Ensure the backend "Bridge" is active to persist the new UDI to the filesystem.</li>
          <li>Generated UDIs follow the <code>udi://[type]-[hash]</code> format.</li>
        </ul>
      </div>
    </div>
  );
};