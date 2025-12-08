
import React, { useState } from 'react';
import { LinkResolver } from './LinkResolver';
import { LinkMigrator } from './LinkMigrator';

export const ReferenceManager: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'RESOLVER' | 'MIGRATOR'>('RESOLVER');

  return (
    <div className="h-full flex flex-col">
      <div className="mb-6">
          <h2 className="text-2xl font-bold text-white mb-2">Reference System</h2>
          <p className="text-slate-400">
            Manage the integrity of your documentation linkage. Resolve Unique Documentation Identifiers (UDIs) or batch migrate legacy file-path links.
          </p>
      </div>

      <div className="flex-1 bg-slate-800 rounded-2xl shadow-sm border border-slate-700 overflow-hidden flex flex-col">
        {/* Tabs */}
        <div className="flex border-b border-slate-700">
          <button
            onClick={() => setActiveTab('RESOLVER')}
            className={`flex-1 py-4 text-sm font-medium text-center transition-colors border-b-2 ${
              activeTab === 'RESOLVER'
                ? 'border-blue-500 text-blue-400 bg-blue-500/5'
                : 'border-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-700'
            }`}
          >
            UDI Resolver & Lookup
          </button>
          <button
            onClick={() => setActiveTab('MIGRATOR')}
            className={`flex-1 py-4 text-sm font-medium text-center transition-colors border-b-2 ${
              activeTab === 'MIGRATOR'
                ? 'border-blue-500 text-blue-400 bg-blue-500/5'
                : 'border-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-700'
            }`}
          >
            Batch Link Migrator
          </button>
        </div>

        <div className="p-6 flex-1 overflow-y-auto">
          {activeTab === 'RESOLVER' ? <LinkResolver /> : <LinkMigrator />}
        </div>
      </div>
    </div>
  );
};
