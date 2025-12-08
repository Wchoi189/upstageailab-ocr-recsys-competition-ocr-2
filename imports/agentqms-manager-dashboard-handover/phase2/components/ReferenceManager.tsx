import React, { useState } from 'react';
import { LinkResolver } from './LinkResolver';
import { LinkMigrator } from './LinkMigrator';

export const ReferenceManager: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'RESOLVER' | 'MIGRATOR'>('RESOLVER');

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
          <p className="text-gray-500 max-w-3xl">
            Manage the integrity of your documentation linkage. Resolve Unique Documentation Identifiers (UDIs) or batch migrate legacy file-path links to the new standard.
          </p>
      </div>

      <div className="bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden min-h-[600px] flex flex-col">
        {/* Tabs */}
        <div className="flex border-b border-gray-200">
          <button
            onClick={() => setActiveTab('RESOLVER')}
            className={`flex-1 py-4 text-sm font-medium text-center transition-colors border-b-2 ${
              activeTab === 'RESOLVER'
                ? 'border-indigo-600 text-indigo-600 bg-indigo-50/50'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:bg-gray-50'
            }`}
          >
            UDI Resolver & Lookup
          </button>
          <button
            onClick={() => setActiveTab('MIGRATOR')}
            className={`flex-1 py-4 text-sm font-medium text-center transition-colors border-b-2 ${
              activeTab === 'MIGRATOR'
                ? 'border-indigo-600 text-indigo-600 bg-indigo-50/50'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:bg-gray-50'
            }`}
          >
            Batch Link Migrator
          </button>
        </div>

        <div className="p-6 flex-1">
          {activeTab === 'RESOLVER' ? <LinkResolver /> : <LinkMigrator />}
        </div>
      </div>
    </div>
  );
};