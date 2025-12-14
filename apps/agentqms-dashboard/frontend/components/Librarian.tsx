import React, { useState, useEffect } from 'react';
import { BookOpen, Filter, Search, Eye, Calendar, Tag, FileText, Loader2 } from 'lucide-react';
import { bridgeService, Artifact } from '../services/bridgeService';

export const Librarian: React.FC = () => {
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedType, setSelectedType] = useState<string>('');
  const [selectedStatus, setSelectedStatus] = useState<string>('');
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    loadArtifacts();
  }, [selectedType, selectedStatus]);

  const loadArtifacts = async () => {
    setLoading(true);
    setError(null);
    try {
      const params: any = { limit: 100 };
      if (selectedType) params.type = selectedType;
      if (selectedStatus) params.status = selectedStatus;

      const response = await bridgeService.listArtifacts(params);
      setArtifacts(response.items);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load artifacts');
    } finally {
      setLoading(false);
    }
  };

  const filteredArtifacts = artifacts.filter(artifact => {
    if (!searchQuery) return true;
    const query = searchQuery.toLowerCase();
    return (
      artifact.title.toLowerCase().includes(query) ||
      artifact.id.toLowerCase().includes(query) ||
      artifact.type.toLowerCase().includes(query)
    );
  });

  const getTypeColor = (type: string) => {
    const colors: Record<string, string> = {
      implementation_plan: 'bg-blue-100 text-blue-800',
      assessment: 'bg-purple-100 text-purple-800',
      audit: 'bg-yellow-100 text-yellow-800',
      bug_report: 'bg-red-100 text-red-800',
    };
    return colors[type] || 'bg-gray-100 text-gray-800';
  };

  const getStatusColor = (status: string) => {
    const colors: Record<string, string> = {
      draft: 'bg-gray-100 text-gray-600',
      active: 'bg-green-100 text-green-700',
      completed: 'bg-blue-100 text-blue-700',
      archived: 'bg-slate-100 text-slate-600',
    };
    return colors[status] || 'bg-gray-100 text-gray-600';
  };

  return (
    <div className="h-full flex flex-col">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-white mb-2 flex items-center gap-2">
          <BookOpen size={28} />
          The Librarian
        </h2>
        <p className="text-slate-400">Browse and manage artifacts in the repository</p>
      </div>

      {/* Filters */}
      <div className="bg-slate-800 p-4 rounded-xl border border-slate-700 mb-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-500" size={18} />
            <input
              type="text"
              placeholder="Search artifacts..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-slate-900 border border-slate-700 rounded-lg text-slate-300 focus:border-blue-500 focus:outline-none"
            />
          </div>

          <div className="relative">
            <Filter className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-500" size={18} />
            <select
              value={selectedType}
              onChange={(e) => setSelectedType(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-slate-900 border border-slate-700 rounded-lg text-slate-300 focus:border-blue-500 focus:outline-none appearance-none"
            >
              <option value="">All Types</option>
              <option value="implementation_plan">Implementation Plans</option>
              <option value="assessment">Assessments</option>
              <option value="audit">Audits</option>
              <option value="bug_report">Bug Reports</option>
            </select>
          </div>

          <div className="relative">
            <Tag className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-500" size={18} />
            <select
              value={selectedStatus}
              onChange={(e) => setSelectedStatus(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-slate-900 border border-slate-700 rounded-lg text-slate-300 focus:border-blue-500 focus:outline-none appearance-none"
            >
              <option value="">All Statuses</option>
              <option value="draft">Draft</option>
              <option value="active">Active</option>
              <option value="completed">Completed</option>
              <option value="archived">Archived</option>
            </select>
          </div>
        </div>

        <div className="mt-3 text-sm text-slate-400">
          Showing {filteredArtifacts.length} of {artifacts.length} artifacts
        </div>
      </div>

      {/* Artifact List */}
      <div className="flex-1 overflow-y-auto">
        {loading && (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="animate-spin text-blue-500" size={32} />
            <span className="ml-3 text-slate-400">Loading artifacts...</span>
          </div>
        )}

        {error && (
          <div className="bg-red-900/20 border border-red-500/50 rounded-xl p-4 text-red-300">
            <p className="font-semibold">Error loading artifacts</p>
            <p className="text-sm mt-1">{error}</p>
          </div>
        )}

        {!loading && !error && filteredArtifacts.length === 0 && (
          <div className="text-center py-12 text-slate-500">
            <FileText size={48} className="mx-auto mb-4 opacity-50" />
            <p>No artifacts found</p>
          </div>
        )}

        {!loading && !error && filteredArtifacts.length > 0 && (
          <div className="space-y-3">
            {filteredArtifacts.map((artifact) => (
              <div
                key={artifact.id}
                className="bg-slate-800 border border-slate-700 rounded-xl p-4 hover:border-slate-600 transition-colors"
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-2">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${getTypeColor(artifact.type)}`}>
                        {artifact.type.replace('_', ' ')}
                      </span>
                      <span className={`px-2 py-1 rounded text-xs font-medium ${getStatusColor(artifact.status)}`}>
                        {artifact.status}
                      </span>
                    </div>
                    <h3 className="text-lg font-semibold text-white mb-1 truncate">
                      {artifact.title}
                    </h3>
                    <p className="text-sm text-slate-400 font-mono truncate mb-2">
                      {artifact.id}
                    </p>
                    {artifact.created_at && (
                      <div className="flex items-center gap-2 text-xs text-slate-500">
                        <Calendar size={14} />
                        <span>{artifact.created_at}</span>
                      </div>
                    )}
                  </div>
                  <button
                    onClick={() => window.open(`/api/v1/artifacts/${artifact.id}`, '_blank')}
                    className="px-3 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg flex items-center gap-2 text-sm transition-colors"
                  >
                    <Eye size={16} />
                    View
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
