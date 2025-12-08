
import React, { useState, useEffect } from 'react';
import { Copy, RefreshCw, FileText } from 'lucide-react';
import { ArtifactFormData } from '../types';
import { APP_CONFIG } from '../config/constants';

const ArtifactGenerator: React.FC = () => {
  const [formData, setFormData] = useState<ArtifactFormData>({
    title: '',
    type: 'Assessment',
    status: 'draft',
    author: APP_CONFIG.DEFAULTS.AUTHOR,
    branchName: APP_CONFIG.DEFAULTS.BRANCH_NAME,
    tags: 'documentation, quality',
    description: ''
  });

  const [generatedOutput, setGeneratedOutput] = useState('');
  const [timestamp, setTimestamp] = useState('');

  const generateTimestamp = () => {
    const now = new Date();
    const year = now.getFullYear();
    const month = String(now.getMonth() + 1).padStart(2, '0');
    const day = String(now.getDate()).padStart(2, '0');
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    return `${year}-${month}-${day} ${hours}:${minutes} (${APP_CONFIG.DEFAULTS.TIMEZONE})`;
  };

  useEffect(() => {
    setTimestamp(generateTimestamp());
  }, []);

  useEffect(() => {
    const yaml = `---
title: "${formData.title}"
type: ${formData.type}
status: ${formData.status}
author: ${formData.author}
branch_name: ${formData.branchName}
created_at: ${timestamp}
last_updated: ${timestamp}
tags: [${formData.tags}]
description: "${formData.description}"
version: 1.0.0
---

# ${formData.title || 'Untitled Document'}

## Context
<!-- Add context here -->

## Analysis
<!-- Add analysis here -->
`;
    setGeneratedOutput(yaml);
  }, [formData, timestamp]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(generatedOutput);
    alert("Copied to clipboard!");
  };

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-white mb-2">Artifact Generator</h2>
        <p className="text-slate-400">Create standardized documentation with mandatory branch names and timestamps.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-full overflow-y-auto pb-8">
        {/* Form */}
        <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 h-fit">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-1">Title</label>
              <input
                name="title"
                value={formData.title}
                onChange={handleChange}
                className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-white focus:outline-none focus:border-blue-500"
                placeholder="e.g. Audit Framework V1"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-1">Type</label>
              <select
                name="type"
                value={formData.type}
                onChange={handleChange}
                className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-white focus:outline-none focus:border-blue-500"
              >
                <option value="Assessment">Assessment</option>
                <option value="ImplementationPlan">Implementation Plan</option>
                <option value="BugReport">Bug Report</option>
                <option value="ArchitectureDecision">Architecture Decision</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-1">Status</label>
              <select
                name="status"
                value={formData.status}
                onChange={handleChange}
                className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-white focus:outline-none focus:border-blue-500"
              >
                <option value="draft">Draft</option>
                <option value="review">In Review</option>
                <option value="approved">Approved</option>
                <option value="deprecated">Deprecated</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-1">Author</label>
              <input
                name="author"
                value={formData.author}
                onChange={handleChange}
                className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-white focus:outline-none focus:border-blue-500"
                placeholder="GitHub Username"
              />
            </div>
          </div>

          <div className="mb-4">
            <label className="block text-sm font-medium text-blue-400 mb-1 flex items-center gap-2">
              Branch Name (Required)
              <span className="text-xs text-slate-500 bg-slate-900 px-2 py-0.5 rounded">Enforced</span>
            </label>
            <input
              name="branchName"
              value={formData.branchName}
              onChange={handleChange}
              className="w-full bg-slate-900 border border-blue-500/50 rounded px-3 py-2 text-white focus:outline-none focus:border-blue-400"
              placeholder="feature/..."
            />
          </div>

          <div className="mb-4">
             <label className="block text-sm font-medium text-blue-400 mb-1 flex items-center gap-2">
              Timestamp (Auto-generated)
              <span className="text-xs text-slate-500 bg-slate-900 px-2 py-0.5 rounded">Format Enforced</span>
            </label>
             <div className="flex gap-2">
                <input
                  disabled
                  value={timestamp}
                  className="w-full bg-slate-900/50 border border-slate-700 rounded px-3 py-2 text-slate-400 cursor-not-allowed"
                />
                <button
                  onClick={() => setTimestamp(generateTimestamp())}
                  className="bg-slate-700 hover:bg-slate-600 text-white px-3 rounded flex items-center justify-center transition-colors"
                  title="Update Timestamp"
                >
                  <RefreshCw size={18} />
                </button>
             </div>
          </div>

          <div className="mb-4">
            <label className="block text-sm font-medium text-slate-300 mb-1">Description</label>
            <textarea
              name="description"
              value={formData.description}
              onChange={handleChange}
              className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-white focus:outline-none focus:border-blue-500 h-24 resize-none"
              placeholder="Brief summary of the artifact..."
            />
          </div>
           <div className="mb-4">
            <label className="block text-sm font-medium text-slate-300 mb-1">Tags</label>
            <input
              name="tags"
              value={formData.tags}
              onChange={handleChange}
              className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-white focus:outline-none focus:border-blue-500"
              placeholder="Comma separated"
            />
          </div>
        </div>

        {/* Preview */}
        <div className="flex flex-col h-full">
            <div className="bg-slate-950 rounded-t-xl border-t border-x border-slate-700 p-3 flex justify-between items-center">
                <div className="flex items-center gap-2 text-slate-300">
                    <FileText size={16} />
                    <span className="text-sm font-mono">preview.md</span>
                </div>
                <button
                    onClick={copyToClipboard}
                    className="flex items-center gap-2 text-xs bg-blue-600 hover:bg-blue-500 text-white px-3 py-1.5 rounded transition-colors"
                >
                    <Copy size={14} />
                    Copy Code
                </button>
            </div>
            <div className="flex-1 bg-slate-950 border-x border-b border-slate-700 rounded-b-xl p-4 overflow-auto">
                <pre className="font-mono text-sm text-green-400 whitespace-pre-wrap">
                    {generatedOutput}
                </pre>
            </div>
        </div>
      </div>
    </div>
  );
};

export default ArtifactGenerator;
