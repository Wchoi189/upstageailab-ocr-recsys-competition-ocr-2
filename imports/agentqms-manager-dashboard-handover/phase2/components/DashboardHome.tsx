import React, { useEffect, useState } from 'react';
import { AppView } from '../types';
import { Activity, FileCheck, Link, AlertTriangle, RefreshCw } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { getSystemStats, SystemStats } from '../services/registry';

interface DashboardHomeProps {
  onViewChange: (view: AppView) => void;
}

export const DashboardHome: React.FC<DashboardHomeProps> = ({ onViewChange }) => {
  const [stats, setStats] = useState<SystemStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const data = await getSystemStats();
        setStats(data);
      } catch (error) {
        console.error("Failed to fetch system stats", error);
      } finally {
        setLoading(false);
      }
    };
    fetchStats();
  }, []);

  if (loading) {
    return (
      <div className="flex h-[50vh] items-center justify-center flex-col space-y-4">
        <RefreshCw className="w-10 h-10 text-indigo-400 animate-spin" />
        <p className="text-gray-500 font-medium">Connecting to AgentQMS Bridge...</p>
      </div>
    );
  }

  if (!stats) return null;

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 flex items-start space-x-4">
          <div className="p-3 bg-blue-50 rounded-lg">
            <FileCheck className="w-6 h-6 text-blue-600" />
          </div>
          <div>
            <p className="text-sm font-medium text-gray-500">Total Documents</p>
            <h3 className="text-2xl font-bold text-gray-900">{stats.totalDocs}</h3>
            <p className="text-xs text-green-600 mt-1">+{stats.docGrowth} this week</p>
          </div>
        </div>

        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 flex items-start space-x-4">
          <div className="p-3 bg-purple-50 rounded-lg">
            <Link className="w-6 h-6 text-purple-600" />
          </div>
          <div>
            <p className="text-sm font-medium text-gray-500">Reference Health</p>
            <h3 className="text-2xl font-bold text-gray-900">{stats.referenceHealth}%</h3>
            <p className="text-xs text-yellow-600 mt-1">{stats.brokenLinks} broken links</p>
          </div>
        </div>

        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 flex items-start space-x-4">
          <div className="p-3 bg-amber-50 rounded-lg">
            <AlertTriangle className="w-6 h-6 text-amber-600" />
          </div>
          <div>
            <p className="text-sm font-medium text-gray-500">Pending Migrations</p>
            <h3 className="text-2xl font-bold text-gray-900">{stats.pendingMigrations}</h3>
            <p className="text-xs text-gray-400 mt-1">Requires attention</p>
          </div>
        </div>
      </div>

      {/* Main Action Area */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
          <h2 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
            <Activity className="w-5 h-5 mr-2 text-gray-500" />
            System Status
          </h2>
          <div className="h-64">
             <ResponsiveContainer width="100%" height="100%">
                <BarChart data={stats.distribution}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="name" axisLine={false} tickLine={false} />
                  <YAxis axisLine={false} tickLine={false} />
                  <Tooltip cursor={{fill: '#f9fafb'}} contentStyle={{borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'}} />
                  <Bar dataKey="valid" fill="#4f46e5" radius={[4, 4, 0, 0]} name="Passing" />
                  <Bar dataKey="issues" fill="#fbbf24" radius={[4, 4, 0, 0]} name="Needs Review" />
                </BarChart>
             </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-gradient-to-br from-indigo-600 to-purple-700 p-6 rounded-xl shadow-md text-white flex flex-col justify-between">
          <div>
            <h2 className="text-xl font-bold mb-2">Phase 2: Reference Migration</h2>
            <p className="text-indigo-100 mb-6">
              The system is ready to migrate hardcoded file paths to UDI references. 
              Use the Reference Manager to audit and update your documentation links.
            </p>
          </div>
          <button 
            onClick={() => onViewChange(AppView.REFERENCE_MANAGER)}
            className="self-start bg-white text-indigo-600 px-5 py-2 rounded-lg font-medium shadow-sm hover:bg-indigo-50 transition-colors"
          >
            Start Migration Tool
          </button>
        </div>
      </div>
    </div>
  );
};