import React, { ReactNode } from 'react';
import { BookOpen, ShieldCheck, LayoutDashboard, Settings, FileText } from 'lucide-react';
import { AppView } from '../types';

interface LayoutProps {
  children: ReactNode;
  currentView: AppView;
  onViewChange: (view: AppView) => void;
}

export const Layout: React.FC<LayoutProps> = ({ children, currentView, onViewChange }) => {
  const navItems = [
    { id: AppView.DASHBOARD, label: 'Dashboard', icon: LayoutDashboard },
    { id: AppView.LIBRARIAN, label: 'Librarian', icon: BookOpen },
    { id: AppView.REFERENCE_MANAGER, label: 'Ref Manager', icon: FileText },
  ];

  return (
    <div className="flex h-screen bg-gray-50 text-gray-900 overflow-hidden">
      {/* Sidebar */}
      <aside className="w-64 bg-white border-r border-gray-200 flex flex-col flex-shrink-0">
        <div className="p-6 border-b border-gray-100 flex items-center space-x-3">
          <div className="bg-indigo-600 p-2 rounded-lg">
            <ShieldCheck className="w-6 h-6 text-white" />
          </div>
          <span className="text-xl font-bold text-gray-800 tracking-tight">AgentQMS</span>
        </div>

        <nav className="flex-1 p-4 space-y-1 overflow-y-auto">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = currentView === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onViewChange(item.id)}
                className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-indigo-50 text-indigo-700'
                    : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                }`}
              >
                <Icon className={`w-5 h-5 ${isActive ? 'text-indigo-600' : 'text-gray-400'}`} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="p-4 border-t border-gray-100">
          <button className="w-full flex items-center space-x-3 px-4 py-3 rounded-lg text-sm font-medium text-gray-600 hover:bg-gray-50 transition-colors">
            <Settings className="w-5 h-5 text-gray-400" />
            <span>Settings</span>
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Top Bar */}
        <header className="bg-white border-b border-gray-200 px-8 py-4 flex justify-between items-center shadow-sm z-10">
          <h1 className="text-2xl font-semibold text-gray-800">
            {currentView === AppView.DASHBOARD && 'System Overview'}
            {currentView === AppView.LIBRARIAN && 'The Librarian'}
            {currentView === AppView.REFERENCE_MANAGER && 'Reference System Migration'}
          </h1>
          <div className="flex items-center space-x-4">
             <div className="px-3 py-1 bg-green-100 text-green-700 text-xs font-semibold rounded-full border border-green-200">
                Bridge: Active
             </div>
             <div className="w-8 h-8 bg-indigo-100 rounded-full flex items-center justify-center text-indigo-700 font-bold border border-indigo-200">
               JS
             </div>
          </div>
        </header>

        {/* Scrollable Content Area */}
        <div className="flex-1 overflow-y-auto p-8 bg-gray-50/50">
          <div className="max-w-6xl mx-auto">
            {children}
          </div>
        </div>
      </main>
    </div>
  );
};