
import React, { useState } from 'react';
import { LayoutDashboard, FilePlus, ShieldAlert, Network, Menu, X, Settings as SettingsIcon, Cable, Telescope, BookOpen, Link2 } from 'lucide-react';
import ArtifactGenerator from './components/ArtifactGenerator';
import FrameworkAuditor from './components/FrameworkAuditor';
import StrategyDashboard from './components/StrategyDashboard';
import IntegrationHub from './components/IntegrationHub';
import ContextExplorer from './components/ContextExplorer';
import Settings from './components/Settings';
import ErrorBoundary from './components/ErrorBoundary';
import { Librarian } from './components/Librarian';
import { ReferenceManager } from './components/ReferenceManager';
import { AppView } from './types';

const App: React.FC = () => {
  const [currentView, setCurrentView] = useState<AppView>(AppView.DASHBOARD);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  const renderContent = () => {
    switch (currentView) {
      case AppView.DASHBOARD:
        return <StrategyDashboard />;
      case AppView.STRATEGY_MAP:
        return <StrategyDashboard />;
      case AppView.ARTIFACT_GENERATOR:
        return <ArtifactGenerator />;
      case AppView.FRAMEWORK_AUDITOR:
        return <FrameworkAuditor />;
      case AppView.INTEGRATION_HUB:
        return <IntegrationHub />;
      case AppView.CONTEXT_EXPLORER:
        return <ContextExplorer />;
      case AppView.LIBRARIAN:
        return <Librarian />;
      case AppView.REFERENCE_MANAGER:
        return <ReferenceManager />;
      case AppView.SETTINGS:
        return <Settings />;
      default:
        return <StrategyDashboard />;
    }
  };

  const NavItem = ({ view, icon: Icon, label }: { view: AppView; icon: any; label: string }) => (
    <button
      onClick={() => setCurrentView(view)}
      className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 ${
        currentView === view
          ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/50'
          : 'text-slate-400 hover:bg-slate-800 hover:text-slate-100'
      }`}
    >
      <Icon size={20} />
      {isSidebarOpen && <span className="font-medium">{label}</span>}
    </button>
  );

  return (
    <div className="flex h-screen bg-slate-950 text-slate-200 font-sans selection:bg-blue-500/30">
      {/* Sidebar */}
      <div
        className={`${
          isSidebarOpen ? 'w-64' : 'w-20'
        } bg-slate-900 border-r border-slate-800 flex flex-col transition-all duration-300 ease-in-out z-20`}
      >
        <div className="p-4 flex items-center justify-between border-b border-slate-800">
          {isSidebarOpen && (
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center font-bold text-white">Q</div>
              <span className="font-bold text-lg tracking-tight">AgentQMS</span>
            </div>
          )}
          <button
            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            className="p-1.5 rounded-md hover:bg-slate-800 text-slate-400 transition-colors"
          >
            {isSidebarOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>

        <nav className="flex-1 p-4 space-y-2 overflow-y-auto">
          <NavItem view={AppView.DASHBOARD} icon={LayoutDashboard} label="Dashboard" />
          <NavItem view={AppView.INTEGRATION_HUB} icon={Cable} label="Integration" />
          
          <div className="py-2">
             <div className={`px-4 text-xs font-bold text-slate-500 uppercase tracking-wider mb-2 ${!isSidebarOpen && 'hidden'}`}>Core Tools</div>
             <NavItem view={AppView.ARTIFACT_GENERATOR} icon={FilePlus} label="Generator" />
             <NavItem view={AppView.FRAMEWORK_AUDITOR} icon={ShieldAlert} label="Auditor" />
          </div>

          <div className="py-2">
             <div className={`px-4 text-xs font-bold text-slate-500 uppercase tracking-wider mb-2 ${!isSidebarOpen && 'hidden'}`}>Registry</div>
             <NavItem view={AppView.LIBRARIAN} icon={BookOpen} label="Librarian" />
             <NavItem view={AppView.REFERENCE_MANAGER} icon={Link2} label="Ref Manager" />
             <NavItem view={AppView.CONTEXT_EXPLORER} icon={Telescope} label="Explorer" />
          </div>
          
        </nav>

        <div className="p-4 border-t border-slate-800">
          <button 
            onClick={() => setCurrentView(AppView.SETTINGS)}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all ${
                currentView === AppView.SETTINGS 
                ? 'bg-blue-600 text-white' 
                : 'text-slate-500 hover:text-slate-300'
            }`}
          >
            <SettingsIcon size={20} />
            {isSidebarOpen && <span className="font-medium text-sm">Settings</span>}
          </button>
        </div>
      </div>

      {/* Main Content */}
      <main className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Header */}
        <header className="h-16 bg-slate-900/50 backdrop-blur-md border-b border-slate-800 flex items-center justify-between px-8">
            <div className="flex items-center gap-4">
               <h1 className="text-xl font-semibold text-white">
                  {currentView === AppView.DASHBOARD && "Dashboard Overview"}
                  {currentView === AppView.ARTIFACT_GENERATOR && "Artifact Generator"}
                  {currentView === AppView.FRAMEWORK_AUDITOR && "Compliance Auditor"}
                  {currentView === AppView.STRATEGY_MAP && "Strategic Architecture"}
                  {currentView === AppView.INTEGRATION_HUB && "Integration Hub"}
                  {currentView === AppView.CONTEXT_EXPLORER && "Context & Traceability"}
                  {currentView === AppView.LIBRARIAN && "The Librarian"}
                  {currentView === AppView.REFERENCE_MANAGER && "Reference Migration"}
                  {currentView === AppView.SETTINGS && "System Settings"}
               </h1>
            </div>
            <div className="flex items-center gap-4">
                <div className="px-3 py-1 bg-blue-500/10 border border-blue-500/20 rounded-full">
                    <span className="text-xs font-medium text-blue-400">Environment: v2.5.0</span>
                </div>
            </div>
        </header>

        {/* Viewport */}
        <div className="flex-1 overflow-hidden p-6 relative">
             <ErrorBoundary key={currentView}>
                {renderContent()}
             </ErrorBoundary>
        </div>
      </main>
    </div>
  );
};

export default App;
