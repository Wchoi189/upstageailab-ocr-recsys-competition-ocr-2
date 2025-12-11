import React from 'react';
import { 
  LayoutDashboard, 
  Files, 
  ScanLine, 
  FileText, 
  Tags, 
  Settings,
  ChevronRight,
  Bot
} from 'lucide-react';
import { cn } from '../lib/utils';

interface NavItemProps {
  icon: React.ElementType;
  label: string;
  isActive?: boolean;
}

const NavItem: React.FC<NavItemProps> = ({ icon: Icon, label, isActive }) => (
  <button
    className={cn(
      "w-full flex items-center gap-3 px-3 py-2 text-sm font-medium rounded-md transition-colors",
      isActive 
        ? "bg-primary-light text-primary" 
        : "text-gray-600 hover:bg-gray-100 hover:text-gray-900"
    )}
  >
    <Icon size={18} />
    <span>{label}</span>
  </button>
);

const SidebarGroup: React.FC<{ title: string; children: React.ReactNode }> = ({ title, children }) => (
  <div className="mb-6">
    <h3 className="px-3 mb-2 text-xs font-semibold text-gray-500 uppercase tracking-wider">
      {title}
    </h3>
    <div className="space-y-0.5">
      {children}
    </div>
  </div>
);

export const Sidebar: React.FC = () => {
  return (
    <aside className="w-64 border-r border-gray-200 bg-gray-50 flex flex-col h-full flex-shrink-0">
      {/* Logo Area */}
      <div className="h-14 flex items-center px-4 border-b border-gray-200 bg-white">
        <div className="flex items-center gap-2 font-bold text-lg text-gray-900">
          <Bot className="text-primary" />
          <span>Console</span>
        </div>
      </div>

      {/* Navigation */}
      <div className="flex-1 overflow-y-auto py-6 px-3">
        
        <SidebarGroup title="Generate">
          <NavItem icon={Bot} label="Chat" />
        </SidebarGroup>

        <SidebarGroup title="Digitize">
          <NavItem icon={Files} label="Document Parsing" />
          <NavItem icon={ScanLine} label="Document OCR" isActive={true} />
        </SidebarGroup>

        <SidebarGroup title="Extract">
          <NavItem icon={LayoutDashboard} label="Universal Extraction" />
          <NavItem icon={FileText} label="Prebuilt Extraction" />
        </SidebarGroup>

        <SidebarGroup title="Classify">
          <NavItem icon={Tags} label="Document Classification" />
        </SidebarGroup>

      </div>

      {/* Footer / Settings */}
      <div className="p-3 border-t border-gray-200">
        <NavItem icon={Settings} label="Settings" />
      </div>
    </aside>
  );
};