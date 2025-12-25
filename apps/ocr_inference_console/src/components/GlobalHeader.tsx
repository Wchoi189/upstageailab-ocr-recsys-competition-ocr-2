
import { Search, Grid } from 'lucide-react';

export const GlobalHeader = () => {
    return (
        <div className="h-14 w-full bg-white flex items-center px-0 shrink-0 z-50 relative">
            {/* Left: Logo/Branding - Fixed width to match Sidebar */}
            <div className="w-60 flex items-center gap-2 pl-4 shrink-0 border-r border-transparent">
                <span className="text-base font-bold tracking-tight">inference</span>
                <span className="text-base font-light text-gray-700">Console</span>
            </div>

            {/* Center: Navigation - Starts after sidebar */}
            <nav className="flex items-center gap-8 text-base font-medium text-gray-500 ml-4">
                <a href="#" className="hover:text-blue-600 transition-colors">Documentation</a>
                <a href="#" className="hover:text-blue-600 transition-colors">API Reference</a>
                <a href="#" className="text-blue-600 border-b-2 border-blue-600 h-14 flex items-center">Playground</a>
                <a href="#" className="hover:text-blue-600 transition-colors">Dashboard</a>
            </nav>

            {/* Right: Search & Actions */}
            <div className="flex items-center gap-3 ml-auto pr-4">
                <div className="relative hidden md:block">
                    <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 text-gray-400" size={16} />
                    <input
                        type="text"
                        placeholder="Search"
                        className="h-9 pl-9 pr-4 bg-gray-100/50 border-gray-200 rounded-md text-sm w-48 lg:w-64 focus:bg-white focus:border-blue-500 transition-all outline-none border"
                    />
                </div>
                <button className="p-2 text-gray-500 hover:bg-gray-100 rounded-md">
                    <Grid size={20} />
                </button>
                <button className="h-9 px-4 bg-blue-600 text-white text-sm font-medium rounded-md hover:bg-blue-700 transition-colors shadow-sm">
                    Log in
                </button>
            </div>
        </div>
    );
};
