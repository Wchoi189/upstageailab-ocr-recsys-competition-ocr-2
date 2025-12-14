
import { Home, FileText, Settings, Database, UploadCloud } from 'lucide-react';
import { cn } from '../utils';
import { CheckpointSelector } from './CheckpointSelector';

const NavItem = ({ icon: Icon, label, isActive = false }: { icon: any, label: string, isActive?: boolean }) => (
    <button className={cn(
        "flex items-center gap-3 w-full px-4 py-2 text-sm font-medium rounded-md transition-colors",
        isActive
            ? "bg-blue-50 text-blue-700"
            : "text-gray-600 hover:bg-gray-50 hover:text-gray-900"
    )}>
        <Icon size={18} />
        {label}
    </button>
);

interface SidebarProps {
    selectedCheckpoint: string | null;
    onCheckpointChange: (checkpoint: string) => void;
    enablePerspectiveCorrection: boolean;
    onPerspectiveCorrectionChange: (enabled: boolean) => void;
    displayMode: string;
    onDisplayModeChange: (mode: string) => void;
}

export const Sidebar = ({
    selectedCheckpoint,
    onCheckpointChange,
    enablePerspectiveCorrection,
    onPerspectiveCorrectionChange,
    displayMode,
    onDisplayModeChange
}: SidebarProps) => {
    return (
        <div className="w-64 border-r border-gray-200 h-full bg-white flex flex-col">
            <div className="p-4 border-b border-gray-100 flex items-center gap-2">
                <div className="h-6 w-6 bg-blue-600 rounded-md flex items-center justify-center text-white font-bold">U</div>
                <span className="font-semibold text-gray-900 tracking-tight">Inference Console</span>
            </div>

            <div className="flex-1 overflow-y-auto py-4 px-2 space-y-6">
                <div>
                    <h4 className="px-4 text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Generate</h4>
                    <div className="space-y-1">
                        <NavItem icon={Home} label="Chat" />
                    </div>
                </div>

                <div>
                    <h4 className="px-4 text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Digitize</h4>
                    <div className="space-y-1">
                        <NavItem icon={FileText} label="Document Parsing" />
                        <NavItem icon={UploadCloud} label="Document OCR" isActive />
                    </div>
                </div>

                <div>
                    <h4 className="px-4 text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Extract</h4>
                    <div className="space-y-1">
                        <NavItem icon={Database} label="Universal Extraction" />
                    </div>
                </div>

                <div className="pt-4 border-t border-gray-100">
                    <CheckpointSelector
                        selectedCheckpoint={selectedCheckpoint}
                        onCheckpointChange={onCheckpointChange}
                    />
                </div>

                <div className="pt-4 border-t border-gray-100">
                    <h4 className="px-4 text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
                        Perspective Correction
                    </h4>
                    <div className="px-4 space-y-2">
                        <label className="flex items-center gap-2 text-sm">
                            <input
                                type="checkbox"
                                checked={enablePerspectiveCorrection}
                                onChange={(e) => onPerspectiveCorrectionChange(e.target.checked)}
                                className="rounded border-gray-300"
                            />
                            <span>Enable Correction</span>
                        </label>
                        {enablePerspectiveCorrection && (
                            <div className="ml-6 space-y-1">
                                <label className="flex items-center gap-2 text-xs">
                                    <input
                                        type="radio"
                                        name="displayMode"
                                        value="corrected"
                                        checked={displayMode === "corrected"}
                                        onChange={(e) => onDisplayModeChange(e.target.value)}
                                    />
                                    <span>Show Corrected</span>
                                </label>
                                <label className="flex items-center gap-2 text-xs">
                                    <input
                                        type="radio"
                                        name="displayMode"
                                        value="original"
                                        checked={displayMode === "original"}
                                        onChange={(e) => onDisplayModeChange(e.target.value)}
                                    />
                                    <span>Show Original</span>
                                </label>
                            </div>
                        )}
                    </div>
                </div>
            </div>

            <div className="p-4 border-t border-gray-100">
                <NavItem icon={Settings} label="Settings" />
            </div>
        </div>
    );
};
