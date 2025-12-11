
import { Plus } from 'lucide-react';
import { cn } from '../utils';

const Thumbnail = ({ label, isActive = false }: { label: string, isActive?: boolean }) => (
    <div className="group flex flex-col items-center gap-2 cursor-pointer min-w-[100px]">
        <div className={cn(
            "h-24 w-20 bg-white border rounded-md shadow-sm transition-all flex items-center justify-center overflow-hidden relative",
            isActive ? "border-blue-500 ring-1 ring-blue-500" : "border-gray-200 group-hover:border-blue-300"
        )}>
            <div className="w-12 h-16 bg-gray-50 border border-gray-100 rounded-[2px]" />
            {isActive && <div className="absolute inset-0 bg-blue-500/5" />}
        </div>
        <span className={cn(
            "text-xs font-medium",
            isActive ? "text-blue-600" : "text-gray-500 group-hover:text-gray-700"
        )}>{label}</span>
    </div>
);

const AddButton = () => (
    <div className="flex flex-col items-center gap-2 cursor-pointer min-w-[100px]">
        <div className="h-24 w-24 border-2 border-dashed border-gray-300 rounded-md flex items-center justify-center hover:border-blue-400 hover:bg-blue-50 transition-colors">
            <Plus className="text-gray-400" />
        </div>
        <span className="text-xs font-medium text-gray-500">New File</span>
    </div>
);

export const TopRibbon = () => {
    return (
        <div className="w-full h-40 bg-white border-b border-gray-200 flex flex-col">
            <div className="flex items-center justify-between px-6 py-3 border-b border-gray-50">
                <h1 className="text-lg font-semibold text-gray-900">Document OCR</h1>
                <div className="flex gap-2">
                    <button className="text-xs font-medium text-gray-600 bg-gray-100 px-3 py-1.5 rounded-md hover:bg-gray-200">Documentation</button>
                </div>
            </div>
            <div className="flex-1 overflow-x-auto flex items-center px-6 gap-4 no-scrollbar">
                <AddButton />
                <div className="h-20 w-[1px] bg-gray-200 mx-2" />
                <div className="flex items-start gap-4">
                    <div className="flex flex-col gap-2">
                        <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider pl-1">Business</span>
                        <div className="flex gap-4">
                            <Thumbnail label="Report" />
                            <Thumbnail label="Invoice" isActive />
                        </div>
                    </div>
                    <div className="flex flex-col gap-2">
                        <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider pl-1">Finance</span>
                        <div className="flex gap-4">
                            <Thumbnail label="Receipt" />
                            <Thumbnail label="Statement" />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};
