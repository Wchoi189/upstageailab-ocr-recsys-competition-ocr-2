
import { Plus } from 'lucide-react';
import { cn } from '../utils';

const Thumbnail = ({ label, isActive = false }: { label: string, isActive?: boolean }) => (
    <button className={cn(
        "flex flex-col items-center justify-center self-end mb-2.5 w-[60px] h-[68px] bg-white border rounded-md transition-all overflow-hidden relative p-1 cursor-pointer",
        isActive ? "border-blue-500 border-2" : "border-gray-300 hover:border-blue-300"
    )}>
        <div className="w-full h-10 bg-gray-50 border border-gray-100 rounded-[2px] mb-1" />
        <span className={cn(
            "text-xs font-medium truncate w-full text-center",
            isActive ? "text-blue-600" : "text-gray-500"
        )}>{label}</span>
    </button>
);

const AddButton = ({ onClick }: { onClick?: () => void }) => (
    <button
        className="flex items-center justify-center self-end mb-2.5 w-[60px] h-[68px] border-[1.5px] border-dashed border-gray-300 rounded-md hover:border-blue-400 hover:bg-blue-50/50 transition-colors cursor-pointer bg-white"
        onClick={onClick}
    >
        <Plus className="text-gray-400" size={24} />
    </button>
);

export const TopRibbon = ({ onUploadClick }: { onUploadClick?: () => void }) => {
    return (
        <div className="w-full bg-white border-b border-gray-200">
            {/* Thumbnails Section */}
            <div className="h-[130px] overflow-x-hidden flex items-start px-6 gap-2 pt-1 pb-3">
                <AddButton onClick={onUploadClick} />
                <div className="h-[68px] w-[1px] bg-gray-200 self-end mb-2.5" />
                <div className="flex items-center gap-2 self-end">
                    <div className="flex gap-2">
                        <div className="h-[68px] w-[1px] bg-gray-200 self-end mb-2.5" />
                        <div className="flex flex-col gap-1">
                            <span className="text-[10px] font-medium text-gray-400 uppercase tracking-wider">Business</span>
                            <div className="flex gap-2">
                                <Thumbnail label="Report" />
                                <Thumbnail label="Invoice" isActive />
                            </div>
                        </div>
                    </div>
                    <div className="flex gap-2">
                        <div className="h-[68px] w-[1px] bg-gray-200 self-end mb-2.5" />
                        <div className="flex flex-col gap-1">
                            <span className="text-[10px] font-medium text-gray-400 uppercase tracking-wider">Finance</span>
                            <div className="flex gap-2">
                                <Thumbnail label="Receipt" />
                                <Thumbnail label="Statement" />
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};
