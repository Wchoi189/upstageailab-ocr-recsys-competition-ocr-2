import React from 'react';
import { Plus, FileText, FileImage, FileSpreadsheet } from 'lucide-react';
import { cn } from '../lib/utils';
import { Button } from './ui/button';

interface ThumbnailProps {
  title: string;
  type: string;
  isActive?: boolean;
  imageUrl?: string;
}

const ThumbnailCard: React.FC<ThumbnailProps> = ({ title, type, isActive, imageUrl }) => (
  <div 
    className={cn(
      "group relative flex flex-col w-24 flex-shrink-0 cursor-pointer rounded-md border transition-all duration-200",
      isActive 
        ? "border-primary ring-1 ring-primary bg-primary-light/10" 
        : "border-gray-200 hover:border-gray-300 bg-white"
    )}
  >
    <div className="h-20 w-full bg-gray-100 rounded-t-md overflow-hidden flex items-center justify-center">
        {imageUrl ? (
            <img src={imageUrl} alt={title} className="h-full w-full object-cover opacity-80 group-hover:opacity-100 transition-opacity" />
        ) : (
            <FileText className="text-gray-400" size={24} />
        )}
    </div>
    <div className="p-2 border-t border-gray-100">
      <p className="text-xs font-medium text-gray-700 truncate">{type}</p>
      <p className="text-[10px] text-gray-500 truncate">{title}</p>
    </div>
    {isActive && (
        <div className="absolute top-1 right-1 h-2 w-2 rounded-full bg-primary" />
    )}
  </div>
);

export const TopRibbon: React.FC = () => {
  return (
    <header className="h-auto bg-white border-b border-gray-200 flex flex-col flex-shrink-0">
      {/* Breadcrumbs / Top Bar */}
      <div className="h-14 flex items-center justify-between px-6 border-b border-gray-100">
        <div className="flex items-center gap-2">
            <h1 className="text-lg font-semibold text-gray-900">Document OCR</h1>
            <span className="text-gray-300">/</span>
            <span className="text-sm text-gray-500 font-medium">Playground</span>
        </div>
        <div className="flex items-center gap-3">
            <Button variant="outline" size="sm">Documentation</Button>
            <Button variant="secondary" size="sm">API Reference</Button>
        </div>
      </div>

      {/* Thumbnail Scroll Area */}
      <div className="w-full overflow-x-auto py-4 px-6">
        <div className="flex items-start gap-4">
            
            {/* Add New Button */}
            <div className="flex flex-col items-center gap-2">
                 <button className="h-28 w-24 border-2 border-dashed border-gray-300 rounded-lg flex flex-col items-center justify-center text-gray-400 hover:border-primary hover:text-primary transition-colors bg-gray-50 hover:bg-white">
                    <Plus size={24} />
                    <span className="text-xs font-medium mt-1">Upload</span>
                </button>
            </div>

            <div className="w-[1px] h-28 bg-gray-200 mx-2" />

            {/* Thumbnails */}
            <div className="flex gap-4">
                <ThumbnailCard 
                    title="INV-2023-001" 
                    type="Invoice" 
                    imageUrl="https://picsum.photos/100/120?random=1" 
                />
                <ThumbnailCard 
                    title="Biz_Card_04" 
                    type="Card" 
                    isActive={true}
                    imageUrl="https://picsum.photos/100/120?random=2" 
                />
                <ThumbnailCard 
                    title="Q3_Report" 
                    type="Report" 
                    imageUrl="https://picsum.photos/100/120?random=3" 
                />
                 <ThumbnailCard 
                    title="Contract_v2" 
                    type="Legal" 
                    imageUrl="https://picsum.photos/100/120?random=4" 
                />
                 <ThumbnailCard 
                    title="Receipt_Coffee" 
                    type="Receipt" 
                    imageUrl="https://picsum.photos/100/120?random=5" 
                />
            </div>
        </div>
      </div>
    </header>
  );
};