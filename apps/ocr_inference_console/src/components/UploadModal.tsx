import { useRef, useState, useEffect } from 'react';
import { X, Upload } from 'lucide-react';
import { type Checkpoint } from '../api/ocrClient';

interface UploadModalProps {
    isOpen: boolean;
    onClose: () => void;
    onFileSelected: (file: File, checkpoint: string) => void;
    initialCheckpoint: string | null;
    checkpoints: Checkpoint[];
    loading: boolean;
}

export const UploadModal: React.FC<UploadModalProps> = ({
    isOpen,
    onClose,
    onFileSelected,
    initialCheckpoint,
    checkpoints,
    loading
}) => {
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    // const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
    const [selectedCheckpoint, setSelectedCheckpoint] = useState<string>(initialCheckpoint || '');
    const [isDragging, setIsDragging] = useState(false);

    // Set initial checkpoint or first available from props
    useEffect(() => {
        if (isOpen && checkpoints.length > 0 && !selectedCheckpoint) {
            if (initialCheckpoint) {
                setSelectedCheckpoint(initialCheckpoint);
            } else {
                setSelectedCheckpoint(checkpoints[0].path);
            }
        }
    }, [isOpen, checkpoints, initialCheckpoint, selectedCheckpoint]);

    // Reset file selection when modal opens/closes
    useEffect(() => {
        if (!isOpen) {
            setSelectedFile(null);
        }
    }, [isOpen]);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setSelectedFile(e.target.files[0]);
        }
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);

        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            const file = e.dataTransfer.files[0];
            if (file.type.startsWith('image/')) {
                setSelectedFile(file);
            }
        }
    };

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = () => {
        setIsDragging(false);
    };

    const handleUpload = () => {
        if (selectedFile && selectedCheckpoint) {
            onFileSelected(selectedFile, selectedCheckpoint);
            onClose();
        }
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
            <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl mx-4">
                {/* Header */}
                <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
                    <h2 className="text-lg font-semibold text-gray-900">Upload file</h2>
                    <button
                        onClick={onClose}
                        className="text-gray-400 hover:text-gray-600 transition-colors"
                    >
                        <X size={20} />
                    </button>
                </div>

                {/* Content */}
                <div className="p-6 space-y-6">
                    {/* Model Selector */}
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            Model
                        </label>
                        {loading ? (
                            <div className="text-sm text-gray-500">Loading checkpoints...</div>
                        ) : checkpoints.length === 0 ? (
                            <div className="text-sm text-gray-500">No checkpoints available</div>
                        ) : (
                            <select
                                value={selectedCheckpoint}
                                onChange={(e) => setSelectedCheckpoint(e.target.value)}
                                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"
                            >
                                {checkpoints.map((ckpt) => (
                                    <option key={ckpt.path} value={ckpt.path}>
                                        {ckpt.name}
                                    </option>
                                ))}
                            </select>
                        )}
                    </div>

                    {/* File Upload Area */}
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            File
                        </label>
                        <div
                            className={`border-2 border-dashed rounded-lg p-12 text-center transition-colors cursor-pointer ${isDragging
                                ? 'border-blue-500 bg-blue-50'
                                : 'border-gray-300 hover:border-gray-400'
                                }`}
                            onClick={() => fileInputRef.current?.click()}
                            onDragOver={handleDragOver}
                            onDragLeave={handleDragLeave}
                            onDrop={handleDrop}
                        >
                            <input
                                type="file"
                                ref={fileInputRef}
                                className="hidden"
                                accept="image/*"
                                onChange={handleFileChange}
                            />

                            {selectedFile ? (
                                <div className="space-y-2">
                                    <Upload className="mx-auto text-green-500" size={40} />
                                    <p className="text-sm font-medium text-gray-900">
                                        {selectedFile.name}
                                    </p>
                                    <p className="text-xs text-gray-500">
                                        Click or drag to replace
                                    </p>
                                </div>
                            ) : (
                                <div className="space-y-2">
                                    <Upload className="mx-auto text-gray-400" size={40} />
                                    <p className="text-sm text-gray-600">
                                        <span className="text-blue-600 hover:text-blue-700 font-medium">
                                            Select a file
                                        </span>{' '}
                                        or drop it here
                                    </p>
                                    <p className="text-xs text-gray-500">
                                        The demo shows only the first page for multi-page files.
                                        <br />
                                        JPEG, PNG, BMP, PDF, TIFF, HEIC, DOCX, XLSX, PPTX, up to 50MB
                                    </p>
                                </div>
                            )}
                        </div>
                    </div>
                </div>

                {/* Footer */}
                <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-gray-200 bg-gray-50">
                    <button
                        onClick={onClose}
                        className="px-4 py-2 text-sm font-medium text-gray-700 hover:text-gray-900 transition-colors"
                    >
                        Cancel
                    </button>
                    <button
                        onClick={handleUpload}
                        disabled={!selectedFile || !selectedCheckpoint}
                        className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                    >
                        Upload
                    </button>
                </div>
            </div>
        </div>
    );
};
