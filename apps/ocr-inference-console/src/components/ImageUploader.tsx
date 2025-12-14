import React, { useRef, useState } from 'react';

interface ImageUploaderProps {
    onImageSelected: (file: File) => void;
    disabled?: boolean;
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({ onImageSelected, disabled }) => {
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [preview, setPreview] = useState<string | null>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0];
            const url = URL.createObjectURL(file);
            setPreview(url);
            onImageSelected(file);
        }
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        if (disabled) return;

        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            const file = e.dataTransfer.files[0];
            if (file.type.startsWith('image/')) {
                const url = URL.createObjectURL(file);
                setPreview(url);
                onImageSelected(file);
            }
        }
    };

    return (
        <div className="flex flex-col items-center gap-4">
            <div
                className={`w-full max-w-xl h-64 border-2 border-dashed rounded-lg flex items-center justify-center cursor-pointer transition-colors
            ${disabled ? 'opacity-50 cursor-not-allowed' : 'hover:border-blue-500 border-gray-300'}`}
                onClick={() => !disabled && fileInputRef.current?.click()}
                onDragOver={(e) => e.preventDefault()}
                onDrop={handleDrop}
            >
                {preview ? (
                    <img src={preview} alt="Preview" className="h-full object-contain" />
                ) : (
                    <div className="text-center text-gray-500">
                        <p>Click or drag image here</p>
                        <p className="text-xs">Supports JPG, PNG</p>
                    </div>
                )}
                <input
                    type="file"
                    ref={fileInputRef}
                    className="hidden"
                    accept="image/*"
                    onChange={handleFileChange}
                    disabled={disabled}
                />
            </div>
        </div>
    );
};
