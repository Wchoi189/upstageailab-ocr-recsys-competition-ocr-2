import { Home, FileText, Settings, Database, UploadCloud } from 'lucide-react';
import { cn } from '../utils';
import { CheckpointSelector } from './CheckpointSelector';
import { useInference } from '../contexts/InferenceContext';

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

export const Sidebar = () => {
    const {
        checkpoints,
        loadingCheckpoints,
        retryCount,
        selectedCheckpoint,
        inferenceOptions,
        setSelectedCheckpoint,
        updateInferenceOptions,
    } = useInference();

    const {
        enablePerspectiveCorrection,
        displayMode,
        enableGrayscale,
        enableBackgroundNormalization,
        enableSepiaEnhancement,
        confidenceThreshold,
        nmsThreshold,
    } = inferenceOptions;

    return (
        <div className="w-60 h-full bg-white flex flex-col">
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
                        checkpoints={checkpoints}
                        loading={loadingCheckpoints}
                        retryCount={retryCount}
                        selectedCheckpoint={selectedCheckpoint}
                        onCheckpointChange={setSelectedCheckpoint}
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
                                onChange={(e) => updateInferenceOptions({ enablePerspectiveCorrection: e.target.checked })}
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
                                        onChange={(e) => updateInferenceOptions({ displayMode: e.target.value })}
                                    />
                                    <span>Show Corrected</span>
                                </label>
                                <label className="flex items-center gap-2 text-xs">
                                    <input
                                        type="radio"
                                        name="displayMode"
                                        value="original"
                                        checked={displayMode === "original"}
                                        onChange={(e) => updateInferenceOptions({ displayMode: e.target.value })}
                                    />
                                    <span>Show Original</span>
                                </label>
                            </div>
                        )}
                    </div>
                </div>

                <div className="pt-4 border-t border-gray-100">
                    <h4 className="px-4 text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
                        Preprocessing
                    </h4>
                    <div className="px-4 space-y-2">
                        <label className="flex items-center gap-2 text-sm">
                            <input
                                type="checkbox"
                                checked={enableGrayscale}
                                onChange={(e) => updateInferenceOptions({ enableGrayscale: e.target.checked })}
                                className="rounded border-gray-300"
                            />
                            <span>Enable Grayscale</span>
                        </label>
                        <label className="flex items-center gap-2 text-sm">
                            <input
                                type="checkbox"
                                checked={enableBackgroundNormalization}
                                onChange={(e) => updateInferenceOptions({ enableBackgroundNormalization: e.target.checked })}
                                className="rounded border-gray-300"
                            />
                            <span>Background Normalization</span>
                        </label>
                        <label className="flex items-center gap-2 text-sm">
                            <input
                                type="checkbox"
                                checked={enableSepiaEnhancement}
                                onChange={(e) => updateInferenceOptions({ enableSepiaEnhancement: e.target.checked })}
                                className="rounded border-gray-300"
                            />
                            <span>Sepia Enhancement</span>
                        </label>
                    </div>
                </div>

                <div className="pt-4 border-t border-gray-100">
                    <h4 className="px-4 text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
                        Inference Controls
                    </h4>
                    <div className="px-4 space-y-3">
                        <div>
                            <label className="flex items-center justify-between text-xs mb-1">
                                <span className="text-gray-600">Confidence Threshold</span>
                                <span className="text-gray-900 font-mono">{confidenceThreshold.toFixed(2)}</span>
                            </label>
                            <input
                                type="range"
                                min="0.01"
                                max="0.5"
                                step="0.01"
                                value={confidenceThreshold}
                                onChange={(e) => updateInferenceOptions({ confidenceThreshold: parseFloat(e.target.value) })}
                                className="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                            />
                        </div>
                        <div>
                            <label className="flex items-center justify-between text-xs mb-1">
                                <span className="text-gray-600">NMS Threshold</span>
                                <span className="text-gray-900 font-mono">{nmsThreshold.toFixed(2)}</span>
                            </label>
                            <input
                                type="range"
                                min="0.1"
                                max="0.9"
                                step="0.05"
                                value={nmsThreshold}
                                onChange={(e) => updateInferenceOptions({ nmsThreshold: parseFloat(e.target.value) })}
                                className="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                            />
                        </div>
                    </div>
                </div>
            </div>

            <div className="p-4 border-t border-gray-100">
                <NavItem icon={Settings} label="Settings" />
            </div>
        </div>
    );
};
