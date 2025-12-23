import React, { useState } from 'react';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import { RotateCw, ZoomIn, ZoomOut, Code, Eye, Upload, ChevronLeft, ChevronRight, Maximize, Minus, Plus, Box } from 'lucide-react';
import { Button } from './ui/button';
import { cn } from '../utils';

import { Light as SyntaxHighlighter } from 'react-syntax-highlighter';
import jsonStyle from 'react-syntax-highlighter/dist/esm/styles/hljs/github';
import { UploadModal } from './UploadModal';
import { PolygonOverlay } from './PolygonOverlay';
import { ocrClient, type Prediction, type InferenceResponse, type PredictionMetadata } from '../api/ocrClient';
import { useInference } from '../contexts/InferenceContext';

interface WorkspaceProps {
    showUploadModal?: boolean;
    onOpenUploadModal?: () => void;
    onCloseUploadModal?: () => void;
}

export const Workspace: React.FC<WorkspaceProps> = ({
    showUploadModal = false,
    onOpenUploadModal,
    onCloseUploadModal
}) => {
    const {
        checkpoints,
        loadingCheckpoints,
        selectedCheckpoint,
        inferenceOptions,
    } = useInference();

    const {
        enablePerspectiveCorrection,
        displayMode,
        enableGrayscale,
        enableBackgroundNormalization,
        enableSepiaEnhancement,
        enableClahe,
        sepiaDisplayMode,
        confidenceThreshold,
        nmsThreshold,
    } = inferenceOptions;
    const [viewMode, setViewMode] = useState<'preview' | 'json'>('preview');
    const [imageUrl, setImageUrl] = useState<string | null>(null);
    const [predictions, setPredictions] = useState<Prediction[]>([]);
    const [inferenceMeta, setInferenceMeta] = useState<PredictionMetadata | undefined>(undefined);
    const [previewImageBase64, setPreviewImageBase64] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [currentFile, setCurrentFile] = useState<File | null>(null); // Track current file for re-running

    const averageConfidence = predictions.length > 0
        ? Math.round(predictions.reduce((acc, curr) => acc + curr.confidence, 0) / predictions.length * 100)
        : 0;

    const handleImageSelected = async (file: File, checkpoint?: string) => {
        setIsLoading(true);
        setError(null);
        setPredictions([]);        // Store file for re-running
        setCurrentFile(file);

        // Create local URL for preview
        const url = URL.createObjectURL(file);
        setImageUrl(url);

        try {
            const result: InferenceResponse = await ocrClient.predict(
                file,
                checkpoint || selectedCheckpoint || undefined,
                enablePerspectiveCorrection,
                displayMode,
                enableGrayscale,
                enableBackgroundNormalization,
                enableSepiaEnhancement,
                enableClahe,
                confidenceThreshold,
                nmsThreshold,
                sepiaDisplayMode
            ); setPredictions(result.predictions); setInferenceMeta(result.meta);
            setPreviewImageBase64(result.preview_image_base64 || null);
        } catch (e: any) {
            console.error('Inference error:', e);
            // Display detailed error message from backend if available
            setError(e.message || "Unknown error occurred during inference. Check backend logs.");
        } finally {
            setIsLoading(false);
        }
    };

    // New function to re-run inference on current image
    const handleRerunInference = async () => {
        if (!currentFile) return;

        setIsLoading(true);
        setError(null);
        setPredictions([]);

        try {
            const result: InferenceResponse = await ocrClient.predict(
                currentFile,
                selectedCheckpoint || undefined,
                enablePerspectiveCorrection,
                displayMode,
                enableGrayscale,
                enableBackgroundNormalization,
                enableSepiaEnhancement,
                enableClahe,
                confidenceThreshold,
                nmsThreshold,
                sepiaDisplayMode
            );
            setPredictions(result.predictions);
            setInferenceMeta(result.meta);
            setPreviewImageBase64(result.preview_image_base64 || null);
        } catch (e: any) {
            console.error('Inference error:', e);
            setError(e.message || "Unknown error occurred during inference. Check backend logs.");
        } finally {
            setIsLoading(false);
        }
    };

    const handleReset = () => {
        setImageUrl(null);
        setPredictions([]);
        setInferenceMeta(undefined);
        setPreviewImageBase64(null);
        setError(null);
        setCurrentFile(null); // Clear current file
    };

    const handleLoadDemo = async () => {
        setIsLoading(true);
        setError(null);
        setPredictions([]);

        try {
            // 1. Load Image
            const demoImageUrl = "/demo.jpg";
            setImageUrl(demoImageUrl);

            // 2. Load Data
            const response = await fetch("/demo.json");
            if (!response.ok) throw new Error("Failed to load demo data");

            const words = await response.json();

            // 3. Transform to Prediction format
            // The demo json is a dictionary of word_id -> { points: [[x,y], ...], ... }
            const demoPredictions: Prediction[] = Object.values(words).map((word: any) => ({
                points: word.points,
                confidence: 1.0, // Mock confidence
                label: word.transcription || undefined
            }));

            setPredictions(demoPredictions);

        } catch (e: any) {
            console.error(e);
            setError(e.message || "Failed to load demo");
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="h-full w-full bg-white">
            <div className="h-full w-full bg-white overflow-hidden flex flex-col">

                {/* Toolbar */}
                <div className="flex items-center justify-between px-4 bg-gray-50/50 flex-shrink-0">
                    <div className="flex items-center gap-2">
                        <div className="flex items-center gap-1">
                            {/* Run Inference Button - replaces Upload */}
                            <Button
                                variant="ghost"
                                size="sm"
                                className="h-7 text-xs font-medium text-white bg-blue-600 hover:bg-blue-700 px-3 disabled:opacity-50 disabled:cursor-not-allowed"
                                onClick={handleRerunInference}
                                disabled={!currentFile || isLoading}
                                title={!currentFile ? "Upload an image first" : "Run inference with current settings"}
                            >
                                <RotateCw size={14} className={cn("mr-1", isLoading && "animate-spin")} />
                                {isLoading ? "Running..." : "Run Inference"}
                            </Button>
                            <Button variant="ghost" size="icon" className="h-7 w-7 text-gray-500" onClick={handleReset} title="Reset" disabled={!imageUrl}><RotateCw size={14} /></Button>
                            <Button variant="ghost" size="sm" className="h-7 text-xs text-gray-600 px-2" onClick={handleLoadDemo}>Demo</Button>
                            {/* Zoom buttons disabled for now as Viewer doesn't support zoom yet */}
                            <Button variant="ghost" size="icon" className="h-7 w-7 text-gray-300 cursor-not-allowed"><ZoomIn size={14} /></Button>
                            <Button variant="ghost" size="icon" className="h-7 w-7 text-gray-300 cursor-not-allowed"><ZoomOut size={14} /></Button>
                        </div>
                    </div>
                    <div className="flex items-center gap-2">
                        {error && (
                            <div className="flex items-center px-3 py-1.5 bg-red-50 text-red-700 text-xs rounded border border-red-200 mr-2 max-w-md truncate" title={error}>
                                <span className="font-semibold mr-1">Error:</span> {error}
                            </div>
                        )}
                        <div className="flex items-center bg-gray-200 rounded-md p-0.5">
                            <button
                                onClick={() => setViewMode('preview')}
                                className={cn(
                                    "px-3 py-1 text-xs font-medium rounded-sm flex items-center gap-1 transition-all",
                                    viewMode === 'preview' ? "bg-white shadow-sm text-gray-900" : "text-gray-600 hover:text-gray-900"
                                )}
                            >
                                <Eye size={12} /> Preview
                            </button>
                            <button
                                onClick={() => setViewMode('json')}
                                className={cn(
                                    "px-3 py-1 text-xs font-medium rounded-sm flex items-center gap-1 transition-all",
                                    viewMode === 'json' ? "bg-white shadow-sm text-gray-900" : "text-gray-600 hover:text-gray-900"
                                )}
                            >
                                <Code size={12} /> JSON
                            </button>
                        </div>
                    </div>
                </div>

                {/* Resizable Area */}
                <div className="flex-1 overflow-hidden">
                    <PanelGroup direction="horizontal">

                        {/* Left Panel: Image Canvas */}
                        <Panel defaultSize={55} minSize={30} className="relative bg-gray-100 flex flex-col">
                            <div className="flex-1 overflow-hidden relative flex items-center justify-center">
                                {!imageUrl ? (
                                    <div className="flex flex-col items-center justify-center gap-4 text-center">
                                        <div className="text-gray-400">
                                            <Upload size={48} strokeWidth={1.5} />
                                        </div>
                                        <div>
                                            <p className="text-sm text-gray-600 mb-2">No image loaded</p>
                                            <button
                                                onClick={onOpenUploadModal}
                                                className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 transition-colors"
                                            >
                                                Upload Image
                                            </button>
                                        </div>
                                    </div>
                                ) : (
                                    <div className="relative w-full h-full">
                                        {isLoading && (
                                            <div className="absolute inset-0 z-10 bg-white/50 flex items-center justify-center">
                                                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                                            </div>
                                        )}                                        <PolygonOverlay imageUrl={imageUrl} predictions={predictions} meta={inferenceMeta} previewImageBase64={previewImageBase64} />
                                    </div>
                                )}
                            </div>
                            {/* Left Panel Footer: Viewer Controls */}
                            <div className="h-12 bg-white border-t border-gray-200 flex items-center justify-between px-4 shrink-0">
                                <div className="flex items-center gap-2">
                                    <Button variant="ghost" size="icon" className="h-8 w-8 text-gray-500 hover:text-blue-600 hover:bg-blue-50" title="Rotate">
                                        <RotateCw size={16} />
                                    </Button>
                                    <Button variant="ghost" size="icon" className="h-8 w-8 text-gray-500 hover:text-blue-600 hover:bg-blue-50" title="Full Screen">
                                        <Maximize size={16} />
                                    </Button>
                                </div>
                                <div className="flex items-center gap-4">
                                    <div className="flex items-center gap-2 bg-gray-100 rounded-md p-1">
                                        <Button variant="ghost" size="icon" className="h-6 w-6 text-gray-500 hover:bg-white rounded-sm">
                                            <Minus size={14} />
                                        </Button>
                                        <div className="w-16 h-1 bg-gray-300 rounded-full overflow-hidden">
                                            <div className="w-1/2 h-full bg-blue-500" />
                                        </div>
                                        <div className="h-2 w-2 bg-white border-2 border-primary rounded-full shadow-sm -ml-10 z-10 hidden" /> {/* Slider handle mock */}
                                        <Button variant="ghost" size="icon" className="h-6 w-6 text-gray-500 hover:bg-white rounded-sm">
                                            <Plus size={14} />
                                        </Button>
                                    </div>
                                </div>
                                <div className="flex items-center gap-2 text-gray-500 font-mono text-sm">
                                    <Button variant="ghost" size="icon" className="h-8 w-8 text-gray-400 cursor-not-allowed">
                                        <ChevronLeft size={16} />
                                    </Button>
                                    <span>1 / 1</span>
                                    <Button variant="ghost" size="icon" className="h-8 w-8 text-gray-400 cursor-not-allowed">
                                        <ChevronRight size={16} />
                                    </Button>
                                </div>
                            </div>
                        </Panel>

                        <PanelResizeHandle className="w-1 bg-gray-200 hover:bg-primary transition-colors flex items-center justify-center group cursor-col-resize z-10">
                            <div className="h-8 w-1 bg-gray-400 rounded-full group-hover:bg-white" />
                        </PanelResizeHandle>

                        {/* Right Panel: Results View */}
                        <Panel defaultSize={45} minSize={30} className="bg-white flex flex-col">
                            <div className="flex-1 p-0 overflow-y-auto">
                                {/* New Header Section */}
                                <div className="px-6 py-5 bg-blue-50/20 border-b border-blue-100">
                                    <h2 className="text-xl font-semibold text-gray-900 mb-3">Document OCR</h2>
                                    <div className="flex items-center gap-3 bg-blue-50 border border-blue-100 rounded-md p-2 w-full">
                                        <Box size={18} className="text-blue-600" />
                                        <div className="flex items-center gap-2">
                                            <span className="text-sm text-gray-500 font-medium">Model</span>
                                            <span className="text-sm font-semibold text-gray-900">ocr-2.2.1</span>
                                        </div>
                                    </div>
                                </div>

                                <div className="p-4 border-b border-gray-100 flex items-center justify-between">
                                    <h3 className="text-sm font-semibold text-gray-900">Extracted Data</h3>
                                    <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded-full">
                                        {predictions.length} fields
                                    </span>
                                </div>

                                <div className="relative">
                                    {/* Content based on viewMode */}
                                    {viewMode === 'preview' ? (
                                        <div className="p-0">
                                            {predictions.length === 0 && !isLoading && (
                                                <div className="p-8 text-center text-gray-400 text-sm">
                                                    No predictions yet. Upload an image to start.
                                                </div>
                                            )}
                                            {predictions.map((item, idx) => (
                                                <div key={idx} className="group flex flex-col py-3 px-4 border-b border-gray-50 hover:bg-gray-50 transition-colors">
                                                    <div className="flex items-center justify-between mb-1">
                                                        <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">
                                                            Polygon {idx + 1}
                                                        </span>
                                                        <span className={cn(
                                                            "text-[10px] px-1.5 py-0.5 rounded",
                                                            item.confidence > 0.9 ? "bg-green-100 text-green-700" : "bg-yellow-100 text-yellow-700"
                                                        )}>
                                                            {Math.round(item.confidence * 100)}%
                                                        </span>
                                                    </div>
                                                    <div className="text-xs text-gray-500 font-mono">
                                                        {JSON.stringify(item.points)}
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    ) : (
                                        <div className="p-4">
                                            <div className="text-xs font-mono rounded-md overflow-hidden border border-gray-200">
                                                <SyntaxHighlighter
                                                    language="json"
                                                    style={jsonStyle}
                                                    customStyle={{ padding: '1rem', margin: 0, backgroundColor: '#f9fafb' }}
                                                    wrapLongLines={true}
                                                >
                                                    {JSON.stringify(predictions, null, 2)}
                                                </SyntaxHighlighter>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </div>
                            {/* Right Panel Footer: Confidence Score */}
                            <div className="h-12 bg-blue-50/50 border-t border-blue-100 flex items-center px-6 shrink-0">
                                <span className="text-sm text-gray-600 font-medium mr-2">Confidence score</span>
                                <span className={cn(
                                    "text-lg font-bold",
                                    averageConfidence >= 80 ? "text-blue-600" :
                                        averageConfidence >= 50 ? "text-yellow-600" : "text-red-600"
                                )}>
                                    {averageConfidence}
                                </span>
                            </div>
                        </Panel>

                    </PanelGroup>
                </div>

                {/* Upload Modal */}
                <UploadModal
                    isOpen={showUploadModal}
                    onClose={onCloseUploadModal || (() => { })}
                    onFileSelected={handleImageSelected}
                    initialCheckpoint={selectedCheckpoint}
                    checkpoints={checkpoints}
                    loading={loadingCheckpoints}
                />
            </div>
        </div>
    );
};
