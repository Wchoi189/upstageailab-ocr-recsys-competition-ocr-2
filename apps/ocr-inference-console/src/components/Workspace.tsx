import React, { useState } from 'react';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import { RotateCw, ZoomIn, ZoomOut, Code, Eye } from 'lucide-react';
import { Button } from './ui/button';
import { cn } from '../utils';

import { Light as SyntaxHighlighter } from 'react-syntax-highlighter';
import jsonStyle from 'react-syntax-highlighter/dist/esm/styles/hljs/github';
import { ImageUploader } from './ImageUploader';
import { PolygonOverlay } from './PolygonOverlay';
import { ocrClient, type Prediction, type InferenceResponse, type PredictionMetadata } from '../api/ocrClient';

interface WorkspaceProps {
    selectedCheckpoint: string | null;
}

export const Workspace: React.FC<WorkspaceProps> = ({ selectedCheckpoint }) => {
    const [viewMode, setViewMode] = useState<'preview' | 'json'>('preview');
    const [imageUrl, setImageUrl] = useState<string | null>(null);
    const [predictions, setPredictions] = useState<Prediction[]>([]);
    const [inferenceMeta, setInferenceMeta] = useState<PredictionMetadata | undefined>(undefined);
    const [previewImageBase64, setPreviewImageBase64] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleImageSelected = async (file: File) => {
        setIsLoading(true);
        setError(null);
        setPredictions([]);

        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Workspace.tsx:24',message:'Image selected - starting inference',data:{fileName:file.name,fileSize:file.size,fileType:file.type,hasCheckpoint:!!selectedCheckpoint,checkpoint:selectedCheckpoint},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'D'})}).catch(()=>{});
        // #endregion

        // Create local URL for preview
        const url = URL.createObjectURL(file);
        setImageUrl(url);

        try {
            const result: InferenceResponse = await ocrClient.predict(file, selectedCheckpoint || undefined);
            // #region agent log
            fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Workspace.tsx:37',message:'Inference response received - predictions data',data:{predictionsCount:result.predictions.length,filename:result.filename,hasMeta:!!result.meta,firstPrediction:result.predictions[0]?{points:result.predictions[0].points,confidence:result.predictions[0].confidence,label:result.predictions[0].label}:null},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'B'})}).catch(()=>{});
            // #endregion
            setPredictions(result.predictions);
            // #region agent log
            fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Workspace.tsx:44',message:'Setting inference metadata',data:{hasMeta:!!result.meta,hasPreviewImage:!!result.preview_image_base64,meta:result.meta},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
            // #endregion
            setInferenceMeta(result.meta);
            setPreviewImageBase64(result.preview_image_base64 || null);
        } catch (e: any) {
            // #region agent log
            fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Workspace.tsx:40',message:'Inference error occurred',data:{errorMessage:e.message,errorStack:e.stack,errorName:e.name},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'E'})}).catch(()=>{});
            // #endregion
            console.error('Inference error:', e);
            // Display detailed error message from backend if available
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
        <div className="h-full w-full bg-gray-50 p-4">
            <div className="h-full w-full bg-white border border-gray-200 rounded-lg shadow-sm overflow-hidden flex flex-col">

                {/* Toolbar */}
                <div className="h-10 border-b border-gray-200 flex items-center justify-between px-4 bg-gray-50/50 flex-shrink-0">
                    <div className="flex items-center gap-2">
                        <div className="flex items-center gap-1">
                            <Button variant="ghost" size="icon" className="h-7 w-7 text-gray-500" onClick={handleReset} title="Reset"><RotateCw size={14} /></Button>
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
                                    <ImageUploader onImageSelected={handleImageSelected} disabled={isLoading} />
                                ) : (
                                    <div className="relative w-full h-full">
                                        {isLoading && (
                                            <div className="absolute inset-0 z-10 bg-white/50 flex items-center justify-center">
                                                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                                            </div>
                                        )}
                                        {/* #region agent log */}
                                        {(() => {
                                            fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'Workspace.tsx:158',message:'Passing props to PolygonOverlay',data:{hasMeta:!!inferenceMeta,hasPredictions:predictions.length>0,meta:inferenceMeta},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
                                            return null;
                                        })()}
                                        {/* #endregion */}
                                        <PolygonOverlay imageUrl={imageUrl} predictions={predictions} meta={inferenceMeta} previewImageBase64={previewImageBase64} />
                                    </div>
                                )}
                            </div>
                        </Panel>

                        <PanelResizeHandle className="w-1 bg-gray-200 hover:bg-primary transition-colors flex items-center justify-center group cursor-col-resize z-10">
                            <div className="h-8 w-1 bg-gray-400 rounded-full group-hover:bg-white" />
                        </PanelResizeHandle>

                        {/* Right Panel: Results View */}
                        <Panel defaultSize={45} minSize={30} className="bg-white flex flex-col">
                            <div className="flex-1 p-0 overflow-y-auto">
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
                        </Panel>

                    </PanelGroup>
                </div>
            </div>
        </div>
    );
};
