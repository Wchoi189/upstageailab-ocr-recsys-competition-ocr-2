import React, { useState, useRef, useEffect } from 'react';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import { RotateCw, ZoomIn, ZoomOut, Download, Code, Eye, FileImage } from 'lucide-react';
import { Stage, Layer, Image as KonvaImage, Rect } from 'react-konva';
import useImage from 'use-image';
import { Button } from './ui/button';
import { cn } from '../lib/utils';
import { motion, AnimatePresence } from 'framer-motion';
import { Light as SyntaxHighlighter } from 'react-syntax-highlighter';
import jsonStyle from 'react-syntax-highlighter/dist/esm/styles/hljs/github';

// --- Mock Data ---
interface OCRItem {
  id: string;
  text: string;
  confidence: number;
  bbox: [number, number, number, number]; // [x, y, w, h]
  label?: string;
}

const MOCK_DATA: OCRItem[] = [
  {
    id: "item-1",
    text: "INVOICE # 1023",
    confidence: 0.99,
    bbox: [50, 50, 200, 40],
    label: "Header"
  },
  {
    id: "item-2",
    text: "Total Amount: $500.00",
    confidence: 0.95,
    bbox: [50, 110, 260, 35],
    label: "Total"
  },
  {
    id: "item-3",
    text: "Due Date: 2023-12-31",
    confidence: 0.92,
    bbox: [50, 160, 240, 30],
    label: "Date"
  },
  {
    id: "item-4",
    text: "Vendor: Acme Corp",
    confidence: 0.98,
    bbox: [50, 210, 220, 30],
    label: "Vendor"
  }
];

// Use a placeholder image that looks like a document
const DOCUMENT_URL = "https://placehold.co/600x800/png?text=Document+Preview&font=roboto";

// --- Components ---

const URLImage = ({ src, x, y }: { src: string; x: number; y: number }) => {
  const [image] = useImage(src);
  return <KonvaImage image={image} x={x} y={y} />;
};

const SkeletonLoader = () => (
  <div className="p-4 space-y-4 animate-pulse">
    <div className="h-4 bg-gray-200 rounded w-3/4"></div>
    <div className="h-4 bg-gray-200 rounded w-1/2"></div>
    <div className="h-4 bg-gray-200 rounded w-5/6"></div>
    <div className="h-4 bg-gray-200 rounded w-2/3"></div>
  </div>
);

export const Workspace: React.FC = () => {
  /* State for bi-directional highlighting */
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });
  const containerRef = useRef<HTMLDivElement>(null);
  const [viewMode, setViewMode] = useState<'preview' | 'json'>('preview');

  /* Zoom State */
  const [scale, setScale] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });

  /* Loading State */
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Simulate initial loading
    const timer = setTimeout(() => setIsLoading(false), 1200);
    return () => clearTimeout(timer);
  }, []);

  // Resize Observer to handle panel resizing
  useEffect(() => {
    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setContainerSize({
          width: entry.contentRect.width,
          height: entry.contentRect.height,
        });
      }
    });

    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }

    return () => resizeObserver.disconnect();
  }, []);

  const handleWheel = (e: any) => {
    e.evt.preventDefault();
    const stage = e.target.getStage();
    const oldScale = stage.scaleX();
    const pointer = stage.getPointerPosition();

    const mousePointTo = {
      x: (pointer.x - stage.x()) / oldScale,
      y: (pointer.y - stage.y()) / oldScale,
    };

    const newScale = e.evt.deltaY > 0 ? oldScale * 0.9 : oldScale * 1.1;
    setScale(newScale);

    const newPos = {
      x: pointer.x - mousePointTo.x * newScale,
      y: pointer.y - mousePointTo.y * newScale,
    };
    setPosition(newPos);
  };

  return (
    <div className="h-full w-full bg-gray-50 p-4">
      <div className="h-full w-full bg-white border border-gray-200 rounded-lg shadow-sm overflow-hidden flex flex-col">

        {/* Toolbar */}
        <div className="h-10 border-b border-gray-200 flex items-center justify-between px-4 bg-gray-50/50 flex-shrink-0">
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1">
              <Button variant="ghost" size="icon" className="h-7 w-7 text-gray-500" onClick={() => { setScale(1); setPosition({ x: 0, y: 0 }); }}><RotateCw size={14} /></Button>
              <Button variant="ghost" size="icon" className="h-7 w-7 text-gray-500" onClick={() => setScale(s => s * 1.1)}><ZoomIn size={14} /></Button>
              <Button variant="ghost" size="icon" className="h-7 w-7 text-gray-500" onClick={() => setScale(s => s * 0.9)}><ZoomOut size={14} /></Button>
            </div>
          </div>
          <div className="flex items-center gap-2">
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
            <div className="h-4 w-[1px] bg-gray-300 mx-1" />
            <Button variant="ghost" size="icon" className="h-7 w-7 text-gray-500"><Download size={14} /></Button>
          </div>
        </div>

        {/* Resizable Area */}
        <div className="flex-1 overflow-hidden">
          <PanelGroup direction="horizontal">

            {/* Left Panel: Image Canvas */}
            <Panel defaultSize={55} minSize={30} className="relative bg-gray-100 flex flex-col">
              <div
                ref={containerRef}
                className="flex-1 overflow-hidden relative"
              >
                {/* Only render stage if we have dimensions to prevent errors */}
                {containerSize.width > 0 && (
                  <Stage
                    width={containerSize.width}
                    height={containerSize.height}
                    className="bg-gray-200 cursor-grab active:cursor-grabbing"
                    draggable
                    onWheel={handleWheel}
                    scaleX={scale}
                    scaleY={scale}
                    x={position.x}
                    y={position.y}
                  >
                    <Layer>
                      {/* The Document Image */}
                      <URLImage src={DOCUMENT_URL} x={20} y={20} />

                      {/* Render all bounding boxes */}
                      {MOCK_DATA.map((item) => {
                        const isHovered = hoveredId === item.id;
                        return (
                          <Rect
                            key={item.id}
                            x={item.bbox[0] + 20}
                            y={item.bbox[1] + 20}
                            width={item.bbox[2]}
                            height={item.bbox[3]}
                            stroke={isHovered ? "#3B5BDB" : "rgba(59, 91, 219, 0.3)"} // Electric Blue
                            strokeWidth={isHovered ? 2 : 1}
                            fill={isHovered ? "rgba(59, 91, 219, 0.1)" : "transparent"}
                            onMouseEnter={() => setHoveredId(item.id)}
                            onMouseLeave={() => setHoveredId(null)}
                          />
                        );
                      })}

                    </Layer>
                  </Stage>
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
                    {MOCK_DATA.length} fields
                  </span>
                </div>

                <div className="relative">
                  <AnimatePresence mode='wait'>
                    {isLoading ? (
                      <motion.div
                        key="loader"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="absolute inset-0 z-10 bg-white"
                      >
                        <SkeletonLoader />
                      </motion.div>
                    ) : (
                      viewMode === 'preview' ? (
                        <motion.div
                          key="preview"
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, y: -10 }}
                          transition={{ duration: 0.2 }}
                          className="p-0"
                        >
                          {MOCK_DATA.map((item) => (
                            <div
                              key={item.id}
                              className={cn(
                                "group flex flex-col py-3 px-4 border-b border-gray-50 transition-colors cursor-pointer",
                                hoveredId === item.id ? "bg-blue-50" : "hover:bg-gray-50"
                              )}
                              onMouseEnter={() => setHoveredId(item.id)}
                              onMouseLeave={() => setHoveredId(null)}
                            >
                              <div className="flex items-center justify-between mb-1">
                                <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">
                                  {item.label || "Text"}
                                </span>
                                <span className={cn(
                                  "text-[10px] px-1.5 py-0.5 rounded",
                                  item.confidence > 0.9 ? "bg-green-100 text-green-700" : "bg-yellow-100 text-yellow-700"
                                )}>
                                  {Math.round(item.confidence * 100)}%
                                </span>
                              </div>
                              <p className="text-sm text-gray-900 font-medium font-mono">{item.text}</p>
                            </div>
                          ))}
                        </motion.div>
                      ) : (
                        <motion.div
                          key="json"
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, y: -10 }}
                          transition={{ duration: 0.2 }}
                          className="p-4"
                        >
                          <div className="text-xs font-mono rounded-md overflow-hidden border border-gray-200">
                            <SyntaxHighlighter
                              language="json"
                              style={jsonStyle}
                              customStyle={{ padding: '1rem', margin: 0, backgroundColor: '#f9fafb' }}
                              wrapLongLines={true}
                            >
                              {JSON.stringify(MOCK_DATA, null, 2)}
                            </SyntaxHighlighter>
                          </div>
                        </motion.div>
                      )
                    )}
                  </AnimatePresence>
                </div>
              </div>
            </Panel>

          </PanelGroup>
        </div>
      </div>
    </div>
  );
};

