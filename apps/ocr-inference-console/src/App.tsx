import { useEffect, useState, useRef } from 'react';
import { Sidebar } from './components/Sidebar';
import { TopRibbon } from './components/TopRibbon';
import { Workspace } from './components/Workspace';
import { GlobalHeader } from './components/GlobalHeader';
import { ocrClient, type Checkpoint } from './api/ocrClient';

function App() {
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [loadingCheckpoints, setLoadingCheckpoints] = useState(true);
  const [retryCount, setRetryCount] = useState(0);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<string | null>(null);

  const loadingInProgress = useRef(false);

  const [enablePerspectiveCorrection, setEnablePerspectiveCorrection] = useState(false);
  const [displayMode, setDisplayMode] = useState("corrected");
  const [enableGrayscale, setEnableGrayscale] = useState(false);
  const [enableBackgroundNormalization, setEnableBackgroundNormalization] = useState(false);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.1);
  const [nmsThreshold, setNmsThreshold] = useState(0.4);
  const [showUploadModal, setShowUploadModal] = useState(false);

  useEffect(() => {
    // Prevent multiple concurrent retry loops (e.g. from React StrictMode)
    if (loadingInProgress.current) return;
    loadingInProgress.current = true;

    let retryAttempt = 0;
    const maxRetries = 5;
    const retryDelays = [1000, 2000, 5000, 10000, 20000];
    let timeoutId: any;

    const loadCheckpoints = async () => {
      setLoadingCheckpoints(true);
      setRetryCount(retryAttempt);
      try {
        const ckpts = await ocrClient.listCheckpoints();
        if (ckpts && ckpts.length > 0) {
          setCheckpoints(ckpts);
          setLoadingCheckpoints(false);
          setRetryCount(0);
          loadingInProgress.current = false;
          if (!selectedCheckpoint) {
            setSelectedCheckpoint(ckpts[0].path);
          }
        } else if (retryAttempt < maxRetries) {
          const delay = retryDelays[retryAttempt];
          retryAttempt++;
          timeoutId = setTimeout(loadCheckpoints, delay);
        } else {
          setCheckpoints([]);
          setLoadingCheckpoints(false);
          loadingInProgress.current = false;
        }
      } catch (error) {
        if (retryAttempt < maxRetries) {
          const delay = retryDelays[retryAttempt];
          retryAttempt++;
          timeoutId = setTimeout(loadCheckpoints, delay);
        } else {
          setCheckpoints([]);
          setLoadingCheckpoints(false);
          loadingInProgress.current = false;
        }
      }
    };

    loadCheckpoints();
    return () => {
      if (timeoutId) clearTimeout(timeoutId);
      loadingInProgress.current = false;
    };
  }, []);

  return (
    <div className="flex flex-col h-screen w-full bg-white font-sans text-gray-900 overflow-hidden">

      {/* Global Header */}
      <GlobalHeader />

      {/* Main Layout Area */}
      <div className="flex flex-1 overflow-hidden h-full min-h-0">

        {/* Left Sidebar */}
        <Sidebar
          checkpoints={checkpoints}
          loadingCheckpoints={loadingCheckpoints}
          retryCount={retryCount}
          selectedCheckpoint={selectedCheckpoint}
          onCheckpointChange={setSelectedCheckpoint}
          enablePerspectiveCorrection={enablePerspectiveCorrection}
          onPerspectiveCorrectionChange={setEnablePerspectiveCorrection}
          displayMode={displayMode}
          onDisplayModeChange={setDisplayMode}
          enableGrayscale={enableGrayscale}
          onGrayscaleChange={setEnableGrayscale}
          enableBackgroundNormalization={enableBackgroundNormalization}
          onBackgroundNormalizationChange={setEnableBackgroundNormalization}
          confidenceThreshold={confidenceThreshold}
          onConfidenceThresholdChange={setConfidenceThreshold}
          nmsThreshold={nmsThreshold}
          onNmsThresholdChange={setNmsThreshold}
        />

        {/* Right Content Area: TopRibbon + Workspace */}
        <main className="flex-1 flex flex-col min-w-0 h-full bg-green-50/50 rounded-tl-xl overflow-hidden shadow-sm border-l border-t border-gray-200">

          {/* Top Navigation / Thumbnail Strip */}
          <TopRibbon onUploadClick={() => setShowUploadModal(true)} />

          {/* Main Workspace (Image + Results) */}
          <div className="flex-1 overflow-hidden relative">
            <Workspace
              checkpoints={checkpoints}
              loadingCheckpoints={loadingCheckpoints}
              selectedCheckpoint={selectedCheckpoint}
              enablePerspectiveCorrection={enablePerspectiveCorrection}
              displayMode={displayMode}
              enableGrayscale={enableGrayscale}
              enableBackgroundNormalization={enableBackgroundNormalization}
              confidenceThreshold={confidenceThreshold}
              nmsThreshold={nmsThreshold}
              showUploadModal={showUploadModal}
              onOpenUploadModal={() => setShowUploadModal(true)}
              onCloseUploadModal={() => setShowUploadModal(false)}
            />
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
