
import { useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { TopRibbon } from './components/TopRibbon';
import { Workspace } from './components/Workspace';

function App() {
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<string | null>(null);
  const [enablePerspectiveCorrection, setEnablePerspectiveCorrection] = useState(false);
  const [displayMode, setDisplayMode] = useState("corrected");

  return (
    <div className="flex h-screen w-full bg-white font-sans text-gray-900 overflow-hidden">
      {/* Left Sidebar */}
      <Sidebar
        selectedCheckpoint={selectedCheckpoint}
        onCheckpointChange={setSelectedCheckpoint}
        enablePerspectiveCorrection={enablePerspectiveCorrection}
        onPerspectiveCorrectionChange={setEnablePerspectiveCorrection}
        displayMode={displayMode}
        onDisplayModeChange={setDisplayMode}
      />

      {/* Main Content Area */}
      <main className="flex flex-1 flex-col h-full min-w-0">

        {/* Top Navigation / Thumbnail Strip */}
        <TopRibbon />

        {/* Main Workspace (Image + Results) */}
        <div className="flex-1 overflow-hidden relative">
          <Workspace
            selectedCheckpoint={selectedCheckpoint}
            enablePerspectiveCorrection={enablePerspectiveCorrection}
            displayMode={displayMode}
          />
        </div>
      </main>
    </div>
  );
}

export default App;
