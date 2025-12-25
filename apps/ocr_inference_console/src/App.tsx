import { useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { TopRibbon } from './components/TopRibbon';
import { Workspace } from './components/Workspace';
import { GlobalHeader } from './components/GlobalHeader';
import { InferenceProvider } from './contexts/InferenceContext';

function App() {
  const [showUploadModal, setShowUploadModal] = useState(false);

  return (
    <InferenceProvider>
      <div className="flex flex-col h-screen w-full bg-white font-sans text-gray-900 overflow-hidden">

        {/* Global Header */}
        <GlobalHeader />

        {/* Main Layout Area */}
        <div className="flex flex-1 overflow-hidden h-full min-h-0">

          {/* Left Sidebar */}
          <Sidebar />

          {/* Right Content Area: TopRibbon + Workspace */}
          <main className="flex-1 flex flex-col min-w-0 h-full bg-green-50/50 rounded-tl-xl overflow-hidden shadow-sm border-l border-t border-gray-200">

            {/* Top Navigation / Thumbnail Strip */}
            <TopRibbon onUploadClick={() => setShowUploadModal(true)} />

            {/* Main Workspace (Image + Results) */}
            <div className="flex-1 overflow-hidden relative">
              <Workspace
                showUploadModal={showUploadModal}
                onOpenUploadModal={() => setShowUploadModal(true)}
                onCloseUploadModal={() => setShowUploadModal(false)}
              />
            </div>
          </main>
        </div>
      </div>
    </InferenceProvider>
  );
}

export default App;
