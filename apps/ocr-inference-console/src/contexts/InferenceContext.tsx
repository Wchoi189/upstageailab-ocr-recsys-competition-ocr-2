import React, { createContext, useContext, useEffect, useState, useRef, useCallback } from 'react';
import { ocrClient, type Checkpoint } from '../api/ocrClient';

interface InferenceOptions {
  enablePerspectiveCorrection: boolean;
  displayMode: string;
  enableGrayscale: boolean;
  enableBackgroundNormalization: boolean;
  enableSepiaEnhancement: boolean;
  enableClahe: boolean;
  confidenceThreshold: number;
  nmsThreshold: number;
}

interface InferenceState {
  checkpoints: Checkpoint[];
  loadingCheckpoints: boolean;
  retryCount: number;
  selectedCheckpoint: string | null;
  inferenceOptions: InferenceOptions;
}

interface InferenceActions {
  setSelectedCheckpoint: (path: string | null) => void;
  updateInferenceOptions: (opts: Partial<InferenceOptions>) => void;
  refreshCheckpoints: () => Promise<void>;
}

type InferenceContextValue = InferenceState & InferenceActions;

const InferenceContext = createContext<InferenceContextValue | null>(null);

export function useInference(): InferenceContextValue {
  const context = useContext(InferenceContext);
  if (!context) {
    throw new Error('useInference must be used within an InferenceProvider');
  }
  return context;
}

interface InferenceProviderProps {
  children: React.ReactNode;
}

export function InferenceProvider({ children }: InferenceProviderProps): React.ReactElement {
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [loadingCheckpoints, setLoadingCheckpoints] = useState(true);
  const [retryCount, setRetryCount] = useState(0);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<string | null>(null);

  const loadingInProgress = useRef(false);

  const [inferenceOptions, setInferenceOptions] = useState<InferenceOptions>({
    enablePerspectiveCorrection: false,
    displayMode: 'corrected',
    enableGrayscale: false,
    enableBackgroundNormalization: false,
    enableSepiaEnhancement: false,
    enableClahe: false,
    confidenceThreshold: 0.1,
    nmsThreshold: 0.4,
  });

  const updateInferenceOptions = useCallback((opts: Partial<InferenceOptions>) => {
    setInferenceOptions((prev) => ({ ...prev, ...opts }));
  }, []);

  const refreshCheckpoints = useCallback(async () => {
    if (loadingInProgress.current) return;
    loadingInProgress.current = true;

    let retryAttempt = 0;
    const maxRetries = 5;
    const retryDelays = [1000, 2000, 5000, 10000, 20000];

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
          setTimeout(loadCheckpoints, delay);
        } else {
          setCheckpoints([]);
          setLoadingCheckpoints(false);
          loadingInProgress.current = false;
        }
      } catch (error) {
        if (retryAttempt < maxRetries) {
          const delay = retryDelays[retryAttempt];
          retryAttempt++;
          setTimeout(loadCheckpoints, delay);
        } else {
          setCheckpoints([]);
          setLoadingCheckpoints(false);
          loadingInProgress.current = false;
        }
      }
    };

    await loadCheckpoints();
  }, [selectedCheckpoint]);

  useEffect(() => {
    refreshCheckpoints();
  }, []);

  const value: InferenceContextValue = {
    checkpoints,
    loadingCheckpoints,
    retryCount,
    selectedCheckpoint,
    inferenceOptions,
    setSelectedCheckpoint,
    updateInferenceOptions,
    refreshCheckpoints,
  };

  return <InferenceContext.Provider value={value}>{children}</InferenceContext.Provider>;
}
