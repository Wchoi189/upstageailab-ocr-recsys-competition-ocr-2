import { useEffect, useState } from 'react';
// import { Database } from 'lucide-react';
import { ocrClient, type Checkpoint } from '../api/ocrClient';

interface CheckpointSelectorProps {
    selectedCheckpoint: string | null;
    onCheckpointChange: (checkpoint: string) => void;
}

export const CheckpointSelector = ({ selectedCheckpoint, onCheckpointChange }: CheckpointSelectorProps) => {
    const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const loadCheckpoints = async () => {
            setLoading(true);
            // #region agent log
            fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'CheckpointSelector.tsx:15',message:'Starting checkpoint load',data:{timestamp:Date.now()},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'F'})}).catch(()=>{});
            // #endregion

            // Test direct API access first with timeout
            try {
                // #region agent log
                const testStartTime = Date.now();
                fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'CheckpointSelector.tsx:20',message:'Testing direct API fetch',data:{url:'http://localhost:8000/api/inference/checkpoints?limit=5'},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'F'})}).catch(()=>{});
                // #endregion
                const fetchPromise = fetch('http://localhost:8000/api/inference/checkpoints?limit=5', {
                    method: 'GET',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                });
                const timeoutPromise = new Promise<never>((_, reject) => {
                    setTimeout(() => reject(new Error('Fetch timeout after 10 seconds')), 10000);
                });
                const testResponse = await Promise.race([fetchPromise, timeoutPromise]);
                // #region agent log
                const testDuration = Date.now() - testStartTime;
                fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'CheckpointSelector.tsx:30',message:'Direct API fetch result',data:{status:testResponse.status,statusText:testResponse.statusText,ok:testResponse.ok,durationMs:testDuration},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'F'})}).catch(()=>{});
                // #endregion
            } catch (testError: any) {
                // #region agent log
                const testDuration = Date.now() - Date.now();
                fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'CheckpointSelector.tsx:35',message:'Direct API fetch failed',data:{errorMessage:testError?.message,errorName:testError?.name,errorStack:testError?.stack},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'F'})}).catch(()=>{});
                // #endregion
            }

            try {
                // Add timeout to prevent infinite hanging
                const timeoutPromise = new Promise<never>((_, reject) => {
                    setTimeout(() => reject(new Error('Checkpoint loading timeout after 30 seconds')), 30000);
                });

                const ckpts = await Promise.race([
                    ocrClient.listCheckpoints(),
                    timeoutPromise
                ]);
                // #region agent log
                fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'CheckpointSelector.tsx:40',message:'Checkpoints loaded successfully',data:{checkpointsCount:ckpts.length,firstCheckpoint:ckpts[0]||null},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'F'})}).catch(()=>{});
                // #endregion
                setCheckpoints(ckpts);
                setLoading(false);

                // Auto-select first checkpoint if none selected
                if (!selectedCheckpoint && ckpts.length > 0) {
                    // #region agent log
                    fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'CheckpointSelector.tsx:65',message:'Auto-selecting first checkpoint',data:{checkpointPath:ckpts[0].path,checkpointName:ckpts[0].name},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'H'})}).catch(()=>{});
                    // #endregion
                    onCheckpointChange(ckpts[0].path);
                } else {
                    // #region agent log
                    fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'CheckpointSelector.tsx:70',message:'Not auto-selecting checkpoint',data:{hasSelectedCheckpoint:!!selectedCheckpoint,checkpointsCount:ckpts.length},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'H'})}).catch(()=>{});
                    // #endregion
                }
            } catch (error: any) {
                // #region agent log
                fetch('http://127.0.0.1:7242/ingest/842889c6-5ff1-47b5-bc88-99b58e395178',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'CheckpointSelector.tsx:50',message:'Checkpoint load error',data:{errorMessage:error?.message,errorStack:error?.stack,errorName:error?.name},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'F'})}).catch(()=>{});
                // #endregion
                console.error('Failed to load checkpoints:', error);
                setCheckpoints([]);
                setLoading(false);
            }
        };
        loadCheckpoints();
    }, []);

    if (loading) {
        return (
            <div className="px-4 py-2 text-sm text-gray-500">
                Loading checkpoints...
            </div>
        );
    }

    if (checkpoints.length === 0) {
        return (
            <div className="px-4 py-2 text-sm text-gray-500">
                No checkpoints found
            </div>
        );
    }

    return (
        <div className="px-2">
            <h4 className="px-2 text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
                Model Checkpoint
            </h4>
            <select
                value={selectedCheckpoint || ''}
                onChange={(e) => onCheckpointChange(e.target.value)}
                className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"
            >
                {checkpoints.map((ckpt) => (
                    <option key={ckpt.path} value={ckpt.path}>
                        {ckpt.name} ({ckpt.size_mb}MB)
                    </option>
                ))}
            </select>
        </div>
    );
};
