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
    const [retryCount, setRetryCount] = useState(0);

    useEffect(() => {
        let retryAttempt = 0;
        const maxRetries = 5;
        const retryDelays = [1000, 2000, 5000, 10000, 20000]; // 1s, 2s, 5s, 10s, 20s
        let timeoutId: number;

        const loadCheckpointsWithRetry = async () => {
            setLoading(true);
            setRetryCount(retryAttempt);

            try {
                const ckpts = await ocrClient.listCheckpoints();

                if (ckpts.length > 0) {
                    // Success!
                    setCheckpoints(ckpts);
                    setLoading(false);
                    setRetryCount(0);

                    // Auto-select first checkpoint if none selected
                    if (!selectedCheckpoint) {
                        onCheckpointChange(ckpts[0].path);
                    }
                } else if (retryAttempt < maxRetries) {
                    // Empty response, retry
                    const delay = retryDelays[retryAttempt];
                    console.log(`No checkpoints found. Retrying in ${delay}ms... (${retryAttempt + 1}/${maxRetries})`);
                    retryAttempt++;
                    timeoutId = setTimeout(loadCheckpointsWithRetry, delay);
                } else {
                    // Max retries reached
                    console.error('Failed to load checkpoints after', maxRetries, 'retries');
                    setCheckpoints([]);
                    setLoading(false);
                }
            } catch (error: any) {
                console.error('Failed to load checkpoints:', error);

                if (retryAttempt < maxRetries) {
                    // Error occurred, retry
                    const delay = retryDelays[retryAttempt];
                    console.log(`Retrying in ${delay}ms... (${retryAttempt + 1}/${maxRetries})`);
                    retryAttempt++;
                    timeoutId = setTimeout(loadCheckpointsWithRetry, delay);
                } else {
                    // Max retries reached
                    setCheckpoints([]);
                    setLoading(false);
                }
            }
        };

        loadCheckpointsWithRetry();

        // Cleanup: cancel pending retry on unmount
        return () => {
            if (timeoutId) {
                clearTimeout(timeoutId);
            }
        };
    }, []);

    if (loading) {
        return (
            <div className="px-4 py-2 text-sm text-gray-500">
                Loading checkpoints...
                {retryCount > 0 && (
                    <div className="text-xs mt-1 text-gray-400">
                        Waiting for backend... (attempt {retryCount + 1}/6)
                    </div>
                )}
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
