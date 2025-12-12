import { useEffect, useState } from 'react';
import { Database } from 'lucide-react';
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
            const ckpts = await ocrClient.listCheckpoints();
            setCheckpoints(ckpts);
            setLoading(false);

            // Auto-select first checkpoint if none selected
            if (!selectedCheckpoint && ckpts.length > 0) {
                onCheckpointChange(ckpts[0].path);
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
