import { TelemetryEvent } from './telemetryWatcher';

export interface FilterCriteria {
    status?: 'success' | 'error' | 'policy_violation' | 'all';
    toolName?: string;
    startTime?: Date;
    endTime?: Date;
}

export class TelemetryCache {
    private cache: Map<string, TelemetryEvent>;
    private accessOrder: string[] = [];
    private maxSize: number;

    constructor(maxSize: number = 1000) {
        this.cache = new Map();
        this.maxSize = maxSize;
    }

    add(event: TelemetryEvent): void {
        const key = `${event.timestamp}-${event.args_hash}`;

        // If already exists, update access order
        if (this.cache.has(key)) {
            this.updateAccessOrder(key);
            this.cache.set(key, event);
            return;
        }

        // Add new event
        this.cache.set(key, event);
        this.accessOrder.push(key);

        // LRU eviction if needed
        if (this.cache.size > this.maxSize) {
            const oldestKey = this.accessOrder.shift();
            if (oldestKey) {
                this.cache.delete(oldestKey);
            }
        }
    }

    addBatch(events: TelemetryEvent[]): void {
        events.forEach(event => this.add(event));
    }

    query(filters?: FilterCriteria): TelemetryEvent[] {
        let results = Array.from(this.cache.values());

        if (!filters) {
            return results;
        }

        // Filter by status
        if (filters.status && filters.status !== 'all') {
            results = results.filter(e => e.status === filters.status);
        }

        // Filter by tool name
        if (filters.toolName) {
            const searchTerm = filters.toolName.toLowerCase();
            results = results.filter(e =>
                e.tool_name.toLowerCase().includes(searchTerm)
            );
        }

        // Filter by time range
        if (filters.startTime) {
            results = results.filter(e =>
                new Date(e.timestamp) >= filters.startTime!
            );
        }

        if (filters.endTime) {
            results = results.filter(e =>
                new Date(e.timestamp) <= filters.endTime!
            );
        }

        return results;
    }

    getRecent(limit: number = 20): TelemetryEvent[] {
        const allEvents = Array.from(this.cache.values());
        return allEvents
            .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
            .slice(0, limit);
    }

    getByToolName(toolName: string): TelemetryEvent[] {
        return this.query({ toolName });
    }

    getViolations(): TelemetryEvent[] {
        return this.query({ status: 'policy_violation' });
    }

    getErrors(): TelemetryEvent[] {
        return this.query({ status: 'error' });
    }

    getStats() {
        const events = Array.from(this.cache.values());
        const total = events.length;
        const success = events.filter(e => e.status === 'success').length;
        const errors = events.filter(e => e.status === 'error').length;
        const violations = events.filter(e => e.status === 'policy_violation').length;

        const durations = events
            .filter(e => typeof e.duration_ms === 'number')
            .map(e => e.duration_ms!);

        const avgDuration = durations.length > 0
            ? durations.reduce((a, b) => a + b, 0) / durations.length
            : 0;

        // Tool-specific stats
        const toolStats = new Map<string, {
            count: number;
            successes: number;
            errors: number;
            totalDuration: number;
        }>();

        events.forEach(event => {
            if (!toolStats.has(event.tool_name)) {
                toolStats.set(event.tool_name, {
                    count: 0,
                    successes: 0,
                    errors: 0,
                    totalDuration: 0
                });
            }
            const stats = toolStats.get(event.tool_name)!;
            stats.count++;
            if (event.status === 'success') {
                stats.successes++;
            }
            if (event.status === 'error') {
                stats.errors++;
            }
            if (event.duration_ms) {
                stats.totalDuration += event.duration_ms;
            }
        });

        return {
            total,
            success,
            errors,
            violations,
            avgDuration: Math.round(avgDuration * 100) / 100,
            cacheSize: this.cache.size,
            maxSize: this.maxSize,
            toolStats: Array.from(toolStats.entries()).map(([name, stats]) => ({
                name,
                ...stats,
                avgDuration: stats.count > 0 ? stats.totalDuration / stats.count : 0
            }))
        };
    }

    clear(): void {
        this.cache.clear();
        this.accessOrder = [];
    }

    size(): number {
        return this.cache.size;
    }

    private updateAccessOrder(key: string): void {
        const index = this.accessOrder.indexOf(key);
        if (index > -1) {
            this.accessOrder.splice(index, 1);
            this.accessOrder.push(key);
        }
    }
}
