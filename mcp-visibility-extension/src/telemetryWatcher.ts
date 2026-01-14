import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import * as readline from 'readline';
import { TelemetryCache } from './telemetryCache';

/* eslint-disable @typescript-eslint/naming-convention */
export interface TelemetryEvent {
    timestamp: string;
    tool_name: string;
    args_hash: string;
    status: 'success' | 'error' | 'policy_violation';
    duration_ms?: number;
    module?: string;
    policy?: string;
    error?: string;
}
/* eslint-enable @typescript-eslint/naming-convention */

export interface TelemetrySummary {
    total: number;
    success: number;
    errors: number;
    policyViolations: number;
    avgDurationMs: number;
    recentCalls: TelemetryEvent[];
}

export class TelemetryWatcher implements vscode.Disposable {
    private watcher: vscode.FileSystemWatcher | undefined;
    private telemetryPath: string;
    private lastSize: number = 0;
    private lastInode: number | undefined;
    private events: TelemetryEvent[] = [];
    private cache: TelemetryCache;
    private readonly maxEvents = 100;
    private debounceTimer: NodeJS.Timeout | undefined;
    private readonly debounceMs = 100;
    private isHealthy: boolean = true;
    private lastError: string | undefined;
    private consecutiveErrors: number = 0;
    private readonly maxConsecutiveErrors = 5;

    private _onUpdate = new vscode.EventEmitter<TelemetrySummary>();
    readonly onUpdate = this._onUpdate.event;

    private _onHealthChange = new vscode.EventEmitter<{ healthy: boolean; error?: string }>();
    readonly onHealthChange = this._onHealthChange.event;

    constructor(workspaceRoot: string, maxCacheSize: number = 1000) {
        this.telemetryPath = path.join(workspaceRoot, 'AgentQMS', '.mcp-telemetry.jsonl');
        this.cache = new TelemetryCache(maxCacheSize);
        this.initWatcher();
        this.loadExisting();
    }

    private initWatcher() {
        // Watch for changes to the telemetry file
        const pattern = new vscode.RelativePattern(
            path.dirname(this.telemetryPath),
            '.mcp-telemetry.jsonl'
        );

        this.watcher = vscode.workspace.createFileSystemWatcher(pattern);

        this.watcher.onDidChange(() => this.debouncedHandleFileChange());
        this.watcher.onDidCreate(() => this.handleFileChange());
        this.watcher.onDidDelete(() => this.handleFileDeleted());
    }

    private debouncedHandleFileChange() {
        if (this.debounceTimer) {
            clearTimeout(this.debounceTimer);
        }
        this.debounceTimer = setTimeout(() => this.handleFileChange(), this.debounceMs);
    }

    private handleFileDeleted() {
        console.warn('Telemetry file deleted, resetting state');
        this.lastSize = 0;
        this.lastInode = undefined;
        this.events = [];
        this.setHealth(false, 'Telemetry file deleted');
        this.emitUpdate();
    }

    private async loadExisting() {
        try {
            if (fs.existsSync(this.telemetryPath)) {
                const content = fs.readFileSync(this.telemetryPath, 'utf-8');
                const lines = content.trim().split('\n').filter(l => l);

                // Parse all events and add to cache
                const events = lines.map(line => {
                    try {
                        return JSON.parse(line) as TelemetryEvent;
                    } catch {
                        return null;
                    }
                }).filter((e): e is TelemetryEvent => e !== null);

                // Add to cache
                this.cache.addBatch(events);

                // Keep recent events in memory for quick access
                this.events = this.cache.getRecent(this.maxEvents);

                const stats = fs.statSync(this.telemetryPath);
                this.lastSize = stats.size;
                this.lastInode = stats.ino;

                this.setHealth(true);
                this.emitUpdate();
            } else {
                this.setHealth(false, 'Telemetry file not found');
            }
        } catch (err) {
            this.handleError('Failed to load existing telemetry', err);
        }
    }

    private async handleFileChange() {
        try {
            if (!fs.existsSync(this.telemetryPath)) {
                this.handleFileDeleted();
                return;
            }

            const stats = fs.statSync(this.telemetryPath);

            // Detect file rotation (inode changed or file truncated)
            const fileRotated = this.lastInode !== undefined && stats.ino !== this.lastInode;
            const fileTruncated = stats.size < this.lastSize;

            if (fileRotated || fileTruncated) {
                console.log('File rotation or truncation detected, reloading...');
                await this.loadExisting();
                return;
            }

            if (stats.size > this.lastSize) {
                // Read only new content
                const stream = fs.createReadStream(this.telemetryPath, {
                    start: this.lastSize,
                    encoding: 'utf-8'
                });

                const rl = readline.createInterface({ input: stream });

                for await (const line of rl) {
                    if (line.trim()) {
                        try {
                            const event = JSON.parse(line) as TelemetryEvent;

                            // Add to cache
                            this.cache.add(event);

                            // Update in-memory recent events
                            this.events.push(event);
                            if (this.events.length > this.maxEvents) {
                                this.events.shift();
                            }
                        } catch {
                            // Skip malformed lines
                        }
                    }
                }

                this.lastSize = stats.size;
                this.lastInode = stats.ino;
                this.setHealth(true);
                this.emitUpdate();
            }
        } catch (err) {
            this.handleError('Failed to read telemetry update', err);
            // Attempt recovery by reloading
            setTimeout(() => this.loadExisting(), 1000);
        }
    }

    private emitUpdate() {
        const summary = this.getSummary();
        this._onUpdate.fire(summary);
    }

    getSummary(): TelemetrySummary {
        const success = this.events.filter(e => e.status === 'success').length;
        const errors = this.events.filter(e => e.status === 'error').length;
        const policyViolations = this.events.filter(e => e.status === 'policy_violation').length;

        const durations = this.events
            .filter(e => typeof e.duration_ms === 'number')
            .map(e => e.duration_ms!);

        const avgDurationMs = durations.length > 0
            ? durations.reduce((a, b) => a + b, 0) / durations.length
            : 0;

        return {
            total: this.events.length,
            success,
            errors,
            policyViolations,
            avgDurationMs: Math.round(avgDurationMs * 100) / 100,
            recentCalls: [...this.events].reverse().slice(0, 20)
        };
    }

    public async refresh() {
        await this.loadExisting();
    }

    public getHealth(): { healthy: boolean; error?: string } {
        return {
            healthy: this.isHealthy,
            error: this.lastError
        };
    }

    public getCache(): TelemetryCache {
        return this.cache;
    }

    public getCacheStats() {
        return this.cache.getStats();
    }

    private setHealth(healthy: boolean, error?: string) {
        if (this.isHealthy !== healthy || this.lastError !== error) {
            this.isHealthy = healthy;
            this.lastError = error;

            if (healthy) {
                this.consecutiveErrors = 0;
            }

            this._onHealthChange.fire({ healthy, error });
        }
    }

    private handleError(message: string, err: unknown) {
        const errorMsg = `${message}: ${err}`;
        console.error(errorMsg);

        this.consecutiveErrors++;

        if (this.consecutiveErrors >= this.maxConsecutiveErrors) {
            this.setHealth(false, `Too many errors (${this.consecutiveErrors})`);
        } else {
            // Don't mark as unhealthy for occasional errors
            console.warn(`Error ${this.consecutiveErrors}/${this.maxConsecutiveErrors}: ${errorMsg}`);
        }
    }

    dispose() {
        if (this.debounceTimer) {
            clearTimeout(this.debounceTimer);
        }
        this.watcher?.dispose();
        this._onUpdate.dispose();
        this._onHealthChange.dispose();
    }
}
