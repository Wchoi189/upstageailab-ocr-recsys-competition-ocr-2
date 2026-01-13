import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import * as readline from 'readline';

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
    private events: TelemetryEvent[] = [];
    private readonly maxEvents = 100;

    private _onUpdate = new vscode.EventEmitter<TelemetrySummary>();
    readonly onUpdate = this._onUpdate.event;

    constructor(workspaceRoot: string) {
        this.telemetryPath = path.join(workspaceRoot, '.mcp-telemetry.jsonl');
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

        this.watcher.onDidChange(() => this.handleFileChange());
        this.watcher.onDidCreate(() => this.handleFileChange());
    }

    private async loadExisting() {
        try {
            if (fs.existsSync(this.telemetryPath)) {
                const content = fs.readFileSync(this.telemetryPath, 'utf-8');
                const lines = content.trim().split('\n').filter(l => l);

                // Take last maxEvents
                const recentLines = lines.slice(-this.maxEvents);
                this.events = recentLines.map(line => {
                    try {
                        return JSON.parse(line) as TelemetryEvent;
                    } catch {
                        return null;
                    }
                }).filter((e): e is TelemetryEvent => e !== null);

                this.lastSize = fs.statSync(this.telemetryPath).size;
                this.emitUpdate();
            }
        } catch (err) {
            console.error('Failed to load existing telemetry:', err);
        }
    }

    private async handleFileChange() {
        try {
            const stats = fs.statSync(this.telemetryPath);

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
                            this.events.push(event);

                            // Trim old events
                            if (this.events.length > this.maxEvents) {
                                this.events.shift();
                            }
                        } catch {
                            // Skip malformed lines
                        }
                    }
                }

                this.lastSize = stats.size;
                this.emitUpdate();
            }
        } catch (err) {
            console.error('Failed to read telemetry update:', err);
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

    dispose() {
        this.watcher?.dispose();
        this._onUpdate.dispose();
    }
}
