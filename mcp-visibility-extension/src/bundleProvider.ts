import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';

export interface ContextBundle {
    name: string;
    description: string;
    fileCount: number;
}

export class BundleProvider implements vscode.Disposable {
    private bundlesPath: string;
    private _onUpdate = new vscode.EventEmitter<ContextBundle[]>();
    readonly onUpdate = this._onUpdate.event;
    private watcher: vscode.FileSystemWatcher | undefined;

    constructor(workspaceRoot: string) {
        this.bundlesPath = path.join(workspaceRoot, 'AgentQMS', '.agentqms', 'plugins', 'context_bundles');
        this.initWatcher();
    }

    private initWatcher() {
        if (!fs.existsSync(this.bundlesPath)) {
            console.warn('Bundle path does not exist:', this.bundlesPath);
            return;
        }

        const pattern = new vscode.RelativePattern(this.bundlesPath, '*.yaml');
        this.watcher = vscode.workspace.createFileSystemWatcher(pattern);

        this.watcher.onDidChange(() => this.emitUpdate());
        this.watcher.onDidCreate(() => this.emitUpdate());
        this.watcher.onDidDelete(() => this.emitUpdate());
    }

    public refresh() {
        this.emitUpdate();
    }

    private emitUpdate() {
        const bundles = this.listBundles();
        this._onUpdate.fire(bundles);
    }

    listBundles(): ContextBundle[] {
        try {
            if (!fs.existsSync(this.bundlesPath)) {
                return [];
            }

            const files = fs.readdirSync(this.bundlesPath);
            const bundles: ContextBundle[] = [];

            for (const file of files) {
                if (!file.endsWith('.yaml')) {
                    continue;
                }

                const filePath = path.join(this.bundlesPath, file);
                const content = fs.readFileSync(filePath, 'utf-8');

                // Simple YAML parsing for key fields
                const nameMatch = content.match(/^name:\s*["']?(.+?)["']?\s*$/m);
                const descMatch = content.match(/^description:\s*["']?(.+?)["']?\s*$/m);
                const filesMatch = content.match(/^\s*-\s+path:.*$/gm);

                const nameStr = nameMatch ? nameMatch[1] : file.replace('.yaml', '');
                const descStr = descMatch ? descMatch[1] : 'No description';
                const count = filesMatch ? filesMatch.length : 0;

                bundles.push({ name: nameStr, description: descStr, fileCount: count });
            }

            return bundles;
        } catch (err) {
            console.error('Failed to list bundles:', err);
            return [];
        }
    }

    dispose() {
        this.watcher?.dispose();
        this._onUpdate.dispose();
    }
}
