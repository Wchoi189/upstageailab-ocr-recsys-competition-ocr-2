import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';

export interface Policy {
    category: string;
    name: string;
    path: string;
}

export class PolicyProvider implements vscode.Disposable {
    private standardsPath: string;
    private _onUpdate = new vscode.EventEmitter<Policy[]>();
    readonly onUpdate = this._onUpdate.event;
    private watcher: vscode.FileSystemWatcher | undefined;

    constructor(workspaceRoot: string) {
        this.standardsPath = path.join(workspaceRoot, 'AgentQMS', 'standards', 'INDEX.yaml');
        this.initWatcher(workspaceRoot);
    }

    private initWatcher(root: string) {
        const pattern = new vscode.RelativePattern(path.join(root, 'AgentQMS', 'standards'), 'INDEX.yaml');
        this.watcher = vscode.workspace.createFileSystemWatcher(pattern);

        this.watcher.onDidChange(() => this.emitUpdate());
        this.watcher.onDidCreate(() => this.emitUpdate());
        this.watcher.onDidDelete(() => this.emitUpdate());
    }

    public refresh() {
        this.emitUpdate();
    }

    private emitUpdate() {
        const policies = this.getPolicies();
        this._onUpdate.fire(policies);
    }

    getPolicies(): Policy[] {
        try {
            if (!fs.existsSync(this.standardsPath)) {
                return [];
            }

            const content = fs.readFileSync(this.standardsPath, 'utf-8');
            const policies: Policy[] = [];

            const lines = content.split('\n');
            let currentTier = '';

            // Tier mapping for nicer display
            const tierNames: { [key: string]: string } = {
                'tier1_sst': 'Tier 1: Single Source of Truth',
                'tier2_framework': 'Tier 2: Framework Standards',
                'tier3_agents': 'Tier 3: Agent Configurations',
                'tier4_workflows': 'Tier 4: Workflows & Compliance'
            };

            for (const line of lines) {
                // Check for Tier headers (e.g., "tier1_sst:")
                const tierMatch = line.match(/^(\w+):/);
                if (tierMatch) {
                    const key = tierMatch[1];
                    if (key.startsWith('tier')) {
                        currentTier = tierNames[key] || key;
                    } else if (key === 'root_map' || key === 'glob_patterns') {
                        currentTier = ''; // Reset for metadata sections
                    }
                    continue;
                }

                // Check for Policy entries (e.g., "  architecture: ...")
                if (currentTier) {
                    const policyMatch = line.match(/^\s+(\w+):\s+(.+)/);
                    if (policyMatch) {
                        const name = policyMatch[1];
                        const filePath = policyMatch[2];

                        // Ignore sub-properties or simple lists for now
                        if (!filePath.startsWith('AgentQMS')) {
                            continue;
                        }

                        policies.push({
                            category: currentTier,
                            name: this.formatName(name),
                            path: filePath
                        });
                    }
                }
            }

            return policies;
        } catch (err) {
            console.error('Failed to parse policies:', err);
            return [];
        }
    }

    private formatName(key: string): string {
        // defined_name -> Defined Name
        return key.split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }

    dispose() {
        this.watcher?.dispose();
        this._onUpdate.dispose();
    }
}
