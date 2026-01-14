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
    private workspaceRoot: string;
    private _onUpdate = new vscode.EventEmitter<Policy[]>();
    readonly onUpdate = this._onUpdate.event;
    private watcher: vscode.FileSystemWatcher | undefined;

    constructor(workspaceRoot: string) {
        this.workspaceRoot = workspaceRoot;
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

                        // Special handling for Middleware Policies (User Request)
                        if (name === 'middleware' && filePath.includes('middleware-policies.yaml')) {
                            const expanded = this.parseMiddlewarePolicies(filePath);
                            if (expanded.length > 0) {
                                policies.push(...expanded);
                                continue;
                            }
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

    private parseMiddlewarePolicies(relPath: string): Policy[] {
        try {
            const absPath = path.join(this.workspaceRoot, relPath);
            if (!fs.existsSync(absPath)) {
                console.warn('[PolicyProvider] Middleware file not found:', absPath);
                return [];
            }

            const content = fs.readFileSync(absPath, 'utf-8');
            const lines = content.split('\n');
            const policies: Policy[] = [];

            let inItems = false;
            let currentName = '';

            for (const line of lines) {
                // Check if we entered the items section
                if (line.trim() === 'items:') {
                    inItems = true;
                    continue;
                }
                if (!inItems) continue;

                // Match name entry: "  - name: "Something""
                const nameMatch = line.match(/^\s+-\s+name:\s+["']?([^"']+)["']?/);
                if (nameMatch) {
                    currentName = nameMatch[1];
                    policies.push({
                        category: 'Tier 4: Middleware Policies',
                        name: currentName,
                        path: 'AgentQMS/middleware/policies.py'
                    });
                }
            }
            console.log('[PolicyProvider] Parsed middleware policies:', policies.length);
            return policies;
        } catch (error) {
            console.error('[PolicyProvider] Failed to parse middleware policies:', error);
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
