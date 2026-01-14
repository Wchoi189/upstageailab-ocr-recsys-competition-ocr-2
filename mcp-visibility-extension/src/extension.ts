import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import { TelemetryWatcher, TelemetrySummary } from './telemetryWatcher';
import { DashboardPanel } from './dashboardPanel';
import { BundleProvider } from './bundleProvider';
import { PolicyProvider } from './policyProvider';

let telemetryWatcher: TelemetryWatcher | undefined;
let bundleProvider: BundleProvider | undefined;
let policyProvider: PolicyProvider | undefined;
let dashboardPanel: DashboardPanel | undefined;

export function activate(context: vscode.ExtensionContext) {
    console.log('MCP Visibility extension activated');

    // 1. Try Configuration
    const config = vscode.workspace.getConfiguration('mcpVisibility');
    let configPath = config.get<string>('telemetryPath');

    let workspaceRoot: string | undefined;

    if (configPath) {
        // Normalize: We need the parent of 'AgentQMS'
        if (configPath.endsWith('.mcp-telemetry.jsonl')) {
            configPath = path.dirname(configPath);
        }
        if (configPath.endsWith('AgentQMS')) {
            workspaceRoot = path.dirname(configPath);
        } else {
            workspaceRoot = configPath;
        }
        console.log(`Using configured workspace root: ${workspaceRoot}`);
    }

    // 2. Try Workspace Folders
    if (!workspaceRoot && vscode.workspace.workspaceFolders?.length) {
        workspaceRoot = vscode.workspace.workspaceFolders[0].uri.fsPath;
    }

    // 3. Try Fallback (Development/Container structure)
    if (!workspaceRoot) {
        const candidate = path.resolve(context.extensionUri.fsPath, '..');
        const agentQmsPath = path.join(candidate, 'AgentQMS');

        console.log(`Checking fallback path: ${agentQmsPath}`);
        if (fs.existsSync(agentQmsPath) && fs.statSync(agentQmsPath).isDirectory()) {
            console.log('Found AgentQMS directory in parent, using as workspace root:', candidate);
            workspaceRoot = candidate;
        }
    }

    if (!workspaceRoot) {
        vscode.window.showWarningMessage('MCP Visibility: Could not resolve AgentQMS path. Please configure mcpVisibility.telemetryPath or open a workspace.');
        // Still register empty if needed or return? Just continue.
    }

    // Initialize watchers if we have a root
    if (workspaceRoot) {
        telemetryWatcher = new TelemetryWatcher(workspaceRoot);
        bundleProvider = new BundleProvider(workspaceRoot);
        policyProvider = new PolicyProvider(workspaceRoot);
        context.subscriptions.push(telemetryWatcher);
        context.subscriptions.push(bundleProvider);
        context.subscriptions.push(policyProvider);
    }

    // Register webview view provider for activity bar
    const provider = new DashboardViewProvider(context.extensionUri, telemetryWatcher, bundleProvider, policyProvider);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider('mcpVisibility.dashboard', provider),
        provider
    );

    // Register show dashboard command
    const showDashboardCmd = vscode.commands.registerCommand('mcpVisibility.showDashboard', () => {
        if (!dashboardPanel) {
            dashboardPanel = new DashboardPanel(
                context.extensionUri,
                telemetryWatcher,
                bundleProvider,
                policyProvider
            );
            dashboardPanel.onDidDispose(() => {
                dashboardPanel = undefined;
            });
        }
        dashboardPanel.reveal();
    });
    context.subscriptions.push(showDashboardCmd);
}

export function deactivate() {
    telemetryWatcher?.dispose();
    dashboardPanel?.dispose();
    bundleProvider?.dispose();
    policyProvider?.dispose();
}

class DashboardViewProvider implements vscode.WebviewViewProvider {
    private currentPanel: DashboardPanel | undefined;
    private disposables: vscode.Disposable[] = [];

    constructor(
        private readonly extensionUri: vscode.Uri,
        private telemetryWatcher?: TelemetryWatcher,
        private bundleProvider?: BundleProvider,
        private policyProvider?: PolicyProvider
    ) {}

    setWatchers(telemetryWatcher: TelemetryWatcher, bundleProvider: BundleProvider, policyProvider: PolicyProvider) {
        this.telemetryWatcher = telemetryWatcher;
        this.bundleProvider = bundleProvider;
        this.policyProvider = policyProvider;

        // If panel already exists, reconnect it to new watchers
        if (this.currentPanel) {
            this.reconnectListeners();
        }
    }

    resolveWebviewView(webviewView: vscode.WebviewView) {
        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this.extensionUri]
        };

        // Clean up old listeners
        this.disposeListeners();

        // Reuse or create panel
        if (!this.currentPanel) {
            this.currentPanel = new DashboardPanel(
                this.extensionUri,
                this.telemetryWatcher,
                this.bundleProvider,
                this.policyProvider,
                webviewView.webview
            );
        } else {
            // Reattach to new webview instance
            this.currentPanel.attachWebview(webviewView.webview);
        }

        // Register listeners with proper cleanup tracking
        this.reconnectListeners();

        // Handle webview disposal
        webviewView.onDidDispose(() => {
            this.disposeListeners();
        });
    }

    private reconnectListeners() {
        // Clear existing listeners
        this.disposeListeners();

        if (!this.currentPanel) {
            return;
        }

        // Register telemetry updates
        if (this.telemetryWatcher) {
            const telemetryListener = this.telemetryWatcher.onUpdate((data: TelemetrySummary) => {
                this.currentPanel?.updateTelemetry(data);
            });
            this.disposables.push(telemetryListener);

            // Send initial telemetry data
            const initialData = this.telemetryWatcher.getSummary();
            this.currentPanel.updateTelemetry(initialData);

            // Register health updates
            const healthListener = this.telemetryWatcher.onHealthChange((status) => {
                this.currentPanel?.updateConnectionStatus(status.healthy, status.error);
            });
            this.disposables.push(healthListener);

             // Send connection health status
            const health = this.telemetryWatcher.getHealth();
            this.currentPanel.updateConnectionStatus(health.healthy, health.error);
        }

        // Register bundle updates if provider exists
        if (this.bundleProvider) {
            const bundleListener = this.bundleProvider.onUpdate((bundles) => {
                this.currentPanel?.updateBundles(bundles);
            });
            this.disposables.push(bundleListener);

            // Send initial bundle data
            this.currentPanel.updateBundles(this.bundleProvider.listBundles());
        }

        // Register policy updates
        if (this.policyProvider) {
            const policyListener = this.policyProvider.onUpdate((policies) => {
                this.currentPanel?.updatePolicies(policies);
            });
            this.disposables.push(policyListener);

            // Send initial policy data
            this.currentPanel.updatePolicies(this.policyProvider.getPolicies());
        }
    }

    private disposeListeners() {
        this.disposables.forEach(d => d.dispose());
        this.disposables = [];
    }

    dispose() {
        this.disposeListeners();
        this.currentPanel?.dispose();
    }
}
