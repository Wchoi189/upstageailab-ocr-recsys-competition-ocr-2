import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import { TelemetryWatcher, TelemetrySummary } from './telemetryWatcher';
import { DashboardPanel } from './dashboardPanel';
import { BundleProvider } from './bundleProvider';

let telemetryWatcher: TelemetryWatcher | undefined;
let bundleProvider: BundleProvider | undefined;
let dashboardPanel: DashboardPanel | undefined;

export function activate(context: vscode.ExtensionContext) {
    console.log('MCP Visibility extension activated');

    // Register webview view provider for activity bar
    const provider = new DashboardViewProvider(context.extensionUri, telemetryWatcher, bundleProvider);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider('mcpVisibility.dashboard', provider)
    );

    // Find workspace root for telemetry file
    let workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;

    if (!workspaceRoot) {
        // Fallback: Check parent of extension (development mode or container)
        // If extension is at /a/b/mcp-vis-ext, we check /a/b/.mcp-telemetry.jsonl
        const candidate = path.resolve(context.extensionUri.fsPath, '..');
        const telemetryPath = path.join(candidate, '.mcp-telemetry.jsonl');

        console.log(`Checking fallback path: ${telemetryPath}`);
        if (fs.existsSync(telemetryPath)) {
            console.log('Found telemetry in parent directory, using as workspace root:', candidate);
            workspaceRoot = candidate;
        }
    }

    if (!workspaceRoot) {
        // We still register the provider so the UI doesn't hang,
        // but it will show empty/warning state because watchers are undefined
        vscode.window.showWarningMessage('MCP Visibility: No workspace folder found and telemetry file not detected.');
        return;
    }

    // Initialize watchers
    telemetryWatcher = new TelemetryWatcher(workspaceRoot);
    bundleProvider = new BundleProvider(workspaceRoot);
    context.subscriptions.push(telemetryWatcher);
    context.subscriptions.push(bundleProvider);

    // Update provider with initialized watchers
    provider.setWatchers(telemetryWatcher, bundleProvider);

    // Register show dashboard command
    const showDashboardCmd = vscode.commands.registerCommand('mcpVisibility.showDashboard', () => {
        if (!dashboardPanel) {
            dashboardPanel = new DashboardPanel(context.extensionUri, telemetryWatcher!, bundleProvider);
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
}

class DashboardViewProvider implements vscode.WebviewViewProvider {
    private telemetryWatcher: TelemetryWatcher | undefined;
    private bundleProvider: BundleProvider | undefined;

    constructor(
        private readonly extensionUri: vscode.Uri,
        telemetryWatcher?: TelemetryWatcher,
        bundleProvider?: BundleProvider
    ) {
        this.telemetryWatcher = telemetryWatcher;
        this.bundleProvider = bundleProvider;
    }

    setWatchers(telemetryWatcher: TelemetryWatcher, bundleProvider: BundleProvider) {
        this.telemetryWatcher = telemetryWatcher;
        this.bundleProvider = bundleProvider;
    }

    resolveWebviewView(webviewView: vscode.WebviewView) {
        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this.extensionUri]
        };

        // If watchers aren't ready yet, pass undefined. The Panel handles undefined watchers.
        const panel = new DashboardPanel(
            this.extensionUri,
            this.telemetryWatcher!, // Pass ! but handle inside Panel or ensure we only call if exists
            this.bundleProvider,
            webviewView.webview
        );

        // Listen for telemetry updates only if watcher exists
        if (this.telemetryWatcher) {
            this.telemetryWatcher.onUpdate((data: TelemetrySummary) => {
                panel.updateTelemetry(data);
            });
        }
    }
}
