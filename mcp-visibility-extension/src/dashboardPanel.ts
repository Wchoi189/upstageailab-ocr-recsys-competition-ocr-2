import * as vscode from 'vscode';
import { TelemetryWatcher, TelemetrySummary } from './telemetryWatcher';
import { BundleProvider, ContextBundle } from './bundleProvider';
import { PolicyProvider, Policy } from './policyProvider';

export class DashboardPanel implements vscode.Disposable {
    public static readonly viewType = 'mcpVisibility.dashboard';
    private panel: vscode.WebviewPanel | undefined;
    private webview: vscode.Webview | undefined;
    private disposables: vscode.Disposable[] = [];
    private _onDidDispose = new vscode.EventEmitter<void>();
    readonly onDidDispose = this._onDidDispose.event;

    constructor(
        private readonly extensionUri: vscode.Uri,
        private readonly telemetryWatcher: TelemetryWatcher | undefined,
        private readonly bundleProvider: BundleProvider | undefined,
        private readonly policyProvider: PolicyProvider | undefined,
        existingWebview?: vscode.Webview
    ) {
        if (existingWebview) {
            this.webview = existingWebview;
            this.initializeWebview(existingWebview);
        }
    }

    reveal() {
        if (this.panel) {
            this.panel.reveal();
            return;
        }

        this.panel = vscode.window.createWebviewPanel(
            DashboardPanel.viewType,
            'AgentQMS Visibility',
            vscode.ViewColumn.Two,
            {
                enableScripts: true,
                localResourceRoots: [this.extensionUri],
                retainContextWhenHidden: true
            }
        );

        this.webview = this.panel.webview;
        this.initializeWebview(this.panel.webview);

        this.panel.onDidDispose(() => {
            this.panel = undefined;
            this._onDidDispose.fire();
        }, null, this.disposables);
    }

    private async initializeWebview(webview: vscode.Webview) {
        webview.html = await this.getHtmlContent();

        // Handle messages from webview
        webview.onDidReceiveMessage(
            async (message: { command: string }) => {
                switch (message.command) {
                    case 'refresh':
                        if (this.telemetryWatcher) {
                            await this.telemetryWatcher.refresh();
                            this.updateTelemetry(this.telemetryWatcher.getSummary());
                            const health = this.telemetryWatcher.getHealth();
                            this.updateConnectionStatus(health.healthy, health.error);
                        }
                        if (this.bundleProvider) {
                            this.bundleProvider.refresh();
                            this.updateBundles(this.bundleProvider.listBundles());
                        }
                        if (this.policyProvider) {
                            this.policyProvider.refresh();
                            this.updatePolicies(this.policyProvider.getPolicies());
                        }
                        break;
                    case 'webviewReady':
                        // Send initial state
                        if (this.telemetryWatcher) {
                            this.updateTelemetry(this.telemetryWatcher.getSummary());
                            const health = this.telemetryWatcher.getHealth();
                            this.updateConnectionStatus(health.healthy, health.error);
                        }
                        if (this.bundleProvider) {
                            this.updateBundles(this.bundleProvider.listBundles());
                        }
                        if (this.policyProvider) {
                            this.updatePolicies(this.policyProvider.getPolicies());
                        }
                        break;
                }
            },
            null,
            this.disposables
        );
    }

    attachWebview(webview: vscode.Webview) {
        this.webview = webview;
        this.initializeWebview(webview);
    }

    updateTelemetry(data: TelemetrySummary) {
        this.webview?.postMessage({ command: 'update', data });
    }

    updateBundles(bundles: ContextBundle[]) {
        this.webview?.postMessage({ command: 'updateBundles', bundles });
    }

    updatePolicies(policies: Policy[]) {
        this.webview?.postMessage({ command: 'updatePolicies', policies });
    }

    updateConnectionStatus(connected: boolean, error?: string) {
        this.webview?.postMessage({
            command: 'connectionStatus',
            connected,
            error,
            timestamp: new Date().toISOString()
        });
    }

    private async getHtmlContent(): Promise<string> {
        const nonce = getNonce();
        const htmlPath = vscode.Uri.joinPath(this.extensionUri, 'media', 'dashboard.html');
        const stylePath = vscode.Uri.joinPath(this.extensionUri, 'media', 'dashboard.css');
        const scriptPath = vscode.Uri.joinPath(this.extensionUri, 'media', 'dashboard.js');

        const styleUri = this.webview?.asWebviewUri(stylePath);
        const scriptUri = this.webview?.asWebviewUri(scriptPath);

        try {
            const htmlContent = await vscode.workspace.fs.readFile(htmlPath);
            let html = new TextDecoder('utf-8').decode(htmlContent);

            // Replace placeholders
            html = html.replace(/{{nonce}}/g, nonce);
            html = html.replace(/{{styleUri}}/g, styleUri ? styleUri.toString() : '');
            html = html.replace(/{{scriptUri}}/g, scriptUri ? scriptUri.toString() : '');
            return html;
        } catch (error) {
            console.error('Failed to load dashboard.html', error);
            return `<!DOCTYPE html><html><body><h1>Error loading dashboard</h1><p>${error}</p></body></html>`;
        }
    }

    dispose() {
        this.panel?.dispose();
        this.disposables.forEach(d => d.dispose());
        this._onDidDispose.dispose();
    }
}

function getNonce(): string {
    let text = '';
    const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    for (let i = 0; i < 32; i++) {
        text += possible.charAt(Math.floor(Math.random() * possible.length));
    }
    return text;
}
