import * as vscode from 'vscode';
import { TelemetryWatcher, TelemetrySummary } from './telemetryWatcher';
import { BundleProvider, ContextBundle } from './bundleProvider';

export class DashboardPanel implements vscode.Disposable {
    private panel: vscode.WebviewPanel | undefined;
    private webview: vscode.Webview | undefined;
    private disposables: vscode.Disposable[] = [];
    private _onDidDispose = new vscode.EventEmitter<void>();
    readonly onDidDispose = this._onDidDispose.event;

    constructor(
        private readonly extensionUri: vscode.Uri,
        private readonly telemetryWatcher: TelemetryWatcher | undefined,
        private readonly bundleProvider?: BundleProvider,
        existingWebview?: vscode.Webview
    ) {
        if (existingWebview) {
            this.webview = existingWebview;
            this.setupWebview(existingWebview);
        }
    }

    reveal() {
        if (this.panel) {
            this.panel.reveal();
            return;
        }

        this.panel = vscode.window.createWebviewPanel(
            'mcpVisibility.dashboard',
            'MCP Visibility Dashboard',
            vscode.ViewColumn.Two,
            {
                enableScripts: true,
                localResourceRoots: [this.extensionUri],
                retainContextWhenHidden: true
            }
        );

        this.webview = this.panel.webview;
        this.setupWebview(this.panel.webview);

        this.panel.onDidDispose(() => {
            this.panel = undefined;
            this._onDidDispose.fire();
        }, null, this.disposables);
    }

    private async setupWebview(webview: vscode.Webview) {
        webview.html = await this.getHtmlContent();

        if (this.telemetryWatcher) {
            // Listen for telemetry updates
            const updateListener = this.telemetryWatcher.onUpdate((data: TelemetrySummary) => {
                this.updateTelemetry(data);
            });
            this.disposables.push(updateListener);

            // Send initial data
            const initialData = this.telemetryWatcher.getSummary();
            this.updateTelemetry(initialData);
        }

        // Send initial bundles
        if (this.bundleProvider) {
            this.updateBundles(this.bundleProvider.listBundles());
            const bundleListener = this.bundleProvider.onUpdate((bundles: ContextBundle[]) => {
                this.updateBundles(bundles);
            });
            this.disposables.push(bundleListener);
        }

        // Handle messages from webview
        webview.onDidReceiveMessage(
            async (message: { command: string }) => {
                switch (message.command) {
                    case 'refresh':
                        if (this.telemetryWatcher) {
                            await this.telemetryWatcher.refresh();
                            this.updateTelemetry(this.telemetryWatcher.getSummary());
                        }
                        if (this.bundleProvider) {
                            this.bundleProvider.refresh();
                            this.updateBundles(this.bundleProvider.listBundles());
                        }
                        break;
                }
            },
            null,
            this.disposables
        );
    }

    updateTelemetry(data: TelemetrySummary) {
        this.webview?.postMessage({ command: 'update', data });
    }

    updateBundles(bundles: ContextBundle[]) {
        this.webview?.postMessage({ command: 'updateBundles', bundles });
    }

    private async getHtmlContent(): Promise<string> {
        const nonce = getNonce();
        const htmlPath = vscode.Uri.joinPath(this.extensionUri, 'media', 'dashboard.html');

        try {
            const htmlContent = await vscode.workspace.fs.readFile(htmlPath);
            let html = new TextDecoder('utf-8').decode(htmlContent);

            // Replace nonce placeholder
            html = html.replace(/{{nonce}}/g, nonce);
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
