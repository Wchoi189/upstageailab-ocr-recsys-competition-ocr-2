
import React, { Component, ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RefreshCw, XCircle } from 'lucide-react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error: Error): State {
    // Update state so the next render will show the fallback UI.
    return { hasError: true, error, errorInfo: null };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // You can also log the error to an error reporting service
    console.error("AgentQMS Component Error:", error, errorInfo);
    this.setState({ errorInfo });
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null, errorInfo: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="h-full w-full flex flex-col items-center justify-center bg-slate-900/50 rounded-xl border border-red-500/20 p-8 animate-in fade-in duration-300">
          <div className="bg-red-500/10 p-4 rounded-full mb-6 border border-red-500/20 shadow-lg shadow-red-900/20">
            <AlertTriangle className="text-red-500 w-12 h-12" />
          </div>
          
          <h2 className="text-2xl font-bold text-white mb-2 tracking-tight">Component Malfunction</h2>
          <p className="text-slate-400 max-w-md text-center mb-8 leading-relaxed">
            The requested feature encountered an unexpected error and has been safely isolated. The rest of AgentQMS remains operational.
          </p>
          
          {this.state.error && (
            <div className="w-full max-w-2xl bg-slate-950 rounded-lg border border-slate-800 mb-8 overflow-hidden shadow-inner">
              <div className="bg-slate-900 px-4 py-2 border-b border-slate-800 flex items-center gap-2">
                 <XCircle size={14} className="text-red-500" />
                 <span className="text-xs font-mono text-slate-400 uppercase tracking-wider">Error Stack Trace</span>
              </div>
              <div className="p-4 overflow-auto max-h-64 custom-scrollbar">
                <p className="text-red-400 font-mono text-sm font-semibold mb-2">
                  {this.state.error.toString()}
                </p>
                {this.state.errorInfo && (
                    <pre className="text-slate-600 font-mono text-xs whitespace-pre-wrap break-all leading-relaxed">
                        {this.state.errorInfo.componentStack}
                    </pre>
                )}
              </div>
            </div>
          )}

          <div className="flex gap-4">
            <button
                onClick={() => window.location.reload()}
                className="px-6 py-2.5 rounded-lg border border-slate-700 text-slate-300 hover:bg-slate-800 hover:text-white transition-all font-medium text-sm"
            >
                Reload Application
            </button>
            <button
                onClick={this.handleReset}
                className="px-6 py-2.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-900/30 transition-all font-medium text-sm flex items-center gap-2"
            >
                <RefreshCw size={16} />
                Try Again
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
