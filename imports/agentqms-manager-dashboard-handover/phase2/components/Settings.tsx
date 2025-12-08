
import React, { useState, useEffect, useRef } from 'react';
import { Save, Key, Server, Cpu, CheckCircle, Upload, Download, FileJson, FileCode } from 'lucide-react';
import { AIProvider, AppSettings } from '../types';
import { getSettings, saveSettings } from '../services/aiService';

const Settings: React.FC = () => {
  const [settings, setSettings] = useState<AppSettings>({
    provider: AIProvider.GEMINI,
    apiKey: '',
    model: 'gemini-2.5-flash',
    baseUrl: ''
  });
  const [saved, setSaved] = useState(false);
  const [message, setMessage] = useState('');
  
  const jsonInputRef = useRef<HTMLInputElement>(null);
  const envInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setSettings(getSettings());
  }, []);

  const handleSave = () => {
    saveSettings(settings);
    setSaved(true);
    setMessage('Configuration saved successfully.');
    setTimeout(() => {
        setSaved(false);
        setMessage('');
    }, 2000);
  };

  const handleChange = (field: keyof AppSettings, value: string) => {
    setSettings(prev => ({ ...prev, [field]: value }));
  };

  // Export Settings to JSON
  const handleExport = () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(settings, null, 2));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", "agentqms-config.json");
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
    setMessage('Config exported.');
    setTimeout(() => setMessage(''), 2000);
  };

  // Import JSON Config
  const handleImportJson = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
        try {
            const parsed = JSON.parse(event.target?.result as string);
            // Basic validation
            if (parsed.provider && parsed.model) {
                setSettings(prev => ({ ...prev, ...parsed }));
                setMessage('Configuration imported.');
            } else {
                alert("Invalid configuration file.");
            }
        } catch (err) {
            alert("Error parsing JSON.");
        }
    };
    reader.readAsText(file);
    if (jsonInputRef.current) jsonInputRef.current.value = ''; // Reset
  };

  // Import .env file
  const handleImportEnv = (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = (event) => {
          const content = event.target?.result as string;
          const lines = content.split('\n');
          let foundKey = '';
          
          // Simple .env parser strategy
          for (const line of lines) {
              const trimmed = line.trim();
              if (trimmed.startsWith('#') || !trimmed.includes('=')) continue;
              
              const [key, ...valParts] = trimmed.split('=');
              const val = valParts.join('=').replace(/^['"]|['"]$/g, '').trim(); // Remove quotes

              if (key === 'API_KEY' || key === 'GOOGLE_API_KEY' || key === 'GEMINI_API_KEY') {
                  foundKey = val;
                  // If we found a generic key, assume generic provider if not set
                  if (!settings.apiKey) handleChange('provider', AIProvider.GEMINI);
              } else if (key === 'OPENAI_API_KEY') {
                  foundKey = val;
                  handleChange('provider', AIProvider.OPENAI);
              }
          }

          if (foundKey) {
              handleChange('apiKey', foundKey);
              setMessage('API Key extracted from .env');
          } else {
              alert("No supported API keys (API_KEY, OPENAI_API_KEY) found in .env file.");
          }
      };
      reader.readAsText(file);
      if (envInputRef.current) envInputRef.current.value = ''; // Reset
  };

  return (
    <div className="h-full max-w-3xl mx-auto overflow-y-auto pr-2">
      <div className="mb-6 flex justify-between items-end">
        <div>
            <h2 className="text-2xl font-bold text-white mb-2">Framework Settings</h2>
            <p className="text-slate-400">Configure your AI Provider and Framework preferences.</p>
        </div>
        <div className="flex gap-2">
            <input type="file" ref={jsonInputRef} onChange={handleImportJson} accept=".json" className="hidden" />
            <input type="file" ref={envInputRef} onChange={handleImportEnv} accept=".env,text/plain" className="hidden" />
            
            <button onClick={() => jsonInputRef.current?.click()} className="bg-slate-800 hover:bg-slate-700 text-slate-300 px-3 py-2 rounded flex items-center gap-2 text-sm border border-slate-700 transition">
                <Upload size={14} /> Import Config
            </button>
            <button onClick={() => envInputRef.current?.click()} className="bg-slate-800 hover:bg-slate-700 text-slate-300 px-3 py-2 rounded flex items-center gap-2 text-sm border border-slate-700 transition">
                <FileCode size={14} /> Import .env
            </button>
            <button onClick={handleExport} className="bg-slate-800 hover:bg-slate-700 text-slate-300 px-3 py-2 rounded flex items-center gap-2 text-sm border border-slate-700 transition">
                <Download size={14} /> Export
            </button>
        </div>
      </div>

      <div className="bg-slate-800 rounded-xl border border-slate-700 p-6 shadow-xl mb-6">
        <div className="mb-6 pb-6 border-b border-slate-700">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <Server className="text-blue-400" size={20} />
                AI Provider Configuration
            </h3>
            
            <div className="grid gap-6">
                <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2">Select Provider</label>
                    <div className="grid grid-cols-3 gap-3">
                        {Object.values(AIProvider).map((p) => (
                            <button
                                key={p}
                                onClick={() => handleChange('provider', p)}
                                className={`py-3 px-4 rounded-lg border text-sm font-semibold transition-all ${
                                    settings.provider === p 
                                    ? 'bg-blue-600 border-blue-500 text-white shadow-lg shadow-blue-900/20' 
                                    : 'bg-slate-900 border-slate-700 text-slate-400 hover:bg-slate-800'
                                }`}
                            >
                                {p}
                            </button>
                        ))}
                    </div>
                </div>

                <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2 flex items-center gap-2">
                        <Key size={16} className="text-yellow-400" /> API Key
                    </label>
                    <input 
                        type="password"
                        value={settings.apiKey}
                        onChange={(e) => handleChange('apiKey', e.target.value)}
                        placeholder="sk-..."
                        className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-blue-500 font-mono"
                    />
                    <p className="text-xs text-slate-500 mt-2">
                        Keys are stored locally in browser. {settings.provider === AIProvider.GEMINI && "Default uses process.env.API_KEY if left empty."}
                    </p>
                </div>

                <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2 flex items-center gap-2">
                        <Cpu size={16} className="text-purple-400" /> Model ID
                    </label>
                    <input 
                        type="text"
                        value={settings.model}
                        onChange={(e) => handleChange('model', e.target.value)}
                        placeholder={settings.provider === AIProvider.GEMINI ? "gemini-2.5-flash" : "gpt-4-turbo"}
                        className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-blue-500 font-mono"
                    />
                </div>

                {settings.provider !== AIProvider.GEMINI && (
                     <div>
                        <label className="block text-sm font-medium text-slate-300 mb-2">Base URL (Optional)</label>
                        <input 
                            type="text"
                            value={settings.baseUrl}
                            onChange={(e) => handleChange('baseUrl', e.target.value)}
                            placeholder="https://api.example.com/v1"
                            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-blue-500 font-mono"
                        />
                         <p className="text-xs text-slate-500 mt-2">Use for custom proxies or Alibaba/Qwen compatible endpoints.</p>
                    </div>
                )}
            </div>
        </div>

        <div className="flex items-center justify-between">
            <span className={`text-sm transition-opacity ${message ? 'opacity-100 text-green-400' : 'opacity-0'}`}>
                {message}
            </span>
            <button
                onClick={handleSave}
                className="bg-blue-600 hover:bg-blue-500 text-white px-6 py-3 rounded-lg font-semibold flex items-center gap-2 shadow-lg shadow-blue-900/30 transition-colors"
            >
                {saved ? <CheckCircle size={18} /> : <Save size={18} />}
                {saved ? 'Saved' : 'Save Configuration'}
            </button>
        </div>
      </div>
    </div>
  );
};

export default Settings;
