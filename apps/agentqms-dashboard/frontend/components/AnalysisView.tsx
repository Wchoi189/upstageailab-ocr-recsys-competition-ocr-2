import React from 'react';
import { SessionAnalysis, NextStep } from '../types';
import { CheckCircle2, AlertCircle, ArrowRight, Activity, Clock, Target } from 'lucide-react';

interface AnalysisViewProps {
  analysis: SessionAnalysis;
  onExecuteStep: (step: NextStep) => void;
  isGenerating: boolean;
}

export const AnalysisView: React.FC<AnalysisViewProps> = ({ analysis, onExecuteStep, isGenerating }) => {
  return (
    <div className="h-full overflow-y-auto bg-slate-950 p-6 md:p-8 space-y-8">
      
      {/* Header Summary */}
      <div className="space-y-4">
        <div className="flex items-center gap-3 mb-2">
            <div className="bg-emerald-500/10 p-2 rounded-lg">
                <Activity className="w-6 h-6 text-emerald-400" />
            </div>
            <h2 className="text-2xl font-bold text-white">Session Status</h2>
        </div>
        <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 backdrop-blur-sm">
          <p className="text-lg text-slate-200 leading-relaxed">
            {analysis.summary}
          </p>
          <div className="mt-4 flex flex-wrap gap-2">
            {analysis.sentiment && (
                <span className="px-3 py-1 rounded-full text-xs font-medium bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                    Mood: {analysis.sentiment}
                </span>
            )}
             {analysis.keyContextPoints.map((point, idx) => (
              <span key={idx} className="px-3 py-1 rounded-full text-xs font-medium bg-slate-800 text-slate-400 border border-slate-700">
                {point}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Blockers */}
      {analysis.blockers.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wider flex items-center gap-2">
            <AlertCircle className="w-4 h-4" />
            Identified Blockers
          </h3>
          <div className="grid gap-3">
            {analysis.blockers.map((blocker, idx) => (
              <div key={idx} className="flex items-start gap-3 bg-red-950/20 border border-red-900/30 p-4 rounded-xl">
                <div className="mt-1 w-2 h-2 rounded-full bg-red-500 flex-shrink-0" />
                <p className="text-red-200/80 text-sm">{blocker}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Next Steps */}
      <div className="space-y-4 pt-4">
        <div className="flex items-center justify-between">
            <h3 className="text-xl font-semibold text-white flex items-center gap-2">
                <Target className="w-6 h-6 text-blue-400" />
                Recovery Plan
            </h3>
            <span className="text-xs text-slate-500">
                {analysis.suggestedNextSteps.length} Steps
            </span>
        </div>
        
        <div className="grid gap-4">
          {analysis.suggestedNextSteps.map((step, idx) => (
            <div 
              key={idx}
              className="group bg-slate-900 border border-slate-800 hover:border-blue-500/50 p-5 rounded-2xl transition-all duration-300 hover:shadow-lg hover:shadow-blue-900/10"
            >
              <div className="flex justify-between items-start mb-2">
                <div className="flex items-center gap-2">
                    <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wide
                        ${step.priority === 'High' ? 'bg-orange-500/20 text-orange-400' : 
                          step.priority === 'Medium' ? 'bg-blue-500/20 text-blue-400' : 
                          'bg-slate-700 text-slate-400'}`}>
                        {step.priority} Priority
                    </span>
                    <span className="flex items-center gap-1 text-[10px] text-slate-500 bg-slate-800/50 px-2 py-0.5 rounded">
                        <Clock className="w-3 h-3" /> {step.estimatedTime}
                    </span>
                </div>
              </div>
              
              <h4 className="text-lg font-medium text-slate-200 mb-1 group-hover:text-blue-400 transition-colors">
                {step.title}
              </h4>
              <p className="text-slate-400 text-sm mb-4 leading-relaxed">
                {step.description}
              </p>
              
              <button 
                onClick={() => onExecuteStep(step)}
                disabled={isGenerating}
                className="w-full py-2.5 rounded-lg bg-slate-800 hover:bg-blue-600 text-slate-300 hover:text-white text-sm font-medium transition-all flex items-center justify-center gap-2 group-hover:bg-blue-600/10 group-hover:text-blue-400"
              >
                 {isGenerating ? (
                     <span className="animate-pulse">Generating...</span>
                 ) : (
                    <>
                        Start This Step <ArrowRight className="w-4 h-4" />
                    </>
                 )}
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};