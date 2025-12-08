import { GoogleGenAI, Type } from "@google/genai";
import { SessionAnalysis, NextStep } from '../types';

// Initialize the Gemini API client
// Note: We create a new instance per call in the components to ensure the latest API key is used if it changes
// but for simplicity in this service helper, we will accept the key or instance as needed.

export const analyzeSessionContext = async (
  contextText: string,
  apiKey: string
): Promise<SessionAnalysis> => {
  const ai = new GoogleGenAI({ apiKey });

  // Schema for structured output
  const analysisSchema = {
    type: Type.OBJECT,
    properties: {
      summary: {
        type: Type.STRING,
        description: "A concise summary of the current session state based on the input.",
      },
      keyContextPoints: {
        type: Type.ARRAY,
        items: { type: Type.STRING },
        description: "List of 3-5 critical facts or context items extracted from the input.",
      },
      blockers: {
        type: Type.ARRAY,
        items: { type: Type.STRING },
        description: "Potential reasons why the session stalled or became over-extended.",
      },
      sentiment: {
        type: Type.STRING,
        description: "The detected mood of the session (e.g., 'Frustrated', 'Productive but tired', 'Chaotic').",
      },
      suggestedNextSteps: {
        type: Type.ARRAY,
        items: {
          type: Type.OBJECT,
          properties: {
            title: { type: Type.STRING },
            description: { type: Type.STRING },
            priority: { type: Type.STRING, enum: ['High', 'Medium', 'Low'] },
            estimatedTime: { type: Type.STRING, description: "e.g., '15 mins'" },
          },
          required: ['title', 'description', 'priority', 'estimatedTime'],
        },
        description: "A list of actionable steps to resume the session effectively.",
      },
    },
    required: ['summary', 'keyContextPoints', 'blockers', 'suggestedNextSteps', 'sentiment'],
  };

  const response = await ai.models.generateContent({
    model: 'gemini-2.5-flash',
    contents: `The user has an 'over-extended' session. They are likely tired, confused, or have too much data.
    Analyze the following input text which represents their current state (notes, logs, code, or brain dump).
    Structure the output to help them regain clarity and resume work.
    
    Input Context:
    """
    ${contextText}
    """`,
    config: {
      responseMimeType: 'application/json',
      responseSchema: analysisSchema,
      systemInstruction: "You are an expert technical project manager and productivity coach skilled in decluttering complex workflows.",
    },
  });

  const text = response.text;
  if (!text) throw new Error("No response from Gemini");
  
  return JSON.parse(text) as SessionAnalysis;
};

export const generateStepContent = async (
  step: NextStep,
  contextText: string,
  apiKey: string
): Promise<string> => {
  const ai = new GoogleGenAI({ apiKey });
  
  const response = await ai.models.generateContent({
    model: 'gemini-2.5-flash', // Using Flash for speed, switch to Pro if complex coding is needed
    contents: `Context:
    ${contextText.slice(0, 2000)}... [truncated]
    
    Task:
    The user wants to execute the following step to resume their session:
    Title: ${step.title}
    Description: ${step.description}
    
    Please generate the actual content, code, or text draft for this step.`,
    config: {
      systemInstruction: "You are a helpful assistant executing a specific task to help the user resume work.",
    },
  });

  return response.text || "Could not generate content.";
};