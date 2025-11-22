import { BrowserRouter, Routes, Route } from "react-router-dom";
import type React from "react";
import { Home } from "./pages/Home";
import { CommandBuilder } from "./pages/CommandBuilder";
import { Preprocessing } from "./pages/Preprocessing";
import { Inference } from "./pages/Inference";
import { Comparison } from "./pages/Comparison";
import { TestApi } from "./pages/TestApi";
import { ErrorBoundary } from "./components/ErrorBoundary";
import "./App.css";

function App(): React.JSX.Element {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/command-builder" element={<CommandBuilder />} />
          <Route path="/preprocessing" element={<Preprocessing />} />
          <Route path="/inference" element={<Inference />} />
          <Route path="/comparison" element={<Comparison />} />
          <Route path="/test-api" element={<TestApi />} />
        </Routes>
      </BrowserRouter>
    </ErrorBoundary>
  );
}

export default App;
