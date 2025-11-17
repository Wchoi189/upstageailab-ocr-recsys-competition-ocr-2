import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Home } from "./pages/Home";
import { CommandBuilder } from "./pages/CommandBuilder";
import { Preprocessing } from "./pages/Preprocessing";
import { Inference } from "./pages/Inference";
import { Comparison } from "./pages/Comparison";
import { TestApi } from "./pages/TestApi";
import "./App.css";

function App(): JSX.Element {
  return (
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
  );
}

export default App;
