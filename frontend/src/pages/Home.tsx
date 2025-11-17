import { Link } from "react-router-dom";

export function Home(): JSX.Element {
  return (
    <div style={{ padding: "2rem" }}>
      <h1>OCR Playground</h1>
      <p>High-performance image processing and model training toolkit</p>

      <nav style={{ marginTop: "2rem" }}>
        <ul style={{ listStyle: "none", padding: 0 }}>
          <li style={{ marginBottom: "1rem" }}>
            <Link to="/command-builder">Command Builder</Link>
          </li>
          <li style={{ marginBottom: "1rem" }}>
            <Link to="/preprocessing">Preprocessing Studio</Link>
          </li>
          <li style={{ marginBottom: "1rem" }}>
            <Link to="/inference">Inference Studio</Link>
          </li>
          <li style={{ marginBottom: "1rem" }}>
            <Link to="/comparison">Comparison Studio</Link>
          </li>
          <li style={{ marginBottom: "1rem" }}>
            <Link to="/test-api">Test API Connection</Link>
          </li>
        </ul>
      </nav>
    </div>
  );
}
