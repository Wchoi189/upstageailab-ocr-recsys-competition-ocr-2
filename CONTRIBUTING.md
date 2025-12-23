# Contributing Guidelines

Thank you for contributing to the OCR Text Recognition & Layout Analysis System. We value professional, high-quality contributions that align with our architectural standards and AI-optimized workflows.

## 📜 Code of Conduct

We are committed to providing a welcoming and inclusive environment. All contributors are expected to:
- Be respectful and professional in all interactions.
- Focus on constructive feedback and collaborative problem-solving.
- Adhere to the [Contributor Covenant](https://www.contributor-covenant.org/) standards.

## 🛠️ Development Standards

### Coding Standards
We enforce strict coding standards to ensure maintainability and AI-readability.
- **Python**: Adhere to PEP 8. Use type hints for all public APIs. Line length: 140 chars.
- **Documentation**: Use Google-style docstrings.
- **Quality**: All new features must include unit tests with >80% coverage.
- **Reference**: See [.ai-instructions/tier2-framework/coding-standards.yaml](.ai-instructions/tier2-framework/coding-standards.yaml) for detailed specifications.

### AgentQMS Integration
This project uses **AgentQMS** for quality management.
- **Artifacts**: Major changes (designs, assessments, bug reports) must be documented as AgentQMS artifacts.
- **Validation**: Run `make validate` in `AgentQMS/interface` before submitting a PR.
- **Tools**: Use the provided VS Code tasks or `make` targets for artifact creation.

## 🔄 Contribution Workflow

1. **Issue First**: Open an issue to discuss major changes before implementation.
2. **Branching**: Use descriptive branch names: `feat/`, `fix/`, `docs/`, or `refactor/`.
3. **Testing**: Ensure all tests pass locally using `uv run pytest`.
4. **Pull Request**:
   - Provide a clear description of changes and link to relevant issues/artifacts.
   - Ensure CI checks pass.
   - Address reviewer feedback promptly.

## 🏗️ Project Structure

- `ocr/`: Core OCR engine and models.
- `AgentQMS/`: Quality management and AI documentation.
- `apps/`: Frontend and inference applications.
- `experiment-tracker/`: Experimental artifacts and reports.

For a detailed overview, see [.ai-instructions/tier1-sst/file-placement-rules.yaml](.ai-instructions/tier1-sst/file-placement-rules.yaml).

---

**Questions?** Open an issue or contact the maintainers.
