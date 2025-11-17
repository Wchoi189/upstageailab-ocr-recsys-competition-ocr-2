# Contributing to OCR Text Detection & Recognition System

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Process](#contributing-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)

## 📜 Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of experience level, gender, gender identity and expression, sexual orientation, disability, personal appearance, body size, race, ethnicity, age, religion, or nationality.

### Expected Behavior

- Be respectful and considerate
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Show empathy towards others

### Unacceptable Behavior

- Harassment or discriminatory language
- Trolling or insulting comments
- Public or private harassment
- Publishing others' private information

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- UV package manager
- Git
- CUDA-compatible GPU (recommended for development)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/upstageailab-ocr-recsys-competition-ocr-2.git
   cd upstageailab-ocr-recsys-competition-ocr-2
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/upstageailab-ocr-recsys-competition-ocr-2.git
   ```

## 🛠️ Development Setup

### Initial Setup

```bash
# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup environment
./scripts/setup/00_setup-environment.sh

# Verify installation
uv run pytest tests/ -v
```

### Development Workflow

1. **Update your fork**: Keep your fork up to date with upstream
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

3. **Make your changes**: Follow the coding standards below

4. **Test your changes**:
   ```bash
   # Run all tests
   uv run pytest tests/ -v

   # Run specific test file
   uv run pytest tests/test_your_feature.py -v

   # Run with coverage
   uv run pytest tests/ --cov=ocr --cov-report=html
   ```

5. **Check code quality**:
   ```bash
   # Linting (if configured)
   uv run ruff check .

   # Type checking (if configured)
   uv run mypy ocr/
   ```

## 🔄 Contributing Process

### Types of Contributions

We welcome various types of contributions:

- **Bug Fixes**: Fix issues and improve stability
- **Features**: Add new functionality
- **Documentation**: Improve docs, add examples
- **Tests**: Add or improve test coverage
- **Performance**: Optimize code and improve efficiency
- **Refactoring**: Improve code structure and maintainability

### Finding Issues to Work On

- Check [GitHub Issues](https://github.com/ORIGINAL_OWNER/upstageailab-ocr-recsys-competition-ocr-2/issues)
- Look for issues labeled `good first issue` for beginners
- Check the [Roadmap](README.md#-roadmap) for planned features

### Before You Start

1. **Check existing issues**: Make sure the issue/feature isn't already being worked on
2. **Comment on the issue**: Let others know you're working on it
3. **Ask questions**: If something is unclear, ask in the issue discussion

## 📝 Coding Standards

**See [CODING_STANDARDS.md](docs/CODING_STANDARDS.md) for comprehensive coding standards covering Python, TypeScript/React, code length guidelines, and best practices.**

### Quick Reference

- **Python**: 140 character line length, type hints required, Ruff for linting/formatting
- **TypeScript/React**: 100 character line length, explicit types required, Prettier/ESLint
- **Function length**: Python < 50 lines (target), TypeScript < 40 lines (target)
- **File length**: Python < 500 lines (target), TypeScript < 300 lines (target)

### Code Formatting

```python
# ✅ Good
def process_image(
    image: np.ndarray,
    preprocessing_config: Dict[str, Any],
    device: torch.device = torch.device("cpu")
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Process an image with the given preprocessing configuration.

    Args:
        image: Input image as numpy array
        preprocessing_config: Dictionary containing preprocessing parameters
        device: Device to run processing on

    Returns:
        Tuple of processed image and metadata dictionary
    """
    # Implementation
    pass

# ❌ Bad
def proc(img, cfg, dev="cpu"):
    # Implementation
    pass
```

### Import Organization

```python
# Standard library imports
import os
import sys
from typing import Dict, List, Optional

# Third-party imports
import numpy as np
import torch
from torch import nn

# Local imports
from ocr.models import BaseEncoder
from ocr.utils import image_utils
```

### Documentation

- Use docstrings for all public functions and classes
- Follow Google or NumPy docstring style
- Include type information in docstrings
- Add comments for complex logic

```python
class TextDetector:
    """
    Text detection model based on DBNet architecture.

    This class provides an interface for detecting text regions
    in images using a differentiable binarization approach.

    Attributes:
        model: The underlying PyTorch model
        device: Device to run inference on
        confidence_threshold: Minimum confidence for detections
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize the text detector.

        Args:
            model_path: Path to the model checkpoint
            device: Device to load model on ('cuda' or 'cpu')
        """
        pass
```

## 🧪 Testing Guidelines

### Writing Tests

- Write tests for all new features
- Aim for >80% code coverage
- Use descriptive test names
- Follow Arrange-Act-Assert pattern

```python
# ✅ Good test
def test_text_detector_processes_image_correctly():
    """Test that text detector correctly processes input image."""
    # Arrange
    detector = TextDetector(model_path="test_model.ckpt")
    test_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)

    # Act
    results = detector.detect(test_image)

    # Assert
    assert isinstance(results, list)
    assert all("bbox" in r for r in results)
    assert all("confidence" in r for r in results)

# ❌ Bad test
def test_detector():
    detector = TextDetector("model.ckpt")
    results = detector.detect(img)
    assert results
```

### Test Organization

- Unit tests: `tests/unit/`
- Integration tests: `tests/integration/`
- Test files should mirror source structure: `tests/ocr/models/test_encoder.py` for `ocr/models/encoder.py`

### Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_metrics.py

# Run with verbose output
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=ocr --cov-report=term-missing

# Run only fast tests
uv run pytest tests/ -m "not slow"
```

## 📚 Documentation

### Code Documentation

- Document all public APIs
- Include usage examples in docstrings
- Update docstrings when code changes

### Documentation Files

- Update README.md if adding major features
- Add examples to `docs/examples/` if applicable
- Update API reference if adding new modules

### Writing Documentation

- Use clear, concise language
- Include code examples
- Add diagrams for complex concepts
- Keep documentation up to date with code

## 💬 Commit Guidelines

### Commit Message Format

We follow a conventional commit format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples

```bash
# Good commit messages
feat(ui): add model comparison dashboard
fix(detector): resolve memory leak in batch processing
docs(readme): update installation instructions
refactor(models): simplify encoder architecture
test(metrics): add tests for CLEval implementation

# Bad commit messages
fix bug
update code
changes
```

### Commit Best Practices

- Make atomic commits (one logical change per commit)
- Write clear, descriptive commit messages
- Reference issues in commit messages: `fix(#123): resolve memory issue`

## 🔀 Pull Request Process

### Before Submitting

1. **Update your branch**: Rebase on latest main
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all checks**:
   ```bash
   # Tests
   uv run pytest tests/ -v

   # Code quality
   uv run ruff check .

   # Type checking
   uv run mypy ocr/
   ```

3. **Update documentation**: Ensure all changes are documented

4. **Check for conflicts**: Resolve any merge conflicts

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Commit messages follow guidelines
- [ ] No merge conflicts
- [ ] PR description is clear and complete

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
How was this tested?

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Code follows style guidelines
```

### Review Process

1. **Automated checks**: CI will run tests and checks
2. **Code review**: Maintainers will review your code
3. **Feedback**: Address any requested changes
4. **Approval**: Once approved, your PR will be merged

### After Approval

- Your PR will be merged by a maintainer
- Delete your feature branch after merge
- Update your fork's main branch

## 🐛 Reporting Bugs

### Before Reporting

1. Check if the bug has already been reported
2. Try to reproduce the issue
3. Check documentation and existing issues

### Bug Report Template

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Run command '...'
2. See error

**Expected behavior**
What you expected to happen.

**Environment:**
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.10.8]
- PyTorch version: [e.g., 2.8.0]
- GPU: [e.g., NVIDIA RTX 3090]

**Error messages/logs**
Paste relevant error messages or logs.

**Additional context**
Any other relevant information.
```

## 💡 Suggesting Features

### Feature Request Template

```markdown
**Is your feature request related to a problem?**
A clear description of the problem.

**Describe the solution you'd like**
What you want to happen.

**Describe alternatives you've considered**
Other solutions you've thought about.

**Additional context**
Any other relevant information, mockups, examples, etc.
```

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and discussions
- **Documentation**: Check the [docs](docs/) directory

## 🙏 Thank You!

Your contributions make this project better for everyone. Thank you for taking the time to contribute!

---

**Note**: This is a living document. If you have suggestions for improving these guidelines, please open an issue or submit a PR!

