# .qwen/QWEN.md - Project Context for AI Agents

## Project Overview
This is an OCR (Optical Character Recognition) and layout analysis system developed for the Upstage AI Bootcamp OCR competition. The system focuses on Korean text recognition and document layout analysis with an emphasis on agentic observability and high-performance data engineering.

## Key Components
- **OCR Module**: Core text recognition functionality located in `/ocr/`
- **Configurations**: Hydra-based configuration system in `/configs/`
- **Project Compass**: Agentic navigation system in `/project_compass/`
- **AgentQMS**: Quality management system for standards and artifacts in `/AgentQMS/`
- **Experiment Manager**: System for managing experiments in `/experiment_manager/`
- **Agent Debug Toolkit**: AST-based debugging tools in `/agent-debug-toolkit/`

## Architecture
- **Text Recognition**: PARSeq and CRNN architectures
- **Text Detection**: DBNet + PAN deployed on Hugging Face
- **Data Pipeline**: High-performance ETL with LMDB serialization
- **Training Framework**: PyTorch Lightning + Hydra
- **Observability**: Weights & Biases logging

## Important Directories
- `/ocr/` - Main OCR implementation
- `/configs/` - Hydra configuration files
- `/project_compass/` - Agentic navigation system
- `/AgentQMS/` - Quality management and standards
- `/experiment_manager/` - Experiment tracking and management
- `/agent-debug-toolkit/` - AST debugging tools
- `/scripts/` - Utility scripts
- `/docs/` - Documentation
- `/tests/` - Test suite

## Development Standards
- Use `uv` for package management (not pip)
- Follow ADS (Agentic Documentation Standard) v1.0
- Use AgentQMS for artifact creation and validation
- Apply Project Compass vessel/pulse lifecycle for work cycles

## Key Commands
- `uv run python` - Execute Python scripts
- `aqms` - AgentQMS CLI tool
- `adt` - Agent Debug Toolkit
- `make` - Project Makefile targets

## Context Loading Notes
This project uses selective context loading with the patterns defined in `.qwen/settings.json`. The system prioritizes core project files while excluding temporary, generated, or cached files to maintain efficiency.

## File Location Note
This file is intentionally stored in the `.qwen/` directory to maintain a containerized approach for AI agent context management.