# Implementation Plan - Multi-Agent IACP Migration

## Goal Description
Initiate a new **Project Compass Vessel** for the **Multi-Agent IACP migration**. The primary focus is on **State Virtualization** to reduce `effective.yaml` bloat and establishing a robust implementation plan for multi-session execution.

## User Review Required
> [!IMPORTANT]
> This plan involves significant changes to how configuration is verified and stored.
> - **Virtual Config**: `effective.yaml` will no longer be physically written to disk by default.
> - **Archive**: Stale walkthroughs and plans will be moved to `.archive/`.

## Proposed Changes

### Configuration Management
#### [MODIFY] [config_loader.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/AgentQMS/tools/utils/config_loader.py)
- Implement `generate_virtual_config` to return config as a dictionary without writing to disk.
- Add Redis caching support (optional/conditional) for effective config.

#### [MODIFY] [cli.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/AgentQMS/cli.py)
- Update `generate-config` command to support `--json` flag for direct AI ingestion.
- Default to not writing `effective.yaml` unless explicitly requested.

### Inter-Agent Communication Protocol (IACP)
#### [NEW] [iacp_schemas.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/infrastructure/communication/iacp_schemas.py)
- Define `IACPEnvelope` Pydantic model for strict message validation.

#### [MODIFY] [rabbitmq_transport.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/AgentQMS/tools/multi_agent/rabbitmq_transport.py)
- Integrate `IACPEnvelope` validation into `send_command` and `start_listening`.

### Maintenance
#### [NEW] [janitor.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/AgentQMS/bin/janitor.py)
- Implement cleanup script to archive stale artifacts.

## Verification Plan

### Automated Tests
- **Config Virtualization Test**:
    1. Run `aqms generate-config --json`.
    2. Verify output is valid JSON and no `effective.yaml` is created.
- **IACP Schema Test**:
    1. Create a unit test that attempts to create an invalid `IACPEnvelope` and asserts validation failure.
    2. Serialize/deserialize a valid envelope.

### Manual Verification
- **Janitor Dry Run**:
    1. Run `python AgentQMS/bin/janitor.py --dry-run`.
    2. Verify it identifies the correct files to archive.
