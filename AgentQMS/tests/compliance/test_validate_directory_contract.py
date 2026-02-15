import pytest
from pathlib import Path

from AgentQMS.tools.compliance.validators.directory import validate_directory_placement


def _make_artifact(artifacts_root: Path, subdir: str, filename: str) -> Path:
    target_dir = artifacts_root / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / filename
    file_path.write_text("# test\n")
    return file_path


def test_directory_contract_requires_directory_key(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "docs" / "artifacts"
    artifacts_root.mkdir(parents=True)

    file_path = _make_artifact(
        artifacts_root,
        "assessments",
        "2026-02-14_0000_assessment_test.md",
    )

    valid_artifact_types = {
        "assessment": {"prefix": "assessment_"}
    }

    with pytest.raises(ValueError, match="missing required 'directory' key"):
        validate_directory_placement(
            file_path,
            artifacts_root,
            valid_artifact_types,
            artifact_type_details={},
        )


def test_directory_contract_accepts_dict_and_string(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "docs" / "artifacts"
    artifacts_root.mkdir(parents=True)

    file_path = _make_artifact(
        artifacts_root,
        "assessments",
        "2026-02-14_0000_assessment_test.md",
    )

    dict_types = {
        "assessment": {"prefix": "assessment_", "directory": "assessments/"}
    }
    is_valid, _ = validate_directory_placement(
        file_path,
        artifacts_root,
        dict_types,
        artifact_type_details={},
    )
    assert is_valid

    string_types = {
        "assessment": "assessments/"
    }
    is_valid, _ = validate_directory_placement(
        file_path,
        artifacts_root,
        string_types,
        artifact_type_details={},
    )
    assert is_valid
