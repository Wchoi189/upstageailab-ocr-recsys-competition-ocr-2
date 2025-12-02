#!/usr/bin/env python3
"""
Unit tests for uppercase validation in validate_artifacts.py
Tests the fix for Bug 2: uppercase validation skip for uppercase_prefix artifact types.
"""
import pytest
from validate_artifacts import ArtifactValidator


class TestUppercasePrefixValidation:
    """Test uppercase validation for artifacts with uppercase_prefix case type."""
    
    @pytest.fixture
    def validator(self, tmp_path):
        """Create a validator instance with a temporary artifacts directory."""
        artifacts_root = tmp_path / "docs" / "artifacts"
        artifacts_root.mkdir(parents=True)
        
        # Create required subdirectories
        (artifacts_root / "bug_reports").mkdir()
        (artifacts_root / "completed_plans" / "completion_summaries" / "session_notes").mkdir(parents=True)
        (artifacts_root / "assessments").mkdir()
        
        return ArtifactValidator(str(artifacts_root))
    
    def test_bug_report_lowercase_descriptive_passes(self, validator, tmp_path):
        """Test that BUG_ files with lowercase descriptive part pass validation."""
        # Valid: BUG_001_lowercase-name.md
        file_path = tmp_path / "docs" / "artifacts" / "bug_reports" / "2025-11-30_1200_BUG_001_lowercase-name.md"
        file_path.write_text("# Test Bug Report\n")
        
        is_valid, message = validator.validate_naming_convention(file_path)
        
        assert is_valid, f"Expected valid but got: {message}"
        assert message == "Valid naming convention"
    
    def test_bug_report_uppercase_descriptive_fails(self, validator, tmp_path):
        """Test that BUG_ files with uppercase descriptive part fail validation."""
        # Invalid: BUG_001_ALL-CAPS.md
        file_path = tmp_path / "docs" / "artifacts" / "bug_reports" / "2025-11-30_1200_BUG_001_ALL-CAPS.md"
        file_path.write_text("# Test Bug Report\n")
        
        is_valid, message = validator.validate_naming_convention(file_path)
        
        assert not is_valid, "Expected invalid for uppercase descriptive part"
        assert "uppercase prefix must have lowercase descriptive part" in message.lower()
        assert "BUG_001_" in message
        assert "lowercase-name.md" in message
    
    def test_bug_report_mixed_case_descriptive_fails(self, validator, tmp_path):
        """Test that BUG_ files with mixed case descriptive part fail validation."""
        # Invalid: BUG_001_Mixed-Case.md
        file_path = tmp_path / "docs" / "artifacts" / "bug_reports" / "2025-11-30_1200_BUG_001_Mixed-Case.md"
        file_path.write_text("# Test Bug Report\n")
        
        is_valid, message = validator.validate_naming_convention(file_path)
        
        assert not is_valid, "Expected invalid for mixed case descriptive part"
        assert "uppercase prefix must have lowercase descriptive part" in message
    
    def test_session_note_lowercase_descriptive_passes(self, validator, tmp_path):
        """Test that SESSION_ files with lowercase descriptive part pass validation."""
        # Valid: SESSION_sprint-review.md
        file_path = (tmp_path / "docs" / "artifacts" / "completed_plans" / 
                     "completion_summaries" / "session_notes" /
                     "2025-11-30_1200_SESSION_sprint-review.md")
        file_path.write_text("# Test Session Note\n")
        
        is_valid, message = validator.validate_naming_convention(file_path)
        
        assert is_valid, f"Expected valid, but got: {message}"
        assert "Valid naming convention" in message
    
    def test_session_note_uppercase_descriptive_fails(self, validator, tmp_path):
        """Test that SESSION_ files with uppercase descriptive part fail validation."""
        # Invalid: SESSION_ALL-CAPS.md
        file_path = (tmp_path / "docs" / "artifacts" / "completed_plans" / 
                     "completion_summaries" / "session_notes" /
                     "2025-11-30_1200_SESSION_ALL-CAPS.md")
        file_path.write_text("# Test Session Note\n")
        
        is_valid, message = validator.validate_naming_convention(file_path)
        
        assert not is_valid, "Expected invalid for uppercase descriptive part"
        assert "uppercase prefix must have lowercase descriptive part" in message
        assert "SESSION_" in message
    
    def test_session_note_with_number_lowercase_passes(self, validator, tmp_path):
        """Test that SESSION_ files with numbers in lowercase descriptive part pass."""
        # Valid: SESSION_001_test-session.md
        file_path = (tmp_path / "docs" / "artifacts" / "completed_plans" / 
                     "completion_summaries" / "session_notes" /
                     "2025-11-30_1200_SESSION_001_test-session.md")
        file_path.write_text("# Test Session Note\n")
        
        is_valid, message = validator.validate_naming_convention(file_path)
        
        assert is_valid, f"Expected valid, but got: {message}"
    
    def test_assessment_uppercase_fails(self, validator, tmp_path):
        """Test that non-uppercase_prefix types (like assessment) still fail with uppercase."""
        # Invalid: assessment-ALL-CAPS.md
        file_path = tmp_path / "docs" / "artifacts" / "assessments" / "2025-11-30_1200_assessment-ALL-CAPS.md"
        file_path.write_text("# Test Assessment\n")
        
        is_valid, message = validator.validate_naming_convention(file_path)
        
        assert not is_valid, "Expected invalid for uppercase in non-uppercase_prefix type"
        assert "must be lowercase" in message
        assert "kebab-case" in message
    
    def test_prefix_uppercase_allowed(self, validator, tmp_path):
        """Test that BUG_ and SESSION_ prefixes remain uppercase (as expected)."""
        # Test BUG_ prefix is recognized and allowed
        file_path1 = tmp_path / "docs" / "artifacts" / "bug_reports" / "2025-11-30_1200_BUG_test-bug.md"
        file_path1.write_text("# Test\n")
        
        is_valid1, message1 = validator.validate_naming_convention(file_path1)
        assert is_valid1, f"BUG_ prefix should be allowed: {message1}"
        
        # Test SESSION_ prefix is recognized and allowed
        file_path2 = (tmp_path / "docs" / "artifacts" / "completed_plans" /
                      "completion_summaries" / "session_notes" /
                      "2025-11-30_1200_SESSION_test-session.md")
        file_path2.write_text("# Test\n")
        
        is_valid2, message2 = validator.validate_naming_convention(file_path2)
        assert is_valid2, f"SESSION_ prefix should be allowed: {message2}"


class TestUppercaseValidationExistingBehavior:
    """Test that existing validation behavior is preserved for non-uppercase_prefix types."""
    
    @pytest.fixture
    def validator(self, tmp_path):
        """Create a validator instance."""
        artifacts_root = tmp_path / "docs" / "artifacts"
        artifacts_root.mkdir(parents=True)
        (artifacts_root / "assessments").mkdir()
        (artifacts_root / "implementation_plans").mkdir()
        
        return ArtifactValidator(str(artifacts_root))
    
    def test_lowercase_assessment_passes(self, validator, tmp_path):
        """Test that lowercase assessments still pass."""
        file_path = tmp_path / "docs" / "artifacts" / "assessments" / "2025-11-30_1200_assessment-test-name.md"
        file_path.write_text("# Test\n")
        
        is_valid, message = validator.validate_naming_convention(file_path)
        
        assert is_valid, f"Expected valid: {message}"
    
    def test_uppercase_assessment_fails(self, validator, tmp_path):
        """Test that uppercase assessments still fail."""
        file_path = tmp_path / "docs" / "artifacts" / "assessments" / "2025-11-30_1200_assessment-TEST-NAME.md"
        file_path.write_text("# Test\n")
        
        is_valid, message = validator.validate_naming_convention(file_path)
        
        assert not is_valid, "Expected invalid"
        assert "lowercase" in message.lower()
    
    def test_implementation_plan_lowercase_passes(self, validator, tmp_path):
        """Test that lowercase implementation plans still pass."""
        file_path = tmp_path / "docs" / "artifacts" / "implementation_plans" / "2025-11-30_1200_implementation_plan_test-plan.md"
        file_path.write_text("# Test\n")
        
        is_valid, message = validator.validate_naming_convention(file_path)
        
        assert is_valid, f"Expected valid: {message}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
