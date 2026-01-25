#!/usr/bin/env python3
"""
Bloat Detection Tool

Identifies unused, duplicate, or overly complex code for archival consideration.

Usage:
    uv run python AgentQMS/tools/bloat_detector.py --threshold-days 90
    uv run python AgentQMS/tools/bloat_detector.py --include-complexity
"""

import argparse
import json
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


@dataclass
class BloatCandidate:
    """Represents a potential bloat candidate."""
    
    file_path: str
    reasons: list[str]
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    metrics: dict[str, Any]
    recommended_action: str  # ARCHIVE, REFACTOR, MOVE, REVIEW
    detection_date: str


class BloatDetector:
    """Detects code bloat based on configurable criteria."""
    
    def __init__(self, config: dict):
        self.config = config
        self.candidates: list[BloatCandidate] = []
        self.project_root = Path.cwd()
    
    def get_last_import_date(self, file_path: Path) -> datetime | None:
        """Get last date file was imported (via git blame)."""
        try:
            # Search for imports of this module in git history
            module_name = str(file_path).replace("/", ".").replace(".py", "")
            
            # This is a simplified version - production would need more robust search
            result = subprocess.run(
                ["git", "log", "--all", "--format=%cd", "--date=iso", "-S", f"import {module_name}"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0 and result.stdout.strip():
                # Get most recent date
                date_str = result.stdout.strip().split("\n")[0]
                return datetime.fromisoformat(date_str.split()[0])
        except Exception:
            pass
        
        return None
    
    def get_last_commit_date(self, file_path: Path) -> datetime | None:
        """Get last commit date for file."""
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--format=%cd", "--date=iso", str(file_path)],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0 and result.stdout.strip():
                date_str = result.stdout.strip().split()[0]
                return datetime.fromisoformat(date_str)
        except Exception:
            pass
        
        return None
    
    def has_test_coverage(self, file_path: Path) -> bool:
        """Check if file has corresponding test file."""
        # Simple check: look for test file
        test_patterns = [
            file_path.parent.parent / "tests" / f"test_{file_path.name}",
            self.project_root / "tests" / "unit" / f"test_{file_path.name}",
            self.project_root / "tests" / "integration" / f"test_{file_path.name}",
        ]
        
        return any(p.exists() for p in test_patterns)
    
    def is_referenced_in_experiments(self, file_path: Path) -> bool:
        """Check if file is referenced in experiment configs."""
        module_name = str(file_path).replace("/", ".").replace(".py", "")
        
        configs_dir = self.project_root / "configs"
        if not configs_dir.exists():
            return False
        
        for config_file in configs_dir.rglob("*.yaml"):
            try:
                content = config_file.read_text()
                if module_name in content or file_path.stem in content:
                    return True
            except Exception:
                continue
        
        return False
    
    def count_production_imports(self, file_path: Path) -> int:
        """Count how many production files import this module."""
        count = 0
        module_name = str(file_path).replace("/", ".").replace(".py", "")
        
        for py_file in self.project_root.rglob("*.py"):
            if py_file == file_path or "test" in str(py_file):
                continue
            
            try:
                content = py_file.read_text()
                if f"from {module_name}" in content or f"import {module_name}" in content:
                    count += 1
            except Exception:
                continue
        
        return count
    
    def scan_unused_code(self) -> list[BloatCandidate]:
        """Scan for unused code based on time thresholds."""
        threshold_days = self.config.get("no_imports_days", 90)
        threshold = timedelta(days=threshold_days)
        now = datetime.now()
        
        candidates = []
        
        # Scan ocr/ directory
        for py_file in (self.project_root / "ocr").rglob("*.py"):
            if "__pycache__" in str(py_file) or "__init__" in py_file.name:
                continue
            
            reasons = []
            metrics = {}
            
            # Check last import
            last_import = self.get_last_import_date(py_file)
            if last_import:
                days_since_import = (now - last_import).days
                metrics["days_since_last_import"] = days_since_import
                if days_since_import > threshold_days:
                    reasons.append(f"No imports in {days_since_import} days")
            else:
                reasons.append("No import history found")
                metrics["days_since_last_import"] = "N/A"
            
            # Check test coverage
            has_tests = self.has_test_coverage(py_file)
            metrics["has_test_coverage"] = has_tests
            if not has_tests:
                reasons.append("No test coverage")
            
            # Check experiment references
            in_experiments = self.is_referenced_in_experiments(py_file)
            metrics["referenced_in_experiments"] = in_experiments
            if not in_experiments:
                reasons.append("Not referenced in experiments")
            
            # Check production imports
            import_count = self.count_production_imports(py_file)
            metrics["production_import_count"] = import_count
            if import_count == 0:
                reasons.append("No production imports")
            
            # Determine severity and action
            if len(reasons) >= 3:
                severity = "HIGH" if not has_tests else "MEDIUM"
                action = "ARCHIVE" if import_count == 0 else "REVIEW"
                
                candidates.append(BloatCandidate(
                    file_path=str(py_file.relative_to(self.project_root)),
                    reasons=reasons,
                    severity=severity,
                    metrics=metrics,
                    recommended_action=action,
                    detection_date=now.isoformat()
                ))
        
        return candidates
    
    def generate_report(self, output_path: Path):
        """Generate JSON report of findings."""
        report = {
            "scan_date": datetime.now().isoformat(),
            "config": self.config,
            "summary": {
                "total_candidates": len(self.candidates),
                "by_severity": self._count_by_severity(),
                "by_action": self._count_by_action()
            },
            "candidates": [asdict(c) for c in self.candidates]
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))
        
        print(f"✅ Bloat report generated: {output_path}")
        print(f"   Total candidates: {len(self.candidates)}")
        print(f"   By severity: {report['summary']['by_severity']}")
        print(f"   By action: {report['summary']['by_action']}")
    
    def _count_by_severity(self) -> dict[str, int]:
        """Count candidates by severity."""
        counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        for candidate in self.candidates:
            counts[candidate.severity] = counts.get(candidate.severity, 0) + 1
        return counts
    
    def _count_by_action(self) -> dict[str, int]:
        """Count candidates by recommended action."""
        counts = {}
        for candidate in self.candidates:
            counts[candidate.recommended_action] = counts.get(candidate.recommended_action, 0) + 1
        return counts


def main():
    parser = argparse.ArgumentParser(description="Detect code bloat")
    parser.add_argument("--threshold-days", type=int, default=90,
                       help="Days without imports to consider unused")
    parser.add_argument("--output", type=Path,
                       default=Path("analysis/bloat-report.json"),
                       help="Output file path")
    parser.add_argument("--include-complexity", action="store_true",
                       help="Include complexity analysis (slow)")
    parser.add_argument("--include-duplication", action="store_true",
                       help="Include duplication detection (slow)")
    
    args = parser.parse_args()
    
    config = {
        "no_imports_days": args.threshold_days,
        "include_complexity": args.include_complexity,
        "include_duplication": args.include_duplication
    }
    
    print("🔍 Starting bloat detection...")
    print(f"   Threshold: {args.threshold_days} days")
    
    detector = BloatDetector(config)
    
    # Run scans
    print("\n📊 Scanning for unused code...")
    detector.candidates.extend(detector.scan_unused_code())
    
    if args.include_complexity:
        print("📊 Analyzing complexity...")
        # TODO: Implement complexity scan
        print("   (Complexity scan not yet implemented)")
    
    if args.include_duplication:
        print("📊 Detecting duplication...")
        # TODO: Implement duplication scan
        print("   (Duplication scan not yet implemented)")
    
    # Generate report
    print("\n📝 Generating report...")
    detector.generate_report(args.output)


if __name__ == "__main__":
    main()
