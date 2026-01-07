import os
import yaml # type: ignore
from dataclasses import dataclass
from typing import List, Optional, Any
from agent_debug_toolkit.precomputes.symbol_table import SymbolTable, SymbolDefinition

@dataclass
class ResolutionResult:
    target: str
    status: str # "RESOLVED", "UNRESOLVED_DYNAMIC", "UNRESOLVED_RELATIVE", "NOT_FOUND"
    symbol: Optional[SymbolDefinition] = None
    error: Optional[str] = None

@dataclass
class ConfigMapping:
    config_file: str
    target_key: str # _target_ value
    resolution: ResolutionResult

class HydraConfigResolver:
    def __init__(self, symbol_table: SymbolTable):
        self.symbol_table = symbol_table

    def analyze_config_dir(self, config_dir: str) -> List[ConfigMapping]:
        mappings = []
        for root, _, files in os.walk(config_dir):
            for file in files:
                if file.endswith((".yaml", ".yml")):
                    full_path = os.path.join(root, file)
                    mappings.extend(self._analyze_file(full_path))
        return mappings

    def resolve_target(self, target: str) -> ResolutionResult:
        if "${" in target:
             return ResolutionResult(target, "UNRESOLVED_DYNAMIC")

        if target.startswith("."):
             return ResolutionResult(target, "UNRESOLVED_RELATIVE")

        symbol = self.symbol_table.lookup(target)
        if symbol:
            return ResolutionResult(target, "RESOLVED", symbol=symbol)

        return ResolutionResult(target, "NOT_FOUND")

    def _analyze_file(self, file_path: str) -> List[ConfigMapping]:
        mappings = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f)

            targets = self._find_targets(content)
            for target in targets:
                resolution = self.resolve_target(target)
                mappings.append(ConfigMapping(
                    config_file=file_path,
                    target_key=target,
                    resolution=resolution
                ))
        except Exception:
            # Handle empty files or parsing errors silently
            pass
        return mappings

    def _find_targets(self, data: Any) -> List[str]:
        targets = []
        if isinstance(data, dict):
            for k, v in data.items():
                if k == "_target_":
                    if isinstance(v, str):
                        targets.append(v)
                else:
                    targets.extend(self._find_targets(v))
        elif isinstance(data, list):
            for item in data:
                targets.extend(self._find_targets(item))
        return targets
