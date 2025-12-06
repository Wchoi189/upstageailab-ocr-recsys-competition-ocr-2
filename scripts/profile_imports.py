#!/usr/bin/env python3
"""Profile import times for training script to identify bottlenecks."""

import importlib
import sys
import time
from pathlib import Path


def profile_imports(module_name: str, max_depth: int = 3):
    """Profile import times for a module and its dependencies."""

    class ImportTimer:
        def __init__(self):
            self.import_times = {}
            self.import_stack = []
            self.original_import = __builtins__.__import__

        def __enter__(self):
            __builtins__.__import__ = self._timed_import
            return self

        def __exit__(self, *args):
            __builtins__.__import__ = self.original_import

        def _timed_import(self, name, *args, **kwargs):
            level = len(self.import_stack)
            self.import_stack.append(name)

            start = time.perf_counter()
            try:
                result = self.original_import(name, *args, **kwargs)
                elapsed = time.perf_counter() - start

                if level < max_depth and elapsed > 0.01:  # Only track imports > 10ms
                    self.import_times[name] = {
                        'time': elapsed,
                        'level': level,
                        'parent': self.import_stack[-2] if len(self.import_stack) > 1 else None
                    }

                return result
            finally:
                self.import_stack.pop()

    print(f"Profiling imports for: {module_name}\n")

    with ImportTimer() as timer:
        start_total = time.perf_counter()
        importlib.import_module(module_name)
        total_time = time.perf_counter() - start_total

    # Sort by time (descending)
    sorted_imports = sorted(
        timer.import_times.items(),
        key=lambda x: x[1]['time'],
        reverse=True
    )

    print(f"Total import time: {total_time:.3f}s\n")
    print("Top 30 slowest imports:")
    print("-" * 90)

    for name, info in sorted_imports[:30]:
        indent = "  " * info['level']
        parent = f" (from {info['parent']})" if info['parent'] else ""
        print(f"{indent}{name:60s} {info['time']:6.3f}s{parent}")

    # Categorize heavy imports
    heavy_libs = ['torch', 'lightning', 'transformers', 'wandb', 'albumentations', 'cv2', 'PIL']
    heavy_imports = {}
    for name, info in timer.import_times.items():
        for lib in heavy_libs:
            if lib in name:
                if lib not in heavy_imports:
                    heavy_imports[lib] = []
                heavy_imports[lib].append((name, info['time']))
                break

    if heavy_imports:
        print("\n\nHeavy library imports by category:")
        print("-" * 90)
        for lib, imports in sorted(heavy_imports.items(), key=lambda x: sum(i[1] for i in x[1]), reverse=True):
            total = sum(i[1] for i in imports)
            print(f"\n{lib}: {total:.3f}s total ({len(imports)} imports)")
            for name, t in sorted(imports, key=lambda x: x[1], reverse=True)[:5]:
                print(f"  - {name}: {t:.3f}s")

    return timer.import_times, total_time

if __name__ == "__main__":
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    print("=" * 90)
    print("TRAINING SCRIPT IMPORT PROFILING")
    print("=" * 90)
    print()

    # Profile the main training runner
    try:
        import_times, total = profile_imports("runners.train", max_depth=2)

        print(f"\n\n{'=' * 90}")
        print(f"SUMMARY: Total startup time = {total:.3f}s")
        print(f"{'=' * 90}")

        # Categorize slow imports
        slow_threshold = 0.5  # seconds
        slow_imports = {k: v for k, v in import_times.items() if v['time'] > slow_threshold}

        if slow_imports:
            print(f"\n{len(slow_imports)} imports taking >{slow_threshold}s:")
            for name, info in sorted(slow_imports.items(), key=lambda x: x[1]['time'], reverse=True):
                print(f"  - {name}: {info['time']:.3f}s")
    except Exception as e:
        print(f"\nError profiling imports: {e}")
        import traceback
        traceback.print_exc()
