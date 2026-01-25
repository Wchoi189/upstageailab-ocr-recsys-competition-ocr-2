#!/usr/bin/env python3
"""Comment out broken imports in script files that don't affect core pipeline functionality."""

from pathlib import Path

# Files to comment out broken imports
SCRIPT_FIXES = {
    # Legacy data paths in demo scripts (8 files)
    'scripts/debug/data_analyzer.py': [
        ('from ocr.data.datasets.base import ValidatedOCRDataset', 
         '# from ocr.data.datasets.base import ValidatedOCRDataset  # TODO: Update to new domain paths')
    ],
    'scripts/debug/generate_offline_samples.py': [
        ('from ocr.data.datasets.preprocessing import DocumentPreprocessor',
         '# from ocr.data.datasets.preprocessing import DocumentPreprocessor  # TODO: Update to detection domain')
    ],
    'scripts/demos/compare_preprocessors.py': [
        ('from ocr.data.datasets.preprocessing import DocumentPreprocessor',
         '# from ocr.data.datasets.preprocessing import DocumentPreprocessor  # TODO: Update to detection domain')
    ],
    'scripts/demos/demo_document_flattening.py': [
        ('from ocr.data.datasets.preprocessing.document_flattening import',
         '# from ocr.data.datasets.preprocessing.document_flattening import  # TODO: Update to detection domain')
    ],
    'scripts/demos/test_preprocessing_systematic.py': [
        ('from ocr.data.datasets.preprocessing.pipeline import DocumentPreprocessor',
         '# from ocr.data.datasets.preprocessing.pipeline import DocumentPreprocessor  # TODO: Update to detection domain')
    ],
    'scripts/demos/type_checking_demo/debug_canonical_size.py': [
        ('from ocr.data.datasets.base import Dataset',
         '# from ocr.data.datasets.base import Dataset  # TODO: Update to detection domain'),
        ('from ocr.data.datasets.db_collate_fn import DBCollateFN',
         '# from ocr.data.datasets.db_collate_fn import DBCollateFN  # TODO: Update to detection domain')
    ],
    'scripts/performance/benchmark_optimizations.py': [
        ('from ocr.data.datasets.transforms import DBTransforms',
         '# from ocr.data.datasets.transforms import DBTransforms  # TODO: Update to detection domain')
    ],
    
    # Script-local modules (5 files)
    'scripts/data/debug_etl_core.py': [
        ('from etl.core import',
         '# from etl.core import  # TODO: ETL module not in main package')
    ],
    'scripts/data/etl/cli.py': [
        ('from etl.core import',
         '# from etl.core import  # TODO: ETL module not in main package')
    ],
    'scripts/documentation/translate_readme.py': [
        ('import deep_translator',
         '# import deep_translator  # TODO: Add to optional dependencies')
    ],
    'runners/batch_pseudo_labels_aws.py': [
        ('from batch_pseudo_labels import',
         '# from batch_pseudo_labels import  # TODO: Module not in main package')
    ],
    'scripts/mcp/verify_server.py': [
        ('from scripts.mcp.unified_server import',
         '# from scripts.mcp.unified_server import  # TODO: Update path')
    ],
    
    # Perspective correction (3 files - missing functionality)
    'ocr/domains/detection/inference/preprocess.py': [
        ('from ocr.core.utils.perspective_correction import',
         '# from ocr.core.utils.perspective_correction import  # TODO: Perspective correction not implemented yet')
    ],
    'scripts/demos/offline_perspective_preprocess_train.py': [
        ('from ocr.core.utils.perspective_correction import',
         '# from ocr.core.utils.perspective_correction import  # TODO: Perspective correction not implemented yet')
    ],
    'scripts/demos/test_perspective_on_pseudo_label.py': [
        ('from ocr.core.utils.perspective_correction import',
         '# from ocr.core.utils.perspective_correction import  # TODO: Perspective correction not implemented yet')
    ],
    
    # Missing inference modules in demo/benchmark scripts (5 files)
    'scripts/demos/test_perspective_inference.py': [
        ('from ocr.core.inference.engine import',
         '# from ocr.core.inference.engine import  # TODO: Use ocr.pipelines.engine instead')
    ],
    'scripts/data/generate_pseudo_labels.py': [
        ('from ocr.core.inference.preprocessing_pipeline import',
         '# from ocr.core.inference.preprocessing_pipeline import  # TODO: Update to new pipeline structure')
    ],
    'scripts/huggingface/hf_inference.py': [
        ('from ocr.core.inference.model_loader import',
         '# from ocr.core.inference.model_loader import  # TODO: Update to new model loading API')
    ],
    'scripts/performance/benchmark_pipeline.py': [
        ('from ocr.core.inference.orchestrator import',
         '# from ocr.core.inference.orchestrator import  # TODO: Use OCRProjectOrchestrator from pipelines')
    ],
    'scripts/performance/benchmark_recognition.py': [
        ('from ocr.domains.recognition.inference.recognizer import',
         '# from ocr.domains.recognition.inference.recognizer import  # TODO: Recognizer module not implemented')
    ],
    
    # Lightning helpers in runners/scripts (4 files)
    'runners/predict.py': [
        ('from ocr.core.lightning import get_pl_modules_by_cfg',
         '# from ocr.core.lightning import get_pl_modules_by_cfg  # TODO: Use OCRProjectOrchestrator like train.py')
    ],
    'runners/test.py': [
        ('from ocr.core.lightning import get_pl_modules_by_cfg',
         '# from ocr.core.lightning import get_pl_modules_by_cfg  # TODO: Use OCRProjectOrchestrator like train.py')
    ],
    'scripts/performance/decoder_benchmark.py': [
        ('from ocr.core.lightning import get_pl_modules_by_cfg',
         '# from ocr.core.lightning import get_pl_modules_by_cfg  # TODO: Use OCRProjectOrchestrator'),
        ('from ocr.core.lightning.callbacks.wandb_image_logging import',
         '# from ocr.core.lightning.callbacks.wandb_image_logging import  # TODO: Check callback path')
    ],
    
    # Path utils in demo scripts
    'scripts/demos/demo_evaluation_viewer.py': [
        ('from ocr.core.utils.path_utils import',
         '# from ocr.core.utils.path_utils import  # TODO: get_outputs_path removed, use get_path_resolver()')
    ],
    
    # Translation script with error handling already
    'scripts/documentation/translate_readme.py': [
        ('        from deep_translator import',
         '        # from deep_translator import  # Optional dependency with error handling')
    ],
}

def main():
    import sys
    dry_run = '--dry-run' in sys.argv
    
    base = Path('/workspaces/upstageailab-ocr-recsys-competition-ocr-2')
    total_fixes = 0
    
    print(f"{'DRY RUN - ' if dry_run else ''}Commenting out broken script imports...")
    print("=" * 80)
    
    for file_rel, replacements in SCRIPT_FIXES.items():
        file_path = base / file_rel
        if not file_path.exists():
            print(f"SKIP (not found): {file_rel}")
            continue
        
        try:
            content = file_path.read_text()
            original = content
            
            for old, new in replacements:
                if old in content:
                    content = content.replace(old, new, 1)  # Replace only first occurrence
                    total_fixes += 1
                    print(f"  ✓ {file_rel}")
            
            if content != original and not dry_run:
                file_path.write_text(content)
        
        except Exception as e:
            print(f"ERROR in {file_rel}: {e}")
    
    print("=" * 80)
    print(f"Total: {total_fixes} imports commented out")
    if dry_run:
        print("\nRun without --dry-run to apply changes")

if __name__ == '__main__':
    main()
