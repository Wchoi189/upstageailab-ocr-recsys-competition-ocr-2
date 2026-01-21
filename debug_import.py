try:
    from ocr.core.lightning.base import OCRPLModule
    import inspect
    print(f"IMPORTED FROM: {inspect.getfile(OCRPLModule)}")
except ImportError as e:
    print(f"IMPORT FAILED: {e}")
