import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

# Add scripts directory to path to import the module
sys.path.append(os.path.abspath("scripts/data"))
# We might need to adjust import if the file is not a module
# For now, let's assume we can import it or subprocess it.
# Importing relative path 'scripts.data.process_aihub_validation' might be tricky if no __init__.py
# So I'll just write a test that acts like the script or uses logic copied from it?
# Better: Make the script importable. I'll assume scripts/data/__init__.py exists or create it.

class TestProcessAIHub(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.root_path = Path(self.test_dir)
        self.labels_dir = self.root_path / "labels"
        self.images_dir = self.root_path / "images"
        self.labels_dir.mkdir(parents=True)
        self.images_dir.mkdir(parents=True)

        # Create a sample JSON
        self.sample_json = {
            "images": [
                {
                    "id": 1,
                    "width": 1000,
                    "height": 1000,
                    "file_name": "test_image.jpg"
                }
            ],
            "annotations": [
                {
                    "id": 101,
                    "image_id": 1,
                    "annotation.bbox": [10, 10, 100, 50],
                    "annotation.text": "Hello"
                },
                 {
                    "id": 102,
                    "image_id": 1,
                    "annotation.bbox": [200, 200, 50, 20],
                    "annotation.text": "World"
                }
            ]
        }

        # Save JSON
        with open(self.labels_dir / "test.json", "w") as f:
            json.dump(self.sample_json, f)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_logic(self):
        # We'll just define the logic here to verify it matches my assumptions,
        # or better: verify the *actual* script.
        # But importing the script might run main().
        # I wrapped main in if __name__ == "__main__", so safe to import.
        # BUT 'process_aihub_validation' function takes args properly.

        # Dynamic import
        import importlib.util
        spec = importlib.util.spec_from_file_location("process_aihub_validation", "scripts/data/process_aihub_validation.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        output_parquet = self.root_path / "output.parquet"

        module.process_aihub_validation(
            root_dir=str(self.root_path),
            output_path=str(output_parquet)
        )

        self.assertTrue(output_parquet.exists())

        df = pd.read_parquet(output_parquet)
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row['width'], 1000)
        self.assertEqual(row['height'], 1000)
        self.assertEqual(len(row['texts']), 2)
        self.assertEqual(row['texts'][0], "Hello")
        # Check coordinates: [x, y, w, h] -> [x, y, x+w, y+h]
        # [10, 10, 100, 50] -> [10, 10, 110, 60]
        # Convert likely numpy array to list for comparison
        poly = row['polygons'][0]
        if hasattr(poly, 'tolist'):
            poly = poly.tolist()
        self.assertEqual(poly, [10, 10, 110, 60])

if __name__ == '__main__':
    unittest.main()
