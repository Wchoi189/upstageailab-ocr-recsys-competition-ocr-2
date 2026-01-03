import typer
import logging
from pathlib import Path
from etl.core import LMDBConverter
import lmdb
import cv2
import numpy as np

app = typer.Typer(help="OCR Data ETL Pipeline")

@app.command()
def convert(
    input_dir: Path = typer.Option(..., help="Path to AI Hub validation root (containing labels/images subdirs)"),
    output_dir: Path = typer.Option(..., help="Path to output LMDB directory"),
    num_workers: int = typer.Option(4, help="Number of worker processes"),
    batch_size: int = typer.Option(1000, help="Transaction batch size"),
    limit: int = typer.Option(None, help="Limit number of files to process (for testing)"),
):
    """
    Convert AI Hub dataset to LMDB format with resuming capability.
    """
    converter = LMDBConverter(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        num_workers=num_workers,
        batch_size=batch_size,
        limit=limit
    )
    converter.run()

@app.command()
def inspect(
    lmdb_path: Path = typer.Option(..., help="Path to LMDB directory"),
    num_samples: int = typer.Option(5, help="Number of samples to inspect"),
    output_dir: Path = typer.Option(None, help="Directory to save inspected images (optional)")
):
    """
    Inspect generated LMDB dataset.
    """
    if not lmdb_path.exists():
        typer.echo(f"Path not found: {lmdb_path}")
        raise typer.Exit(code=1)

    env = lmdb.open(str(lmdb_path), readonly=True, lock=False)

    with env.begin() as txn:
        num_samples_encoded = txn.get('num-samples'.encode())
        if num_samples_encoded:
            total = int(num_samples_encoded.decode())
            typer.echo(f"Total samples in DB: {total}")
        else:
            typer.echo("Warning: 'num-samples' key not found.")

        for i in range(1, num_samples + 1):
            key_img = f'image-{i:09d}'.encode()
            key_lbl = f'label-{i:09d}'.encode()

            img_bytes = txn.get(key_img)
            label_bytes = txn.get(key_lbl)

            if not img_bytes or not label_bytes:
                typer.echo(f"Sample {i} missing.")
                continue

            label = label_bytes.decode('utf-8')
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            typer.echo(f"Sample {i}: Label='{label}', ImageShape={img.shape}")

            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                out_p = output_dir / f"sample_{i}.jpg"
                cv2.imwrite(str(out_p), img)
                typer.echo(f"Saved to {out_p}")

if __name__ == "__main__":
    app()
