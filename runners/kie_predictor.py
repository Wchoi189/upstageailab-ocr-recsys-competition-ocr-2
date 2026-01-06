import argparse
import logging
from pathlib import Path

# Setup logging with RichHandler
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    force=True,
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)

# Reduce verbosity
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def draw_boxes(image, boxes, labels, label_map):
    import cv2

    for box, label_id in zip(boxes, labels, strict=False):
        if label_id == -100:
            continue
        label_text = label_map.get(label_id, "O")
        if label_text == "O":
            continue

        x1, y1, x2, y2 = box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, label_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to config (optional, to get specific params if needed)")
    parser.add_argument("--input_data", type=str, required=True, help="Path to input parquet or jsonl")
    parser.add_argument("--image_dir", type=str, default=None, help="Path to images")
    parser.add_argument("--output_dir", type=str, default="predictions", help="Output directory")
    parser.add_argument("--model_type", type=str, default="layoutlmv3", choices=["layoutlmv3", "lilt"])
    parser.add_argument("--model_name", type=str, default="microsoft/layoutlmv3-base")
    # Lazy torch import means we can't default to torch.cuda.is_available() in arg definition easily without importing torch
    # So we use 'auto' or check later
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    # LAZY IMPORTS
    import cv2
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    from tqdm import tqdm
    from transformers import AutoTokenizer, LayoutLMv3Processor

    from ocr.models.kie_models import LayoutLMv3Wrapper, LiLTWrapper

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Processor
    logger.info(f"Loading processor/tokenizer for {args.model_type}...")
    if args.model_type == "layoutlmv3":
        processor = LayoutLMv3Processor.from_pretrained(args.model_name, apply_ocr=False)
    else:
        # LiLT
        _tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 2. Load Model
    logger.info(f"Loading model from {args.checkpoint}...")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return

    # Try to infer num_labels from checkpoint weights to match architecture
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Heuristic to find classifier weight
    classifier_weight = None
    keys_to_check = ["model.classifier.weight", "model.model.classifier.weight", "classifier.weight"]
    for k in keys_to_check:
        if k in state_dict:
            classifier_weight = state_dict[k]
            break

    # If wrapped in PL, keys might have 'model.' prefix
    if classifier_weight is None:
        # Try matching any key ending in classifier.weight
        for k in state_dict.keys():
            if k.endswith("classifier.weight"):
                classifier_weight = state_dict[k]
                break

    if classifier_weight is not None:
        num_labels = classifier_weight.shape[0]
        logger.info(f"Inferred num_labels: {num_labels}")
    else:
        num_labels = 7
        logger.warning(f"Could not infer num_labels from checkpoint, using default: {num_labels}")

    config = {"pretrained_model_name_or_path": args.model_name, "num_labels": num_labels}

    if args.model_type == "layoutlmv3":
        model = LayoutLMv3Wrapper(config)
    else:
        model = LiLTWrapper(config)

    # Load state dict
    # PL saves with "model." prefix if we wrapped it in LightningModule
    # Our LayoutLMv3Wrapper IS the model passed to KIEPLModule.
    # KIEPLModule.model = wrappers.
    # So state_dict keys will be "model.model.layoutlmv3..."
    # Wait, KIEPLModule has self.model = model
    # So state_dict has "model.xxx".
    # We want to load into model (LayoutLMv3Wrapper).

    # Clean keys
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[6:]] = v  # Remove "model." prefix
        else:
            new_state_dict[k] = v

    try:
        model.load_state_dict(new_state_dict)
    except RuntimeError as e:
        logger.warning(f"Strict load failed: {e}. Trying non-strict load...")
        model.load_state_dict(new_state_dict, strict=False)

    model.to(device)
    model.eval()

    # 3. Load Data
    logger.info(f"Loading data from {args.input_data}...")
    try:
        df = pd.read_parquet(args.input_data)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # 4. Inference Loop
    # FIXME: Load from config/artifact
    label_list = ["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"]
    id2label = dict(enumerate(label_list))

    logger.info("Starting inference...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_path = row["image_path"]
        if args.image_dir and not Path(image_path).is_absolute():
            image_path = Path(args.image_dir) / image_path

        try:
            image = Image.open(image_path).convert("RGB")
        except:
            logger.warning(f"Skipping {image_path}")
            continue

        words = row["texts"]
        boxes = row["polygons"]

        # Format boxes
        formatted_boxes = []
        for poly in boxes:
            pts = np.array(poly).reshape(-1, 2)
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)
            formatted_boxes.append([int(x_min), int(y_min), int(x_max), int(y_max)])

        # Preprocess
        # Be careful with max_length
        encoding = processor(
            images=image, text=words, boxes=formatted_boxes, return_tensors="pt", padding="max_length", truncation=True, max_length=512
        )

        # Move to device
        for k, v in encoding.items():
            encoding[k] = v.to(device)

        with torch.no_grad():
            outputs = model(**encoding, return_loss=False)

        logits = outputs["logits"]
        pred_ids = torch.argmax(logits, dim=2)[0]  # (seq_len,)

        # Visualize
        # Map tokens to words
        word_ids = encoding.word_ids()

        # Create a map for word_index -> majority label
        word_labels = {}
        for i, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if word_idx not in word_labels:
                word_labels[word_idx] = []
            word_labels[word_idx].append(pred_ids[i].item())

        # Resolve majority vote
        final_labels = []
        for i in range(len(words)):
            if i in word_labels:
                # Majority vote
                ids = word_labels[i]
                majority = max(set(ids), key=ids.count)
                final_labels.append(majority)
            else:
                final_labels.append(0)  # O

        # Draw on image
        # Use opencv
        cv_img = cv2.imread(str(image_path))
        if cv_img is None:
            cv_img = np.array(image)
            cv_img = cv_img[:, :, ::-1].copy()

        res_img = draw_boxes(cv_img, formatted_boxes, final_labels, id2label)

        save_path = output_dir / Path(image_path).name
        cv2.imwrite(str(save_path), res_img)

    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
