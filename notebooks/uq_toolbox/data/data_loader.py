# --- Standard Library ---
import io
import base64
from typing import Union, List, Optional, Tuple

# --- Third-Party Libraries ---
import pandas as pd
import numpy as np
from datasets import load_dataset
from PIL import Image
from sklearn.model_selection import train_test_split

def prepare_dataset(
    path: str,
    question_col: Union[str, List[str]],
    answer_col: str,
    name: Optional[str] = None,
    context_col: Optional[str] = None,
    image_col: Optional[str] = None,
    split: str = "train",
    max_samples: Optional[int] = None,
    test_size: float = 0.33,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads ANY dataset from Hugging Face Hub, standardizes its schema via explicit
    configuration injection, handles multi-modal conversions, and outputs isolated splits.
    """
    print(f"⏳ Loading dataset via universal parser: {path} (subset: {name or 'default'})...")

    # 1. Pipeline Ingestion Layer
    raw_dataset = load_dataset(path, name, split=split)
    raw_df = raw_dataset.to_pandas()

    # 2. Deterministic Downsampling
    if max_samples and max_samples < len(raw_df):
        raw_df = raw_df.sample(frac=1, random_state=random_state).reset_index(drop=True).head(max_samples)
    else:
        raw_df = raw_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    standardized_records = []
    has_images = image_col is not None and image_col in raw_df.columns

    # 3. Explicit Mapping Processing Loop
    for idx, row in raw_df.iterrows():
        # Extracting the baseline text inputs safely
        if isinstance(question_col, list):
            # Handles nested dictionary extraction (like PubMedQA context sub-keys)
            current_target = row
            for key in question_col:
                current_target = current_target.get(key, {}) if isinstance(current_target, dict) else ""
            question_base = " ".join(current_target) if isinstance(current_target, (list, np.ndarray)) else str(current_target)
        else:
            question_base = str(row.get(question_col, ""))

        # Appending context metadata blocks if explicitly declared
        if context_col and context_col in row and pd.notna(row[context_col]):
            context_data = row[context_col]
            context_str = " ".join(context_data["contexts"]) if isinstance(context_data, dict) and "contexts" in context_data else str(context_data)
            question_text = f"Context: {context_str}\nQuestion: {question_base}"
        else:
            question_text = question_base

        answer_text = str(row.get(answer_col, "")).strip()

        # Multimodal Encoding Engine Layer
        encoded_image = None
        mime_type = None

        if has_images and row[image_col] is not None:
            try:
                img_data = row[image_col]
                # Standard formats handling (Hugging Face Image feature dict vs PIL Object)
                if isinstance(img_data, dict) and "bytes" in img_data and img_data["bytes"] is not None:
                    img = Image.open(io.BytesIO(img_data["bytes"]))
                elif hasattr(img_data, "convert"):
                    img = img_data
                else:
                    continue

                buffered = io.BytesIO()
                img.convert("RGB").save(buffered, format="PNG")
                encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
                mime_type = "image/png"
            except Exception as e:
                print(f"⚠️ Image encoding failed at row index {idx}: {e}")
                continue

        if question_text or answer_text:
            standardized_records.append({
                "question": question_text,
                "answer": answer_text,
                "image_base64": encoded_image,
                "image_mime": mime_type,
            })

    standardized_df = pd.DataFrame(standardized_records)

    # 4. Data-Leakage Free Split Generation
    train_df, test_df = train_test_split(standardized_df, test_size=test_size, random_state=random_state)

    print(f"✅ Processing complete! Mode: {'MULTIMODAL' if has_images else 'TEXT-ONLY'}")
    print(f"   └── Train (Calibration Pool): {len(train_df)} | Test (Holdout Pool): {len(test_df)}\n")

    return train_df, test_df