from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class UpstageConfig:
    base_url: str = "https://api.upstage.ai/v1"
    timeout: int = 60


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        # "Content-Type": "application/json",  <-- Do NOT set this for multipart/form-data
    }


@retry(wait=wait_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(5))
def submit_document_parse_job(image_path: str, api_key: str, cfg: UpstageConfig | None = None) -> dict[str, Any]:
    """
    Submit a document parse request to Upstage API.
    """
    cfg = cfg or UpstageConfig()
    url = f"{cfg.base_url}/document-digitization"

    if not os.path.exists(image_path):
        # For scaffolding, don't fail hard — simulate success
        return {"status": "simulated", "message": f"{image_path} not found; dry-run"}

    with open(image_path, "rb") as f:
        # Key must be 'document' based on curl --form 'document=@...'
        files = {"document": f}
        # Must include 'model' param based on curl --form 'model=ocr'
        data = {"model": "ocr"}
        
        resp = requests.post(
            url, 
            headers=_headers(api_key), 
            files=files, 
            data=data, 
            timeout=cfg.timeout
        )
        resp.raise_for_status()
        return resp.json()
