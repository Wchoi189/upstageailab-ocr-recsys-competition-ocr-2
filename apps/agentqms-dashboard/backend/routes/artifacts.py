import glob
import os
from datetime import datetime
from typing import Any

import frontmatter
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/artifacts", tags=["artifacts"])

# Configuration
ARTIFACTS_ROOT = "docs/artifacts"
ARTIFACT_TYPES = {
    "implementation_plan": "implementation_plans",
    "assessment": "assessments",
    "audit": "audits",
    "bug_report": "bug_reports"
}

# Models
class ArtifactBase(BaseModel):
    type: str
    title: str
    status: str = "draft"
    category: str | None = None
    tags: list[str] = []

class ArtifactCreate(ArtifactBase):
    content: str

class ArtifactUpdate(BaseModel):
    content: str | None = None
    frontmatter_updates: dict[str, Any] | None = None

class ArtifactResponse(ArtifactBase):
    id: str
    path: str
    created_at: str | None = None
    content: str | None = None # Only for detail view

class ArtifactListResponse(BaseModel):
    items: list[ArtifactResponse]
    total: int

# Helpers
def get_artifact_path(artifact_type: str, artifact_id: str) -> str:
    subdir = ARTIFACT_TYPES.get(artifact_type)
    if not subdir:
        raise ValueError(f"Invalid artifact type: {artifact_type}")
    # Assuming ID matches filename without extension, or we search for it
    # The ID in spec is "2025-12-08_1430_plan_dashboard-integration"
    # The file is "2025-12-08_1430_plan_dashboard-integration.md"
    return os.path.join(ARTIFACTS_ROOT, subdir, f"{artifact_id}.md")

def parse_artifact(file_path: str, include_content: bool = False) -> ArtifactResponse:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    post = frontmatter.load(file_path)
    filename = os.path.basename(file_path)
    artifact_id = os.path.splitext(filename)[0]

    # Extract metadata
    metadata = post.metadata

    return ArtifactResponse(
        id=artifact_id,
        path=file_path,
        type=metadata.get("type", "unknown"),
        title=metadata.get("title", "Untitled"),
        status=metadata.get("status", "draft"),
        category=metadata.get("category"),
        tags=metadata.get("tags", []),
        created_at=metadata.get("date"), # Or parse from filename
        content=post.content if include_content else None
    )

# Endpoints
@router.get("", response_model=ArtifactListResponse)
async def list_artifacts(
    type: str | None = None,
    status: str | None = None,
    limit: int = 50
):
    """List artifacts with filtering."""
    items = []

    # Determine directories to search
    subdirs = [ARTIFACT_TYPES[type]] if type and type in ARTIFACT_TYPES else ARTIFACT_TYPES.values()

    for subdir in subdirs:
        search_path = os.path.join(ARTIFACTS_ROOT, subdir, "*.md")
        files = glob.glob(search_path)

        for f in files:
            try:
                artifact = parse_artifact(f, include_content=False)

                # Filter by status
                if status and artifact.status != status:
                    continue

                items.append(artifact)
            except Exception as e:
                print(f"Error parsing {f}: {e}")
                continue

    # Sort by ID (descending -> newest first)
    items.sort(key=lambda x: x.id, reverse=True)

    return {
        "items": items[:limit],
        "total": len(items)
    }

@router.get("/{id}", response_model=ArtifactResponse)
async def get_artifact(id: str):
    """Get a single artifact by ID."""
    # We need to find the file because ID doesn't strictly tell us the type/folder
    # Optimization: Try to guess type from ID if possible, or search all folders

    found_path = None
    for subdir in ARTIFACT_TYPES.values():
        potential_path = os.path.join(ARTIFACTS_ROOT, subdir, f"{id}.md")
        if os.path.exists(potential_path):
            found_path = potential_path
            break

    if not found_path:
        raise HTTPException(status_code=404, detail="Artifact not found")

    try:
        return parse_artifact(found_path, include_content=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("", response_model=ArtifactResponse)
async def create_artifact(artifact: ArtifactCreate):
    """Create a new artifact."""
    if artifact.type not in ARTIFACT_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid artifact type. Must be one of {list(ARTIFACT_TYPES.keys())}")

    # Generate ID and Filename
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H%M")
    # Simple slug generation
    slug = artifact.title.lower().replace(" ", "-").replace("/", "-")
    artifact_id = f"{timestamp}_{artifact.type}_{slug}"

    filename = f"{artifact_id}.md"
    subdir = ARTIFACT_TYPES[artifact.type]
    file_path = os.path.join(ARTIFACTS_ROOT, subdir, filename)

    # Prepare content with frontmatter
    metadata = artifact.dict(exclude={"content"}, exclude_none=True)
    metadata["date"] = now.strftime("%Y-%m-%d %H:%M (KST)") # Mocking KST for now

    content_body = artifact.content
    # Check if content already has frontmatter and parse it
    if artifact.content.strip().startswith("---"):
        try:
            existing_post = frontmatter.loads(artifact.content)
            # Merge metadata: content metadata overrides form metadata if present
            if existing_post.metadata:
                metadata.update(existing_post.metadata)
            content_body = existing_post.content
        except Exception:
            # If parsing fails, treat as raw content
            pass

    post = frontmatter.Post(content_body, **metadata)

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(frontmatter.dumps(post))

        return parse_artifact(file_path, include_content=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{id}", response_model=ArtifactResponse)
async def update_artifact(id: str, update: ArtifactUpdate):
    """Update an existing artifact."""
    # Find file
    found_path = None
    for subdir in ARTIFACT_TYPES.values():
        potential_path = os.path.join(ARTIFACTS_ROOT, subdir, f"{id}.md")
        if os.path.exists(potential_path):
            found_path = potential_path
            break

    if not found_path:
        raise HTTPException(status_code=404, detail="Artifact not found")

    try:
        post = frontmatter.load(found_path)

        # Update content
        if update.content is not None:
            post.content = update.content

        # Update metadata
        if update.frontmatter_updates:
            post.metadata.update(update.frontmatter_updates)
            post.metadata["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M (KST)")

        with open(found_path, "w", encoding="utf-8") as f:
            f.write(frontmatter.dumps(post))

        return parse_artifact(found_path, include_content=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{id}")
async def delete_artifact(id: str):
    """Delete (archive) an artifact."""
    # Find file
    found_path = None
    for subdir in ARTIFACT_TYPES.values():
        potential_path = os.path.join(ARTIFACTS_ROOT, subdir, f"{id}.md")
        if os.path.exists(potential_path):
            found_path = potential_path
            break

    if not found_path:
        raise HTTPException(status_code=404, detail="Artifact not found")

    try:
        # Archive instead of delete
        archive_dir = os.path.join(ARTIFACTS_ROOT, "archive")
        os.makedirs(archive_dir, exist_ok=True)

        filename = os.path.basename(found_path)
        dest_path = os.path.join(archive_dir, filename)

        os.rename(found_path, dest_path)
        return {"success": True, "message": "Artifact archived", "path": dest_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
