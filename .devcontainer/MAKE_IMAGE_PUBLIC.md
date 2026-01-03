# Making the GHCR Image Public

To make your GitHub Container Registry image public so Codespaces can pull it without authentication:

## Steps

1. **Go to the package settings page**:
   ```
   https://github.com/users/Wchoi189/packages/container/upstageailab-ocr-recsys-competition-ocr-2/settings
   ```

2. **Scroll to "Danger Zone"** at the bottom of the page

3. **Find "Change package visibility"**

4. **Click "Change visibility"**

5. **Select "Public"**

6. **Type the package name to confirm**: `upstageailab-ocr-recsys-competition-ocr-2`

7. **Click "I understand the consequences, change package visibility"**

## Verification

After making the image public, verify it's accessible:

```bash
# This should work without authentication
docker pull ghcr.io/wchoi189/upstageailab-ocr-recsys-competition-ocr-2:latest
```

Expected: Image pulls successfully without requiring `docker login`

## Alternative: Keep Private with Authentication

If you prefer to keep the image private, you can add GitHub token authentication to the devcontainer:

```json
{
  "containerEnv": {
    "GITHUB_TOKEN": "${localEnv:GITHUB_TOKEN}"
  }
}
```

Then users would need to set a `GITHUB_TOKEN` environment variable with read:packages permission.

## Recommendation

✅ **Make it public** if this is an open-source project (recommended)
⚠️ **Keep it private** if the image contains proprietary code or secrets

For this OCR competition project, making it public is likely the best choice.
