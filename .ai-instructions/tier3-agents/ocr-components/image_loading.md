# Component: Image Loading (ImageLoader)

## Role
Provides robust image loading capabilities from various sources (NextJS uploads, file paths, raw bytes), ensuring consistent orientation and format.

## Critical Logic

### 1. EXIF Normalization
- **Issue**: Smartphone photos often have EXIF rotation tags (e.g., "Rotate 90 CW"). OpenCV's `cv2.imread` ignores these, leading to sideways text detection.
- **Solution**: Uses `PIL.ImageOps.exif_transpose` to physically rotate the pixel data to match the intended orientation before conversion to numpy array.

### 2. Format Handling
- **JPEG**: Uses `turbojpeg` (if available) for 2-3x faster decoding than PIL/OpenCV.
- **Other**: Falls back to `PIL.Image`.
- **Output**: **BGR** Numpy Array (H, W, 3) compatible with OpenCV ecosystem.

### 3. Constraints
- **Max Image Size**: implicitly limited by server RAM, but no hard cap enforced here.
- **Channels**: Always converts 4-channel (RGBA) to 3-channel (BGR) by dropping alpha.

## Data Contract
**Input**: `bytes`, `str` (path), or `uploaded_file`
**Output**: `np.ndarray` (BGR)
