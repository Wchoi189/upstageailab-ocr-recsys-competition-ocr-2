# ScanTailor Installation Guide

ScanTailor is not available in default Ubuntu/Debian repositories. Here are installation options:

## Option 1: Build from Source (Recommended)

### Prerequisites
```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    qt5-default \
    libqt5svg5-dev \
    libboost-dev \
    libtiff5-dev \
    libjpeg-dev \
    libpng-dev \
    libxrender-dev
```

### Build and Install
```bash
# Clone repository
git clone https://github.com/scantailor/scantailor.git
cd scantailor

# Build
mkdir build && cd build
cmake ..
make -j$(nproc)

# Install (optional, or use from build directory)
sudo make install
```

### Verify Installation
```bash
scantailor-cli --help
# or
scantailor --help
```

## Option 2: Use Docker (Easier Alternative)

If building from source is too complex, use Docker:

```bash
# Pull ScanTailor Docker image (if available)
docker pull scantailor/scantailor

# Or create a Dockerfile
cat > Dockerfile.scantailor << 'EOF'
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y \
    scantailor \
    && rm -rf /var/lib/apt/lists/*
EOF
```

## Option 3: Use Pre-built Binary

Check for pre-built binaries at:
- https://github.com/scantailor/scantailor/releases
- May include AppImage or static binaries

## Option 4: Alternative - Use ScanTailor Advanced

ScanTailor Advanced is a fork with more features:
```bash
git clone https://github.com/4lex4/scantailor-advanced.git
cd scantailor-advanced
# Follow similar build instructions
```

## Testing After Installation

Once installed, test with:
```bash
python scripts/test_scantailor_integration.py \
    --input-dir outputs/perspective_test \
    --output-dir outputs/scantailor_test \
    --num-samples 3
```

## Note

ScanTailor is primarily designed for scanned documents and may work better on certain image types. The Python wrapper is experimental and may require adjustments based on your ScanTailor version.

