# Demo GIFs and Animations

This directory contains animated demonstrations showcasing key features of the OCR system.

## 🎬 Available Demos

<!-- TODO: Create animated GIFs showcasing features -->

### Planned Demos

1. **Model Testing Demo** (`model-testing-demo.gif`)
   - Interactive model testing interface
   - Real-time inference visualization
   - Result comparison

2. **Preprocessing Pipeline Demo** (`preprocessing-demo.gif`)
   - Before/after image enhancement
   - Step-by-step preprocessing visualization
   - Performance improvements

3. **Training Workflow Demo** (`training-demo.gif`)
   - Command builder usage
   - Training progress visualization
   - Experiment tracking

4. **Evaluation Viewer Demo** (`evaluation-demo.gif`)
   - Result gallery navigation
   - Detailed analysis views
   - Export functionality

## 🛠️ Creating Demo GIFs

### Tools

- **Screen Recording**: OBS Studio, SimpleScreenRecorder, or built-in screen recorders
- **GIF Creation**:
  - [Peek](https://github.com/phw/peek) (Linux)
  - [LICEcap](https://www.cockos.com/licecap/) (Windows/Mac)
  - [Gifox](https://gifox.io/) (Mac)
  - FFmpeg for conversion

### Guidelines

- **Duration**: Keep GIFs under 10 seconds when possible
- **Size**: Optimize file size (aim for <5MB)
- **Resolution**: 800x600 or 1280x720 recommended
- **Frame Rate**: 10-15 FPS is usually sufficient
- **Loop**: Set to loop continuously
- **Focus**: Highlight key features clearly

### Recording Tips

1. Use a clean, uncluttered desktop
2. Slow down actions for clarity
3. Add text annotations if helpful
4. Use consistent color schemes
5. Test on different displays

### Optimization

```bash
# Using FFmpeg to optimize GIF
ffmpeg -i input.mp4 -vf "fps=10,scale=800:-1:flags=lanczos" -c:v gif output.gif

# Using gifsicle for further optimization
gifsicle -O3 --lossy=80 -o optimized.gif input.gif
```

## 📝 Usage in README

Reference demos in the README:

```markdown
![Model Testing Demo](docs/assets/images/demos/model-testing-demo.gif)
```

---

*Demo GIFs will be created as features are finalized. Contributions welcome!*

