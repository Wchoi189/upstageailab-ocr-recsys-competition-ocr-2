# Input Data Format

## Annotation JSON (`val.json`)

```json
{
  "images": {
    "drp.en_ko.in_house.selectstar_000007.jpg": {
      "words": {
        "0001": {
          "points": [[273.31, 164.16], [585.2, 162.57], [585.2, 200.61], [271.72, 202.2]],
          "transcription": "텍스트 내용",
          "illegibility": false,
          "language": "ko"
        }
      },
      "img_w": 1280,
      "img_h": 960
    }
  }
}
```

**Fields**:
- `words`: Object mapping word IDs to word data
- `points`: Array of [x, y] coordinate pairs (polygon vertices)
- `transcription`: Text content
- `illegibility`: Boolean flag
- `language`: Language code (e.g., "ko", "en")
- `img_w`, `img_h`: Image dimensions (pixels)
