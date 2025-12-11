# Output Data Format

## Polygon CSV (Submission Format)

```csv
filename,polygons
drp.en_ko.in_house.selectstar_003883.jpg,10 50 100 50 100 150 10 150|110 150 200 150 200 250 110 250
```

**Format**:
- Header: `filename,polygons`
- Polygon format: Space-separated "x y" pairs
- Multiple polygons: Pipe-separated (`|`)
- Coordinates: Integers, pixel-space, top-left origin (0,0)
