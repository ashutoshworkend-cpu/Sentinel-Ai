# Labeled detector evaluation

The included evaluator measures the primary YOLO detector and `unknown` policy against labeled bounding boxes in image sets or video frames. Video evaluation reuses the live track-level temporal classifier. It does not infer accuracy from unlabeled footage or measure security outcomes, and model scores are not calibrated probabilities.

## Dataset layout

```text
dataset/
  images/
    000001.jpg
    000002.jpg
  labels/
    000001.txt
    000002.txt
  classes.json
```

Every image must have a paired label file; an empty `.txt` represents a negative image. Each non-empty label file uses one normalized YOLO row per object:

```text
class_id x_center y_center width height
```

Use the same image dimensions for label normalization. The optional JSON class map maps integer category IDs to dataset names and may include categories outside the detector's training set. Start from [classes.example.json](classes.example.json).

Build a representative, independently labeled set with day/night and low-light clips, small and partly occluded objects, people, vehicles, pets, cleaning equipment, tools, garden equipment, bags, furniture, shadows, and reflections. Include negative frames with no relevant objects. Keep train, validation, and test scenes separate; do not tune thresholds against the final test set.

For videos, make one label file per evaluated source frame using its zero-based frame number: `frame_000000.txt`, `frame_000001.txt`, etc. Empty files mark negative frames. `--frame-step N` evaluates every Nth source frame and expects labels only for those sampled frames.

## Run and compare

From `backend`, use the same test split for both runs:

```powershell
python -m scripts.evaluate_yolo_dataset --images ..\dataset\images --labels ..\dataset\labels --class-map ..\dataset\classes.json --output ..\output\baseline.json
python -m scripts.evaluate_yolo_dataset --images ..\dataset\images --labels ..\dataset\labels --class-map ..\dataset\classes.json --label-confidence 0.60 --image-size 960 --output ..\output\candidate.json
python -m scripts.evaluate_yolo_dataset --video ..\dataset\clip.mp4 --labels ..\dataset\video-labels --class-map ..\dataset\classes.json --output ..\output\video.json
```

Use the same model, detection threshold, matching IoU, class map, and dataset for a fair comparison; change one setting at a time. Report per-class precision/recall/F1 and inspect the confusion matrix, especially broom/mop versus bird/dog/cat. If the optional verifier is installed and configured, add `--verify` to evaluate secondary checks on animal and weapon-like detections. The evaluator's greedy class-agnostic IoU matching uses 0.5 by default.

The output includes:

- Micro precision, recall and F1; per-class precision, recall and F1.
- A confusion matrix with a background row/column for unmatched objects.
- False-positive count (`prediction count - correctly classified matches`) and false-negative count (unmatched or wrongly classified ground-truth objects).
- Unknown rate among geometrically matched detections.
- Mean detector inference latency per image.
- Image false-positive rate: fraction of images containing a wrong-class or unmatched prediction. This is not a negative-image-only statistical false-positive rate.

Image mode evaluates still frames, the primary detector's confidence cutoff, the default abstention policy for animal/weapon-like classes, and optional CLIP verification. Video mode runs ByteTrack plus the same temporal label smoothing and cached crop verification used by the live pipeline. Ensure annotation quality, frame ordering, and category mapping are checked before interpreting results.
