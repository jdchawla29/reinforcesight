# Datasets Directory

This directory will contain the generated datasets for the spatial reasoning model training and evaluation.

To generate the datasets, run:

```bash
./scripts/generate_dataset.sh
```

This will download images from the LocalizedNarratives dataset, process them using VQASynth, and generate spatial reasoning questions.

## Dataset Format

Each line in the JSONL files contains a single training example with the following fields:

- `question`: The question text
- `answer`: The ground truth answer
- `cot`: Chain-of-thought reasoning explanation
- `task`: The task type ('2d_spatial', '3d_depth', or '3d_distance')
- `image_path`: Path to the image file
- `image_id`: Unique identifier for the image
- `objects`: List of objects involved in the question
- `source_dataset`: Source dataset name (e.g., 'localized_narratives')
- `caption`: Original image caption (if available)
- `narrative`: Original image narrative (if available)