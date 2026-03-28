ByteFlow expects a flat image-folder layout:

  <DATASET_ROOT>/
    <class_name>/
      *.jpg, *.jpeg, *.png, or *.webp

Only image files directly inside each class folder are used (no recursive subfolders in v1).

This folder includes tiny placeholder PNGs in class_a/ and class_b/ so you can run
`python train.py` from the project root without preparing your own data.

Replace these with your own compressed images; keep one subfolder per class.
