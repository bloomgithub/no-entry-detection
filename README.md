# No-Entry Sign Detection System

A phython computer vision system for detecting no-entry signs using Viola-Jones object detection, circle detection, and red colour analysis.

## Prerequisites

- Python 3.6
- OpenCV 3.4.x
- NumPy

## Environment Setup

Create virtual environment with Python 3.6:
```bash
conda create -n ipcv36 python=3.6
```

Activate your environment:
```bash
conda activate ipcv36
```

Install OpenCV packages:
```bash
conda install -c menpo opencv
```

Check OpenCV version (should be 3.4.x):
```bash
python -c 'import cv2; print(cv2.__version__)'
```

## Installation

```bash
git clone https://github.com/yourusername/no-entry-detection.git
cd no-entry-detection
```

### Using Pre-trained Model

The classifier file can be found in:
```
NoEntrycascade/
└── cascade.xml
```

### Training Custom Model

1. Generate training samples:
```bash
opencv_createsamples -img no_entry.jpg -vec no_entry.vec -w 20 -h 20 -num 500 -maxidev 80 -maxxangle 0.8 -maxyangle 0.8 -maxzangle 0.2
```

2. Train classifier:
```bash
opencv_traincascade -data NoEntrycascade -vec no_entry.vec -bg negatives.dat -numPos 500 -numNeg 500 -numStages 3 -maxDepth 1 -w 20 -h 20 -minHitRate 0.999 -maxFalseAlarmRate 0.05 -mode ALL
```

## Usage

### Single Image Detection

```bash
python detector.py --image path/to/your/image.jpg
```

Output: `detected.jpg` with marked detections

### Evaluation Mode

```bash
python detector.py --method [detection_method]
```

Methods:
- `vj`: Viola-Jones only
- `vj_circle`: Viola-Jones + circle detection
- `vj_circle_red`: Viola-Jones + circle detection + red color detection

Required structure:
```
.
├── No_entry/           # Test images
├── annotations.txt     # Ground truth
└── NoEntrycascade/    # Cascade classifier
    └── cascade.xml
```

Outputs:
- `out_{method}/`: Detection visualizations
- `debug_{method}/`: Processing step images
- Evaluation metrics (TPR, F1, false positives/negatives)

## Annotation Format

```
image_name.jpg num_signs x1 y1 w1 h1 x2 y2 w2 h2 ...
```

Where:
- `image_name.jpg`: Image filename
- `num_signs`: Number of signs
- `x y w h`: Bounding box coordinates and dimensions

### Creating Custom Annotations

1. Use OpenCV's annotation tool:
```bash
opencv_annotation --annotations=annotations.txt --images=path/to/image/directory
```

2. Controls:
- Click and drag to draw bounding boxes
- Press 'c' to confirm the current box
- Press 'n' for next image
- Press 'esc' to exit

Example:
```bash
opencv_annotation --i=/path/to/No_entry --a=/path/to/annotations.txt
```

## License

This project is licensed under the MIT License.
