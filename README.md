# Number Detection

Real-time handwritten digit recognition from a webcam. A small CNN trained on a 10-class digit dataset (0-9) predicts the digit written inside a fixed box in the camera view.

## Features

- Trained Keras CNN (`mnistv2.h5`) for digits 0-9
- Webcam capture with a fixed region of interest (ROI) box
- Grayscale preprocessing and prediction on each frame
- Predicted digit shown above the box; press `q` to quit
- `train.py` for retraining on a folder of digit images

## Project structure

- `detect.py` - runs real-time digit recognition from the webcam using `mnistv2.h5`.
- `train.py` - builds a `tf.keras` CNN and trains it on the image folder, then saves `mnistv2.h5`.
- `mnistv2.h5` - pre-trained model used by `detect.py`.
- `Data/digits updated/` - training dataset: one subfolder per digit class (`0` to `9`).
- `DEMO.mp4` - demo video.
- `ReadME.txt` - original short notes.

## Requirements

Python packages imported by the code:

- `numpy`
- `matplotlib` (used by `train.py` for dataset previews)
- `tensorflow` (Keras models and dataset loading)
- `opencv-python` (imported as `cv2`)

Install:

```
pip install numpy matplotlib tensorflow opencv-python
```

## Installation

1. Clone the repository.
2. Install the packages above.
3. Connect a webcam (device 0).

## Usage

Run real-time detection with the included model:

```
python detect.py
```

Hold a handwritten digit so it fills the blue box (fixed at x=100, y=100, 200x200 pixels in the frame). The predicted digit is drawn above the box. Press `q` to exit.

Retrain the model:

```
python train.py
```

`train.py` expects the dataset folder to be set in the script. The `data_dir` variable is currently a hard-coded Windows path (`D:\PythonProject\mnist_detect\digits updated`); change it to point to `Data/digits updated` in this repo. Training uses 128x128 grayscale images, a 20% validation split, 10 epochs, and saves `mnistv2.h5` in the current directory.

## Notes

- `mnistv2.h5` is included so `detect.py` runs without training.
- `detect.py` expects `mnistv2.h5` in the current working directory.
- `train.py` is not runnable as-is on other machines because of the hard-coded dataset path. It also contains a duplicated `visualize_dataset` function and a misspelled `RandomFlip("horizorntal")` argument; the flip mode string is misspelled, which may raise an error or disable the intended flip, so fix it before retraining.
- `detect.py` resizes the ROI to 128x128, which matches the training input size in `train.py`. Its comments mention 28x28, which is outdated.
- The box position is fixed, so the digit must be placed inside it. Recognition quality depends on lighting and the drawn digit resembling the training data.
