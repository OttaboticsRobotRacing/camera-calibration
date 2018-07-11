# ```calibration.py``` Usage

## Imports
```python
import cv2
import numpy as np
from calibration import CameraCalibration
```

## Define ```profile```
```python
DIM = (640, 480)
K = np.array([[359.0717640266508, 0.0, 315.08914578097387], [0.0, 358.06497428501837, 240.75242680088732], [0.0, 0.0, 1.0]])
D = np.array([[-0.041705903204711826], [0.3677107787593379], [-1.4047363783373128], [1.578157237454529]])

profile = (DIM, K, D)
```

## Initialize ```CameraCalibration``` object
```python
CC = CameraCalibration(profile)
```

## Undistort image
```python
distorted = cv2.imread('test.jpg')
undistorted = CC.undistort(distorted)
```