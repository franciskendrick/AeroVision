# Ground Aircraft Marshalling System
![Title screen of Ground Aircraft Marshalling System](resources/titlescreen.png)
**Ground Aircraft Marshalling System** is a computer vision project designed to detect airplane landing and takeoff marshalling signals using OpenCV, Mediapipe, and a trained LSTM model. It uses real-time camera input to recognize gestures and provide instant feedback through a Pygame interface.

## Requirements
- Python 3.x 
- Required Python libraries:
  * opencv
  * keras
  * mediapipe
  * numpy
  * pygame
  * requests
  * tensorflow

## Installation
### 1. Download Python
- Go to [python.org/downloads](https://www.python.org/downloads/)
- Download the lastest version for your Operating System
- **IMPORTANT:** When installing, check **"Add Python to PATH"**

### 2. Install Dependencies
Open a terminal or command prompt and run:
```
pip install opencv-python keras mediapipe numpy pygame requests tensorflow
```

If `'pip' is not recognized`, try:
```
py -m pip install opencv-python keras mediapipe numpy pygame requests tensorflow
```

Or install them one by one:
```
pip install opencv-python 
pip install keras
pip install mediapipe
pip install numpy
pip install pygame
pip install requests
pip install tensorflow
```

## How to Use
- Run <code style="color: lightgreen;">_main.exe_</code>

### Project Scructure
<pre>
Ground Aircraft Marshalling Simulator/
├── resources/
├── <code style="color: lightgreen;">main.exe</code>
├── model.h5
├── README.md
├── scores.csv
</pre>

## Output  
Scores will be saved in <code style="color: white;">_scores.csv_</code> in this format:  

| Date & Time          | Start Engine | Straight Ahead | Turn Left | Turn Right | Stop | Set Brakes | Chocks Inserted | Cut Engines | All Clear | Overall Score | Status            |
|----------------------|--------------|----------------|-----------|------------|------|------------|-----------------|-------------|-----------|---------------|-------------------|
| 2025-08-22 20:35:12   | 52           | 48              | 54        | 51         | 45   | 65         | 41              | 39          | 48        | 49            | NEEDS IMPROVEMENT |
| 2025-08-23 17:36:45   | 88           | 82              | 87        | 86         | 84   | 95         | 56              | 79          | 77        | 82            | GOOD              |
| 2025-08-24 21:47:16   | 93           | 80              | 97        | 97         | 97   | 99         | 97              | 90          | 97        | 94            | EXCELLENT         |

