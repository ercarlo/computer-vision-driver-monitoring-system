# Driver Monitoring System using Computer Vision

A real-time **Driver Monitoring System (DMS)** developed in Python using computer vision to detect driver distraction, fatigue and loss of attention from a standard webcam.

The system combines **facial landmark detection, gaze estimation, eye-closure analysis and remote photoplethysmography (rPPG)** to monitor the driver's state and estimate heart rate without requiring dedicated physical sensors.

---

## Overview

Driver distraction and fatigue are major safety concerns in road transport. Driver Monitoring Systems aim to detect potentially dangerous driver states before they result in an accident.

This project implements a modular DMS capable of monitoring the driver in real time using only a **standard laptop webcam and open-source software**.

The system identifies five possible driver states:

- **Focused on the road**
- **Distracted (long)**
- **Distracted (short)**
- **Microsleep**
- **Sleep**

Additionally, it provides a webcam-based **heart rate estimation in BPM**.

---

## Demo

A demonstration of the complete system is available here:

▶️ [Watch demo video](output/demo.avi)

The interface displays facial landmarks, the forehead region used for heart-rate estimation, the current driver state and the estimated heart rate in real time.

---

## Main Features

### Face Tracking

Facial landmarks are detected using **MediaPipe Face Landmarker**.

The model returns **478 facial landmarks**, from which the system extracts the information required for gaze estimation, eye-closure detection and forehead localisation.

Relevant landmarks include:

- Nose tip
- Eye contour landmarks
- Iris centres
- Forehead region

If no face is detected, the application displays a warning and continues running normally.

---

### Gaze and Attention Estimation

Driver attention is estimated from the horizontal displacement of the nose.

At startup, the system performs a **3-second calibration period** while the driver looks straight ahead. The average horizontal position of the nose is stored as the reference position.

After calibration, the driver is considered to be looking away when the nose deviates by more than:

```text
8 % of the image width
```

This lightweight method allows large lateral head movements to be detected without requiring full 3D head-pose estimation.

---

### Distraction Detection

Two different distraction patterns are monitored.

#### Long distraction

The driver is classified as:

```text
Distracted (long)
```

when looking away continuously for at least:

```text
5 seconds
```

The state is cleared after the driver looks forward continuously for at least:

```text
2 seconds
```

#### Short / repeated distraction

Repeated short glances are detected using a **30-second rolling window**.

The state changes to:

```text
Distracted (short)
```

when the accumulated time looking away reaches:

```text
10 seconds within a 30-second window
```

A Python `deque` is used to maintain the rolling time window efficiently.

---

### Eye Closure Detection

Eye closure is measured using the **Eye Aspect Ratio (EAR)**.

For six landmarks surrounding each eye:

```text
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 × ||p1 - p4||)
```

The threshold used to determine whether the eyes are closed is:

```text
EAR < 0.20
```

Two fatigue-related states are detected:

| State | Condition |
|---|---|
| Microsleep | Both eyes closed for ≥ 4 s |
| Sleep | Both eyes closed for ≥ 7 s |

Both states require the eyes to remain open for at least **2 seconds** before the system considers the driver awake again.

---

### Heart Rate Estimation

The driver's heart rate is estimated without physical sensors using **remote photoplethysmography (rPPG)**.

A region of interest is extracted from the forehead and the average intensity of the **green image channel** is recorded over time.

After collecting approximately **8 seconds of signal**, the following processing pipeline is applied:

1. Mean subtraction and detrending
2. 4th-order Butterworth band-pass filtering
3. Frequency range limited to **0.75–3.0 Hz**
4. Hanning window
5. Fast Fourier Transform (FFT)
6. Dominant-frequency detection
7. Conversion from frequency to BPM

The analysed frequency range corresponds approximately to:

```text
45 – 180 BPM
```

Under good lighting conditions and limited movement, the webcam-based estimation produced typical differences of approximately **5–10 BPM** compared with a manually measured reference pulse.

---

## Driver State Priority

When several conditions could potentially be active, the system uses the following priority:

| Priority | State | Trigger |
|---:|---|---|
| 1 | Sleep | Eyes closed ≥ 7 s |
| 2 | Microsleep | Eyes closed ≥ 4 s |
| 3 | Distracted (long) | Looking away ≥ 5 s continuously |
| 4 | Distracted (short) | Looking away ≥ 10 s within 30 s |
| 5 | Focused | None of the previous conditions |

---

## System Architecture

The project follows a modular architecture in which each monitoring task is implemented independently.

```text
Webcam
   │
   ▼
Face Landmark Detection
   │
   ├──► Gaze Estimation
   │        │
   │        ▼
   │    Distraction Logic
   │
   ├──► Eye Closure Detection
   │
   └──► Forehead ROI
            │
            ▼
     Heart Rate Estimation
            │
            ▼
        Video Overlay
            │
            ▼
        Display / Output
```

This structure makes it possible to modify or replace individual algorithms without redesigning the complete application.

---

## Project Structure

```text
.
├── dms/
│   ├── __init__.py
│   ├── distraction_logic.py
│   ├── eye_closure_detector.py
│   ├── face_tracker.py
│   ├── gaze_detector.py
│   ├── heartbeat_estimator.py
│   └── video_overlay.py
│
├── output/
│   └── demo.mp4
│
├── DMS_Report_CarloSquarcia.pdf
├── download_model.py
├── face_landmarker.task
├── requirements.txt
├── runDMS.py
└── README.md
```

### Modules

- **`face_tracker.py`**  
  Face detection and extraction of MediaPipe facial landmarks.

- **`gaze_detector.py`**  
  Gaze and head-direction estimation based on calibrated nose displacement.

- **`distraction_logic.py`**  
  Temporal logic used to detect continuous and repeated driver distraction.

- **`eye_closure_detector.py`**  
  EAR-based eye closure monitoring and detection of microsleep and sleep.

- **`heartbeat_estimator.py`**  
  rPPG signal acquisition and frequency-domain heart-rate estimation.

- **`video_overlay.py`**  
  Real-time visualisation of landmarks, driver state and BPM.

- **`runDMS.py`**  
  Main application that initialises all modules and executes the real-time processing loop.

---

## Technologies

The project was developed using:

- **Python 3**
- **OpenCV**
- **MediaPipe**
- **NumPy**
- **SciPy**
- Computer Vision
- Facial Landmark Detection
- Digital Signal Processing
- Remote Photoplethysmography (rPPG)
- Fast Fourier Transform (FFT)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/driver-monitoring-system.git
cd driver-monitoring-system
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the MediaPipe model

If the model is not already available, run:

```bash
python download_model.py
```

The application uses the MediaPipe Face Landmarker model:

```text
face_landmarker.task
```

---

## Usage

Connect or enable a webcam and run:

```bash
python runDMS.py
```

During the first **3 seconds**, look straight at the camera so the system can calibrate the reference head position.

After calibration, the application continuously evaluates:

- Driver attention
- Repeated distraction
- Prolonged distraction
- Eye closure
- Microsleep
- Sleep
- Heart rate

Press the corresponding exit key defined in the application to stop execution.

---

## Main Parameters

| Parameter | Value | Purpose |
|---|---:|---|
| `CALIBRATION_TIME` | 3.0 s | Initial forward-looking calibration |
| `NOSE_THRESHOLD` | 0.08 | Maximum accepted horizontal nose deviation |
| `LONG_DISTRACTION_TIME` | 5.0 s | Continuous distraction threshold |
| `SHORT_DISTRACTION_WINDOW` | 30.0 s | Rolling distraction window |
| `SHORT_DISTRACTION_ACCUM` | 10.0 s | Accumulated distraction threshold |
| `RECOVERY_FORWARD_TIME` | 2.0 s | Attention recovery confirmation |
| `EYE_AR_THRESHOLD` | 0.20 | Eye closure EAR threshold |
| `MICROSLEEP_TIME` | 4.0 s | Microsleep threshold |
| `SLEEP_TIME` | 7.0 s | Sleep threshold |
| `EYE_OPEN_RECOVERY_TIME` | 2.0 s | Wakefulness confirmation |
| `HEART_RATE_WINDOW` | 8.0 s | Signal window for BPM estimation |

---

## Results

The system was evaluated in a controlled indoor environment using a standard laptop webcam.

All five defined driver states were individually tested and successfully detected:

```text
Focused
Distracted (long)
Distracted (short)
Microsleep
Sleep
```

The heart-rate module was also able to provide a BPM estimate after approximately **8 seconds of signal acquisition** under suitable lighting conditions.

---

## Limitations

This implementation was designed as a proof of concept and has several limitations.

### Gaze estimation

The current method is mainly sensitive to horizontal head movement. It does not reliably detect:

- Vertical head movements
- Eye-only lateral glances
- More subtle changes in gaze direction

A more advanced implementation could use **3D head-pose estimation and a PnP-based approach**.

### Heart rate estimation

Webcam-based rPPG is highly sensitive to:

- Illumination changes
- Driver movement
- Camera compression
- Automatic exposure
- Automatic white balance

More advanced signal-processing and motion-compensation techniques would improve robustness.

### Real driving environments

The current system was evaluated in a static indoor environment.

Real vehicles introduce additional challenges including:

- Rapid illumination changes
- Shadows
- Vehicle vibration
- Driver movement
- Different camera positions
- Occlusions
- Sunglasses

A production-level system would require more robust algorithms and validation using real driving data.

---

## Possible Future Improvements

Potential extensions of the project include:

- 3D head-pose estimation
- Iris-based gaze tracking
- Yawning detection
- Blink-frequency analysis
- Adaptive EAR thresholds
- Improved rPPG signal processing
- Motion compensation for heart-rate estimation
- Low-light driver monitoring
- Sunglasses and occlusion handling
- Audio or visual driver alerts
- Real-vehicle testing
- Machine-learning-based driver state classification

---

## Report

A detailed explanation of the design, implementation, parameters and results is available in the project report:

📄 [DMS Report](DMS_Report_CarloSquarcia.pdf)

---

## Context

This project was developed as part of the **Autonomous Vehicles** course during the **2024/2025 academic year**.

It demonstrates the implementation of a low-cost Driver Monitoring System using only a conventional webcam and open-source computer vision and signal-processing tools.

---

## Author

**Carlo Squarcia Mateo**

Engineering student interested in robotics, artificial intelligence, computer vision and autonomous systems.
