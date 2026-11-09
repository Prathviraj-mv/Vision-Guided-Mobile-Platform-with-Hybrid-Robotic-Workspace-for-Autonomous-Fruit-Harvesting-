# Autonomous Fruit Harvesting Robot

An intelligent robotic system for autonomous apple harvesting using computer vision, inverse kinematics, and real-time hardware control.

## Overview

This project implements a complete autonomous fruit harvesting system featuring:
- **Computer Vision**: YOLOv8-based apple detection using ONNX runtime
- **Robot Control**: RPR (Revolute-Prismatic-Revolute) manipulator with inverse kinematics
- **Hardware Interface**: Arduino-based servo and stepper motor control
- **Autonomous Operation**: Rail scanning, detection, and automated picking
- **Visualization**: Real-time Pygame simulator with live camera feed

## Description

The Autonomous Fruit Harvesting Robot is designed to automatically detect and harvest apples using computer vision and robotic manipulation. The system uses a YOLOv8 neural network trained to identify red apples in real-time, controlling an RPR-configured robotic arm with 3 DOF (degrees of freedom) mounted on a vertical rail. The robot can operate in manual mode via mouse control or autonomous scanning mode where it systematically searches for apples and performs picking operations when detected. The control software provides a Pygame-based visualization showing the robot state, camera feed with detection zones, and real-time joint angles.

## Images

### Robot Build


<p>
  <img src="images/WhatsApp Image 2026-11-07 at 19.57.15 (3).jpeg">
</p>

![Robot Build Detail](images/WhatsApp%20Image%202026-11-07%20at%2019.57.15%20(2).jpeg)

### CAD Designs

![Pygame Simulator](images/Screenshot%202026-11-06%20155337.png)

![Detection View](images/Screenshot%202026-11-06%20161828.png)




## Project Structure

```
.
├── arduino/              # Arduino firmware
│   ├── sketch_nov6a.ino  # BTS7960 motor controller
│   └── sketch_nov6b.ino  # Main robot controller (servos + rail)
├── python/               # Python control software
│   ├── main.py           # RPR inverse kinematics
│   ├── apple.py          # Apple detection (standalone)
│   ├── sneder.py         # Main control loop with Pygame GUI
│   └── transmission.py   # Video streaming to Raspberry Pi
├── models/               # ML models
│   └── best_model2.onnx  # YOLOv8 apple detection model
├── docs/                 # Documentation
│   └── Autonomous Fruit Harvesting.pdf
├── cad/                  # CAD designs (placeholder)
├── images/               # Project images (placeholder)
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Usage

Run the main control script:
```bash
python python/sneder.py
```

**Controls**:
- **Y**: Arm/Disarm robot (enables mouse control)
- **I**: Start/stop autonomous scanning
- **Mouse**: Move robot end-effector in workspace
- **ESC**: Exit
