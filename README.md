# UOK - AINT44052: Visual Navigation System

## 🎯 Project Overview
This repository contains the code and report for our group assignment in **Intelligent Autonomous Robotics (AINT44052)**. We are designing a vision-based navigation system where a simulated robot detects a target object and navigates to it.


## 👥 Team Members & Roles
| Name | Student ID | Role |
| :--- | :--- | :--- |
| R.M.H.D. Ranathunga | CS/2019/060 | Project Lead & Documentation |
| A.M.P.R. Malindu | CS/2019/024 | Object Detection |
| R.W.M.N. Dilruksha | CS/2019/062 | Localization & Motion Simulation |
| A.M.R.H. Andrady | CS/2019/042 | System Integration |


## 📁 Repository Structure
data/  # Sample images and generated test worlds
results/  # Generated plots, screenshots, and visuals for the report
src/  # All Python source code
README.md  # This file
requirements.txt  # Required libraries


## 🚀 Technology Stack
- **Language:** Python 3
- **Libraries:** OpenCV, NumPy, Matplotlib
- **Dataset:** Fruits-360 (Target: `Apple Red 1`)
- **Simulation:** Custom 2D simulation (500x500 px)


## 🧪 How to Run
1. Clone the repository
2. Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
3. Navigate to the src/ directory to run individual scripts
    ```bash
   python object_detection.py
   python robot_simulation.py
   python main_integration.py