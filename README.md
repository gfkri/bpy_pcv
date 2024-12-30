# Renderable Pointcloud Generator for 3D Perception and Blender

---

## Overview

![Screenshot](./doc/screenshot_sm.png)   ![Example Rendering](./doc/egrendering_sm.png)

This repository provides scripting tools to generate blender files in order to render point clouds from the Waymo 3D object detection benchmark. The tool utilizes a Blender file as a template to create an adjustable Blender project setup.
It utilizes the LiDAR camera calibration to place the LiDAR point clouds as well as RGB cameras with the respective image planes. The resulting file allows for individual customization of the rendering setup. 

---

## Features/TODOs

- [x] Generate Blender including renderable point cloud and image planes 
- [x] Preprocessing support for Waymo Open Dataset.
- [ ] Preprocessing support for nuScenes
- [ ] Preprocessing support for KITTI
- [x] Perspective camera model
- [ ] Modelling Radial/Tangential Distortion
- [ ] Modelling Camera Motion
- [ ] Embed object class labels, bounding boxes, and confidence scores into the visualization
---

## Requirements

- Python 3.11
- Blender (`bpy 4.3`)

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pointcloud-generator.git
   cd pointcloud-generator

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```