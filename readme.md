# Centroid Tracker

This repository contains a basic 2D centroid tracker. This centroid tracker can be used alongside an object detector to follow an object across video frames.
The approach is fairly naive and may begin to track the wrong object if they are moving quickly or if the scene changes. However, it works well for simple situations.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Example Notebook](#example-notebook)

## Overview

The Centroid Tracker uses a simple yet effective approach to track objects by assigning unique IDs to new objects, updating their positions, and removing them if they disappear for too long. This tracker can be used in various computer vision applications, such as tracking people or vehicles in a video stream.

## Installation

To get started with the Centroid Tracker, clone the repository and install the necessary dependencies.

```sh
git clone https://github.com/LukeBilsborrow/BasicCentroidTracker.git
cd centroid-tracker
pip install -r requirements.txt
```

## Usage

```python
from CentroidTracker import CentroidTracker

# Simulated bounding box data (top, right, bottom, left)
# a box at the top left corner of the frame, with a width and height of 2
starting_bbox = [0, 2, 2, 0]

bbox_data = [
    # frame 1
    # a box starting at x=0 and y=0 with a width and height of 2
    [[0, 2, 2, 0]],
    # frame 2
    [[1, 3, 3, 1]],
    # frame 3
    [[1, 3, 3, 1]],
    # frame 4
    # add a new unrelated box
    [[2, 4, 4, 2], [5, 7, 7, 5]],
    # frame 5
    # swap the input order of the boxes
    # the correct box will still be tracked
    [[6, 8, 8, 6], [3, 5, 5, 3]],
]

tracker = CentroidTracker(max_disappeared=50)
target_id = tracker.register(starting_bbox)
print(f"Target ID: {target_id}")

for frame in bbox_data:
    tracker.update(frame)
    for object_id in tracker.objects.keys():
        bbox = tracker.get_bounding_box(object_id)
        print(f"Object ID {object_id}: Bounding Box {bbox}")

```

## Example Notebook

To see the Centroid Tracker in action, you can open the example.ipynb notebook on [Google Colab](https://colab.research.google.com/). This notebook demonstrates how to use the Centroid Tracker to track objects in a video, specifically Jim Carrey's face. You can modify the notebook to track different objects or use your own video files.
