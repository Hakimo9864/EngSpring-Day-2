# Day 1: Images as Data & Thresholding

## Overview

Welcome to your first OpenCV lab! In this assignment, you'll learn that **images are just arrays of numbers** that computers can analyze and manipulate. You'll practice loading images, inspecting their properties, and applying **thresholding** - a simple technique that classifies each pixel as either black or white based on a cutoff value.

Think of thresholding like deciding if a student passes or fails based on a test score cutoff. If their score is above the threshold, they pass (white pixel). If it's below, they fail (black pixel). We'll explore when this simple approach works well and when it falls short.

## Learning Objectives

By completing this lab, you will:

- **Understand images as numerical data** (numpy arrays with height, width, and color channels)
- **Load and inspect images** using OpenCV (`cv2.imread()`)
- **Convert color images to grayscale** to simplify processing
- **Apply binary thresholding** to classify pixels into two categories
- **Analyze results** by calculating what percentage of pixels are classified as "white"
- **Practice reading technical documentation** to implement functions on your own

## Setup

### Prerequisites

You'll need Python 3 and two libraries: OpenCV and NumPy.

### Installation

Install the required packages using pip:

```bash
pip install opencv-python numpy
```

To verify the installation worked, try importing in Python:

```python
import cv2
import numpy as np
print(cv2.__version__)
```

If you see a version number (like `4.5.3`), you're ready to go!

## Your Task

Open [Day_1.py](Day_1.py) - this is your starter file. You'll find **three TODOs** that you need to complete by researching the OpenCV documentation.

### TODO 1: Print Image Information

Print three pieces of information about the loaded image:
- The **filename** (already available as `args.image`)
- The **shape** of the image array (tells you height, width, and number of color channels)
- The **data type** of the pixels (usually `uint8` for 8-bit images)

**Hint:** Use the `.shape` and `.dtype` attributes of the numpy array.

### TODO 2: Print BGR Pixel Values

Extract and print the Blue, Green, and Red values for a specific pixel at coordinates `(100, 100)`.

**Hint 1:** OpenCV stores colors as **BGR** (Blue, Green, Red), not RGB!  
**Hint 2:** Images are indexed as `image[y, x]` where `y` is the row (height) and `x` is the column (width).

### TODO 3: Convert to Grayscale

Convert the color (BGR) image to grayscale using OpenCV's color conversion function.

**Hint:** Look up `cv2.cvtColor()` in the OpenCV documentation. You'll need to specify the conversion type as `cv2.COLOR_BGR2GRAY`.

## Usage Examples

Run the script from your terminal with different options:

### Basic Usage (with default thresholds)

```bash
python Day_1.py --image img/pencil.jpg
```

This loads the pencil image and applies three threshold values: 60, 120, and 180.

### Custom Thresholds

```bash
python Day_1.py --image img/dietcoke.jpg --thresholds 50 100 150
```

Try different threshold values to see how they affect the classification!

### Skip the GUI Display

If the display windows aren't working or you just want to save outputs:

```bash
python Day_1.py --image img/pencil2.jpg --no-gui
```

## What Happens When You Run the Script?

1. **Loads** your image from the `img/` folder
2. **Prints** image information (filename, shape, data type, and pixel values)
3. **Converts** the image to grayscale
4. **Applies thresholding** at each threshold value you specified
5. **Saves outputs** to the `outputs/` folder:
   - `grayscale.png` - your image converted to grayscale
   - `thresh_60.png`, `thresh_120.png`, etc. - binary thresholded versions
6. **Calculates** what percentage of pixels are classified as "white" at each threshold
7. **Displays** the results in windows (unless you use `--no-gui`)

## Testing Your Work

After running the script, check the `outputs/` folder. You should see:
- A grayscale version of your image
- Binary (black and white) versions at different thresholds
- Console output showing percentages of white pixels

### Expected Output Example

```
Image: img/pencil.jpg
Shape: (480, 640, 3)
Data type: uint8
BGR values at (100, 100): B=245, G=242, R=239

Threshold 60: 98.52% white pixels
Threshold 120: 87.34% white pixels
Threshold 180: 45.21% white pixels
```

## Reflection Questions

After completing the lab, think about these questions:

1. **When does thresholding work well?** Can you think of scenarios where a simple cutoff creates good classifications? (Examples: separating text from a white background, detecting bright objects)

2. **When does thresholding fail?** What happens when your image has varying lighting or shadows? Try the different sample images and observe.

3. **What assumptions does binary thresholding make?** Think about what thresholding "believes" about the world. (Hint: It assumes a single number can separate all pixels into two meaningful categories)

4. **How could you improve this classifier?** What additional information could help make better decisions? (Consider: neighboring pixels, multiple color channels, context)

5. **Is thresholding "machine learning"?** Why or why not? Think about the difference between a rule you set manually (like "threshold at 120") versus a rule learned from examples.

## Resources

- **OpenCV Python Documentation**: https://docs.opencv.org/
- **OpenCV Tutorials**: https://docs.opencv.org/master/d6/d00/tutorial_py_root.html
- **NumPy Arrays**: https://numpy.org/doc/stable/user/absolute_beginners.html

## Sample Images

Three sample images are provided in the `img/` folder:
- `dietcoke.jpg` - A can of Diet Coke
- `pencil.jpg` - A pencil on a surface
- `pencil2.jpg` - Another pencil image

Try running your script on all three to see how different images respond to thresholding!

---

**Good luck!** Remember: programming is about problem-solving and learning to read documentation. The OpenCV docs are your friend - don't hesitate to search for functions and explore examples.
