# Skin Care Analysis and Recommendations

This project uses various computer vision and natural language processing techniques to analyze skin issues, predict gender and age, and provide personalized skincare suggestions based on the detected issues.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Functions](#functions)
  - [preprocess_image](#preprocess_image)
  - [predict_gender_age](#predict_gender_age)
  - [detect_spots_yolo](#detect_spots_yolo)
  - [detect_skin_issues](#detect_skin_issues)
  - [get_user_response](#get_user_response)
  - [analyze_response](#analyze_response)
  - [print_with_delay](#print_with_delay)
  - [generate_suggestions](#generate_suggestions)
  - [chat_with_gpt2](#chat_with_gpt2)
  - [main](#main)
- [Dependencies](#dependencies)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/malikmoaz01/FaceInsight-Deep-Learning.git
    cd FaceInsight-Deep-Learning
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the necessary NLTK data:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    ```

4. Ensure you have the necessary model files in the `dataset` directory:
    - `gender_deploy.prototxt`
    - `gender_net.caffemodel`
    - `age_deploy.prototxt`
    - `age_net.caffemodel`
    - `yolov3.weights`
    - `yolov3.cfg`
    - `coco.names`

## Usage

To run the script, execute the following command:

```bash
python project.py
```

Replace `main.py` with the script filename if it differs.

## Functions

### preprocess_image

```python
def preprocess_image(image_path):
```

Loads and preprocesses an image to detect a face. Returns the face region and its coordinates if a single face is detected. Raises an exception if multiple faces are detected.

### predict_gender_age

```python
def predict_gender_age(image_path):
```

Predicts the gender and age range of the person in the image using pre-trained Caffe models.

### detect_spots_yolo

```python
def detect_spots_yolo(image_path):
```

Detects spots in the image using the YOLO object detection model. Returns a boolean indicating whether spots were detected.

### detect_skin_issues

```python
def detect_skin_issues(image_path, face_coordinates, age):
```

Detects various skin issues such as acne, spots, pimples, pigmentation, dark circles, redness, and wrinkles. Returns a dictionary with the detection results.

### get_user_response

```python
def get_user_response(prompt):
```

Prompts the user for a response and returns the input.

### analyze_response

```python
def analyze_response(response):
```

Analyzes the user's response to determine if it is affirmative.

### print_with_delay

```python
def print_with_delay(text, delay=0.009):
```

Prints text with a delay between each character, simulating typing.

### generate_suggestions

```python
def generate_suggestions(issue, gender, age):
```

Generates skincare suggestions based on the detected issue, gender, and age range.

### chat_with_gpt2

```python
def chat_with_gpt2():
```

Engages in a conversation with the user using the GPT-2 model for further assistance.

### main

```python
def main(image_path):
```

Main function that ties all the components together. It processes the image, predicts gender and age, detects skin issues, provides suggestions, and offers chatbot assistance.

## Dependencies

- `opencv-python`
- `numpy`
- `nltk`
- `transformers`
- `torch`

Ensure all dependencies are installed using the provided `requirements.txt` file. For NLTK data, execute the download commands mentioned in the installation section.
