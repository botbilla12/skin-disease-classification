CLASS 12 AI PROJECT: CLASSIFICATION OF SKIN CONDITIONS
====================================================

WHAT THIS PROJECT DOES
- Trains a basic image-classification neural network using your labelled images.
- Saves the trained model.
- Opens a small Streamlit website where you can upload an image and view the model output.

IMPORTANT
This is a school demonstration, NOT a medical diagnostic tool. A model score is not
a diagnosis or a measure of medical certainty. Do not use it for treatment decisions.

1. INSTALL SOFTWARE
- Install Python 3.11 (64-bit) from https://www.python.org/downloads/release/python-3119/
  During installation tick "Add python.exe to PATH".
- Install VS Code from https://code.visualstudio.com/download
- In VS Code, install the Microsoft Python extension.

2. OPEN THE PROJECT
- Extract this ZIP file.
- In VS Code choose File > Open Folder and select skin_disease_ai_project.
- Open Terminal > New Terminal. Use PowerShell.

3. CHECK PYTHON
Run:
    py -3.11 --version
If that command does not work, try:
    python --version
If neither works, reinstall Python and tick "Add python.exe to PATH", then close and
reopen VS Code.

4. CREATE AND ACTIVATE A VIRTUAL ENVIRONMENT
Run these commands one at a time in the VS Code PowerShell terminal:
    py -3.11 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install --upgrade pip
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt

You do NOT need to activate the environment. Calling its Python directly avoids
PowerShell execution-policy problems.

5. ADD YOUR IMAGE DATASET
Inside the project folder, create folders like this (the names become the model classes):

    dataset/
        acne/
            image001.jpg
            image002.jpg
        eczema/
            image001.jpg
            image002.jpg
        psoriasis/
            image001.jpg
            image002.jpg

You can use different class names, but use one folder per label. Put only images
belonging to that label inside its folder. Use a reasonably balanced set of images
for every class; aim for at least 50-100 images per class for a classroom experiment.
More high-quality, correctly labelled images are better. Do not mix test images into
training if you are preparing a separate test set. Check dataset licence and source.

6. TRAIN THE MODEL
In the VS Code terminal run:
    .\.venv\Scripts\python.exe train.py

Wait until training finishes. It creates:
    skin_disease_model.keras
    class_names.json

7. START THE WEBSITE
Run:
    .\.venv\Scripts\python.exe -m streamlit run app.py

A browser tab should open. Upload a JPG or PNG image to see the model's predicted
training class and scores. To stop the website, click the terminal and press Ctrl+C.

COMMON PROBLEMS
- "python was not found": use the .venv\Scripts\python.exe commands above, and
  confirm Python 3.11 is installed with `py -3.11 --version`.
- "No such file or directory: dataset": make sure you opened the extracted project
  folder in VS Code and created the class folders inside dataset.
- "Found 0 files": put JPG/PNG images inside each class folder, not directly in dataset.
- "No module named tensorflow": install requirements using the exact .venv command above.
- PowerShell blocks Activate.ps1: do not activate; use .venv\Scripts\python.exe directly.

PROJECT FILES
- train.py: trains and saves the image classifier.
- app.py: Streamlit interface for uploading an image.
- requirements.txt: Python package list.
- dataset/: add your labelled training images here.
