# Face Mask Detection Model Deployment

Welcome to the Face Mask Detection Model Deployment repository! This repository contains the code and resources needed to deploy a face mask detection model using Python. The model can detect whether individuals in images or video streams are wearing face masks.

## Prerequisites

Before running the project, ensure you have the following prerequisites installed:
```
altgraph==0.17.3
future==0.18.3
numpy==1.24.1
opencv-contrib-python==4.7.0.68
opencv-python==4.7.0.68
pefile==2022.5.30
Pillow==9.4.0
pyinstaller==5.7.0
pyinstaller-hooks-contrib==2022.14
pywin32==305
pywin32-ctypes==0.2.0
```

You can install these dependencies using `pip`. For example:
```
pip install altgraph==0.17.3
pip install future==0.18.3
pip install numpy==1.24.1
pip install opencv-contrib-python==4.7.0.68
pip install opencv-python==4.7.0.68
pip install pefile==2022.5.30
pip install Pillow==9.4.0
pip install pyinstaller==5.7.0
pip install pyinstaller-hooks-contrib==2022.14
pip install pywin32==305
pip install pywin32-ctypes==0.2.0
```

Alternatively, you can install all dependencies at once by running:

```bash
pip install -r requirements.txt
```

## Project Setup

1. **Clone the Repository**: First, clone this repository to your local machine:

    ```bash
    git clone https://github.com/likithsshetty/Face-Recognition-Attendance-System.git
    cd Face-Recognition-Attendance-System
    ```

2. **Install Required Packages**: Navigate to the project directory and install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare the Model**: Ensure that you have the trained face mask detection model file (`model.h5`) in the project directory. If you don't have this file, you may need to train the model first or obtain it from a reliable source.

4. **Run the Deployment Script**: To run the face mask detection on a video stream or image, execute the deployment script:

    ```bash
    python main.py
    ```

    This script will start the face mask detection process, utilizing your webcam or a specified video file.

## Usage

The deployment script, `main.py`, supports various options and configurations. For detailed usage instructions, refer to the [Usage section](#usage) in the README file of the repository.

## Acknowledgements

This project was inspired by the need for effective face mask detection during the COVID-19 pandemic. Special thanks to all the contributors and the open-source community for their valuable resources and support.

## Contact

If you have any questions, issues, or suggestions, feel free to open an issue or contact the repository owner at likithsuresh22@gmail.com.
```

Feel free to adjust any sections or add more details according to your project's requirements.
