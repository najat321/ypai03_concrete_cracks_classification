# Concrete Cracks Classification

## Project Description
### i.	What this project does?
This project is to perform image classification to classify concretes with or without cracks applying transfer learning.
### ii.	Any challenges that was faced and how I solved them?
- There were some challenges in the data preparation process where there were no train, test and validation dataset given. I solved them splitting the data into train, test and validation dataset. 
- Data augmentation model was created to artificially increase the size of the training dataset. 
- Transfer learning was applied to achieve good performance and reducing the need for extensive training on expensive hardware.
### iii.	Some challenges / features you hope to implement?
I hope I could explore different pre-trained models and architectures that are suitable for this image classification task. For this project, I chose MobileNetV3Large. There are other numerous pre-trained models available, such as VGG, ResNet, Inception, and MobileNet, each with their own strengths and trade-offs. I could choose other models that aligns with the size of this dataset and the complexity of this classification problem to explore with.
## How to install and run the project 
Here's a step-by-step guide on how to install and run this project:

1. Install Python: Ensure that Python is installed on your system. You can download the latest version of Python from the official Python website (https://www.python.org/) and follow the installation instructions specific to your operating system.

2. Clone the repository: Go to the GitHub repository where your .py file is located. Click on the "Code" button and select "Download ZIP" to download the project as a ZIP file. Extract the contents of the ZIP file to a location on your computer.

3. Set up a virtual environment (optional): It is recommended to set up a virtual environment to keep the project dependencies isolated. Open a terminal or command prompt, navigate to the project directory, and create a virtual environment by running the following command: python -m venv myenv

   Then, activate the virtual environment:

   If you're using Windows: myenv\Scripts\activate

   If you're using macOS/Linux: source myenv/bin/activate

4. Install dependencies: In the terminal or command prompt, navigate to the project directory (where the requirements.txt file is located). Install the project dependencies by running the following command: pip install -r requirements.txt

   This will install all the necessary libraries and packages required by the project.

5. Run the .py file: Once the dependencies are installed, you can run the .py file from the command line. In the terminal or command prompt, navigate to the project directory and run the following command: python your_file.py

   Now, you're done! The project should now run, and you should see the output or any other specified behavior defined in your .py file.

## Output of this project
#### i. Model Accuracy:

![Alt Text](https://raw.githubusercontent.com/najat321/ypai03_concrete_cracks_classification/main/Model%20Accuracy.PNG)

#### ii. Model Architecture:

![Alt Text](https://raw.githubusercontent.com/najat321/ypai03_concrete_cracks_classification/main/Model%20Architecture.PNG)

#### iii. Transfer Learning Model Architecture:

 ![Alt Text](https://raw.githubusercontent.com/najat321/ypai03_concrete_cracks_classification/main/Model%20Architecture_2%20-Follow-up%20training.PNG)
 
#### iv. Model Evaluation Before Training:

 ![Alt Text](https://raw.githubusercontent.com/najat321/ypai03_concrete_cracks_classification/main/Model%20Evaluation%20after%20training.PNG)
 
#### iv. Model Evaluation After Training:

 ![Alt Text](https://raw.githubusercontent.com/najat321/ypai03_concrete_cracks_classification/main/Model%20Evaluation%20before%20training.PNG)

## Source of datasets : 
https://data.mendeley.com/datasets/5y9wdsg2zt/2 
