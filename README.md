# Face_emotion_recognition

## About:
  This is the project I did for AtumX(Atum Robotics). The main goal of the project to classify human emotion with the image.
  We did this to integrate it into a robot prototype. So that the bot can able to know about the emotion of the people who stand before them and it can able to react accordingly.
![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## Working:
  This is the pre-trained model found in the paper with code. I was written in Python. I used the weights and model architecture to build one for us.
  The modified it for our work and then we used it in our project. The architecture used here is DAN.
![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)
## Library Used:
  - Pytorch
  - Numpy
  - Pandas
    
![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)
## File information:
  - .pth file is the weight file which has the learned parameters for the model
  - face_exp_classifier.ipynb is the original file, it takes input from the native camera.
  - face_exp_realsense.ipynb is the file we modified.
  - model.ipynb is the model architecture.
  
![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png) 
## Execution Instructions:

1. Clone the repository:

   ```
   git clone https://github.com/balaji-89/Face_emotion_recognition.git
   ```

2. Connect the camera  and ensure it is properly configured.

3. Run :
      open file under src/models/face_exp_classifier.ipynb, then run it cell by cell
   
Feel free to modify and adapt the content according to your project's specific details and requirements.



