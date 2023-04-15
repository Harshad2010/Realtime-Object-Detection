# Real Time Object Detection

## Introduction:
This project is focused on detecting objects in images and videos using Deep Learnign and computer vision techniques. The project uses a pre-trained YOLOV5 model and fine-tunes it on a custom dataset.

## Data:
The dataset used in this project is custom dataset of 5,012 images with 20 different classes.The images are of varying sizes and have been pre-processed to have consistent size and quality at the input of YOLOV5 model. The dataset is divided into a training set (4,010 images) and test set ( 1,002 images).

Data Source:

## Model:
YOLOv5 (You Only Look Once version 5) is a state-of-the-art object detection model that was introduced in 2020 by the developers of the YOLO (You Only Look Once) model series.

The YOLOv5 model is based on a single-stage object detector architecture, which means it performs the object detection task in a single forward pass through the network, unlike two-stage architectures such as Faster R-CNN. This makes YOLOv5 faster than other models while still maintaining high accuracy.

The YOLOv5 model uses a backbone network, which is a pre-trained feature extractor, that is based on the CSPDarknet53 architecture. The backbone is followed by several neck and head layers that perform the task of predicting the class and location of objects in the image.

The YOLOv5 model also uses a new anchor-free detection method called "ATSS" (Anchor-free Two-Stage Detection), which improves the model's accuracy and generalization ability.

Additionally, the YOLOv5 model also uses data augmentation techniques, such as Mosaic data augmentation, CutMix, and Mixup, to improve the robustness and generalization of the model.

Overall, YOLOv5 is a powerful and efficient object detection model that is suitable for real-time applications.

## Evaluation:
The model's performance is evaluated using metrics such as confusion matrix, precision, recall, and F1-score. The evaluation result saved in model evaluation results directory.

## Results
The results of the object detection are displayed in the form of bounding boxes around the detected objects and the corresponding class labels.

## Instructions to run the Project:
1. Clone the repository.
```
git clone 
```
3. Create new enviroment with python==3.8

4. run command
```
python setup.py install
```
5. Download the data from provided link, unzip it and store it in the src folder.

data source: provide link

6. Install the required libraries mentioned in the requirements.txt file
``` 
pip install -r requirements.txt
```
8. Go to model_training folder. 
```
cd src\model_training
```
9. clone yolov5 model.
```
git clone https://github.com/ultralytics/yolov5.git
```
10. Run main.py
```
python main.py
```
This command will preprocess all the images and xml files (used for bounding box) and save it in "Real_Time_Object_Detection\src\model_training\yolov5\src" directory, so that it data can be used for training.

11. Training YOLO v5 model. Run below command in the terminal.
```
python train.py --data data.yaml --cfg yolov5s.yaml --batch-size 8 --name Model --epochs 50
```
Once training is completed best weights and last weights of trained model are saved in "Real_Time_Object_Detection\src\model_training\yolov5\runs\train\Model\weights" directory 
and evaluation results are saved in "Real_Time_Object_Detection\src\model_training\yolov5\runs\train\Model" directory.

12. convert saved model from pytoch format to onnx format. So that we can use it with opencv.
```
python export.py --weights runs/train/Model/weights/best.pt --include onnx --simplify
```
13. Model is deployed using streamlit. Run "streamlit run app.py" command.

Note:This is a basic object detection project which can be used as a starting point for more complex projects. You can add more features and improve the performance by using more advanced techniques.

## Demo Video:

https://user-images.githubusercontent.com/90838133/218655151-b176b6e3-544e-4553-922d-8534bdaaa491.mp4

## Below are some results:


![img2](https://user-images.githubusercontent.com/90838133/218264692-50de0bbf-60f2-4829-abf0-0014b570ea35.jpg)
![img6](https://user-images.githubusercontent.com/90838133/218265305-fc479ef9-f554-4c77-82c4-04531caa928c.jpg)
![img8](https://user-images.githubusercontent.com/90838133/218265339-7bea13af-d9db-4ded-94df-e5160c770ad1.jpg)
![img4](https://user-images.githubusercontent.com/90838133/218265365-6e98d2e3-5bfd-4450-b0ff-280e8f74d7b8.jpg)
