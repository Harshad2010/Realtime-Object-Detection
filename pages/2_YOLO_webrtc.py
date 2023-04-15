import streamlit as st 
from streamlit_webrtc import webrtc_streamer
import av
from src.prediction.yolo_prediction import YOLO_Pred
import cv2

#load yolo model
yolo = YOLO_Pred(onnx_model='src/model_training/yolov5/runs/train/Model/weights/best.onnx',
                    data_yaml='src/model_training/yolov5/data.yaml')

def run_video_capture():
    # Set the path of the video file
    video_path = st.file_uploader("Upload a video file", type=["mp4", "mov"]
    #cap = cv2.VideoCapture(0)
    if video_path == False:
        # Create a VideoCapture object to read the video file
        video_capture = cv2.VideoCapture(video_path)
    while True:
        # Read the next frame from the video file
        ret, frame = video_capture.read()
         # If the frame was not successfully read, break out of the loop
        if ret == False:
            print("Unable to read video")
            break
        pred_image = yolo.predictions(frame)
        
        # Display the frame in the Streamlit app
        st.image(pred_image, channels="BGR")
            
                  
def main():
    st.title("Video Capture App")
    run_video_capture()

if __name__ == '__main__':
    main()


