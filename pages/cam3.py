import cv2
import glob
import argparse

from ultralytics import YOLO
import supervision as sv
import numpy as np
from pygame import mixer

import streamlit as st
import tempfile


ZONE_POLYGON = np.array([
    [181, 321],
    [378, 310],
    [379, 360],
    [553, 334],
    [544, 274],
    [907, 227],
    [996, 363],
    [895, 390],
    [881, 435],
    [582, 491],
    [570, 435],
    [375, 459],
    [371, 541],
    [170, 551]
])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution", 
        default=[1920, 1080], 
        nargs=2, 
        type=int
    )
    args = parser.parse_args()
    return args

st.title("Мониторинг опасных зон. Камера Pgp-lpc2-K-0-1-38")
frame_placeholder = st.empty()
stop_button_pressed = st.button("Stop")
st.sidebar.success("Выберте камеру") 


def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    #cap = cv2.VideoCapture('img1.jpg')
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8l.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    zone_polygon = (ZONE_POLYGON).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple(args.webcam_resolution))
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone, 
        color=sv.Color.red(),
        thickness=2,
        text_thickness=4,
        text_scale=2
    )
        

    #while True:
    #    ret, frame = cap.read()

    path = "images/i3/*.*"
    for file in glob.glob(path):
        frame = cv2.imread(file)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[detections.class_id == 0]


        labels = [
            result.names[class_id]
            for class_id
            in detections.class_id
        ]


        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        )

        zone.trigger(detections=detections)
        frame = zone_annotator.annotate(scene=frame)
        if zone.current_count > 0:
            mixer.init()
            sound = mixer.Sound("alarm5.mp3")
            sound.play()
 #        

        frame_placeholder.image(frame, channels="RGB")
        
        #cv2.imshow('Color image', frame)
        #k = cv2.waitKey(1000)
        #cv2.destroyAllWindows()
        
        

if __name__ == "__main__":
    main()