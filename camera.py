import cv2
import numpy as np
from imutils.video import FPS

import UploadModel


class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        # self.video = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        self.video = cv2.VideoCapture('Birds VS Planes Compilation.mp4')
        # self.video = cv2.VideoCapture('flow.mp4')
        self.video.set(3, 1280)
        self.video.set(4, 720)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        fps = FPS().start()

        while True:
            # grab the frame from the threaded video stream and resize it
            # to have a maximum width of 400 pixels
            success, frame = self.video.read()

            # grab the frame dimensions and convert it to a blob
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                         0.007843, (300, 300), 127.5)

            # pass the blob through the network and obtain the detections and
            # predictions
            UploadModel.net.setInput(blob)
            detections = UploadModel.net.forward()
            # loop over the detections
            for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with
                # the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if confidence > UploadModel.args["confidence"]:
                    # extract the index of the class label from the
                    # `detections`, then compute the (x, y)-coordinates of
                    # the bounding box for the object
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # draw the prediction on the frame
                    label = "{}: {:.2f}%".format(UploadModel.CLASSES[idx],
                                                 confidence * 100)
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                  UploadModel.COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, UploadModel.COLORS[idx], 2)
                    cv2.imshow("Frame", frame)
                    key = cv2.waitKey(1) & 0xFF

                    # if the `q` key was pressed, break from the loop
                    if key == ord("q"):
                        break

                    # update the FPS counter
                    fps.update()

    @staticmethod
    def detect_and_draw(frame, detections, h, i, idx, w, label):
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        if endX - startX < w / 3:
            # draw the prediction on the frame
            cv2.rectangle(frame, (startX, startY), (endX, endY), UploadModel.COLORS[idx], 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), UploadModel.COLORS[1], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, UploadModel.COLORS[idx], 2)

            coord_x_centroid = (endX + startX) // 2
            coord_y_centroid = (endY + startY) // 2
            object_centroid = (coord_x_centroid, coord_y_centroid)
            cv2.circle(frame, object_centroid, 1, (0, 250, 250), 5)
