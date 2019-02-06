import cv2
import imutils
import numpy as np
from imutils.video import FPS

import UploadModel


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture('Of87U5mAwPg.mp4')
        self.video.set(3, 640)
        self.video.set(4, 480)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        fps = FPS().start()

        while True:
            # grab the frame from the threaded video stream and resize it
            # to have a maximum width of 400 pixels
            success, frame = self.video.read()
            frame = imutils.resize(frame, width=400)
            # grab the frame dimensions and convert it to a blob
            # (h, w) = frame.shape[:2]
            h = np.size(frame, 0)
            w = np.size(frame, 1)
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
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
                    if UploadModel.CLASSES[idx] == "aeroplane" \
                            or UploadModel.CLASSES[idx] == "bird":
                        label = "{}: {:.2f}%".format(UploadModel.CLASSES[idx], confidence * 100)
                        self.detect_and_draw(frame, detections, h, i, w, label)
                    elif UploadModel.CLASSES[idx]:
                        continue
                    else:
                        self.detect_and_draw(frame, detections, h, i, w, "Unknown object")
                else:
                    self.detect_and_draw(frame, detections, h, i, w, "Unknown object")
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF

                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                    break

                # update the FPS counter
                fps.update()

    @staticmethod
    def detect_and_draw(frame, detections, h, i, w, label):
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        cv2.rectangle(frame, (startX, startY), (endX, endY), UploadModel.COLORS[1], 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), UploadModel.COLORS[1], 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, UploadModel.COLORS[1], 2)

        coord_x_centroid = (endX + startX) // 2
        coord_y_centroid = (endY + startY) // 2
        object_centroid = (coord_x_centroid, coord_y_centroid)
        cv2.circle(frame, object_centroid, 1, (0, 250, 250), 5)