import cv2
import numpy as np
import UploadModel
import os


class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        # self.video = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        self.video = cv2.VideoCapture('highway.mp4')
        # self.video = cv2.VideoCapture('flow.mp4')
        self.video.set(3, 640)
        self.video.set(4, 480)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, Frame = self.video.read()

        h = np.size(Frame, 0)
        w = np.size(Frame, 1)

        # blob = cv2.dnn.blobFromImage(cv2.resize(Frame, (300, 300)), 0.007843, (300, 300), 127.5)
        # blob = cv2.dnn.blobFromImage(cv2.resize(Frame, (1248, 352)), 0.007843, (1248, 352), 127.5)
        # blob = cv2.dnn.blobFromImage(cv2.resize(Frame, (700, 700)), 0.007843, (700, 700), 127.5)
        blob = cv2.dnn.blobFromImage(cv2.resize(Frame, (800, 800)), 0.007843, (800, 800), 127.5)

        UploadModel.net.setInput(blob)
        detections = UploadModel.net.forward()
        # print("startX#=" + str((detections)))
        # print detections.shape[2]
        idxs = np.argsort(detections[0])[::-1][:5]
        # print("# " + str(idxs))
        # print("# " + str(detections[0]))

        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]
            # print("conf#=" + str(confidence))

            if confidence > UploadModel.args["confidence"]:
                # if True:
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                idx = int(detections[0, 0, i, 1])
                # print("idx#="+ str(idx))
                # if True:
                if UploadModel.CLASSES[idx] == "aeroplane" \
                        or UploadModel.CLASSES[idx] == "bird":
                    label = "{}: {:.2f}%".format(UploadModel.CLASSES[idx], confidence * 100)
                    self.detect_and_draw(Frame, detections, h, i, idx, w, label)
                elif UploadModel.CLASSES[idx]:
                    continue
                else:
                    self.detect_and_draw(Frame, detections, h, i, idx, w, "Unknown object")

        ret, jpeg = cv2.imencode('.jpg', Frame)
        return jpeg.tobytes()

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
