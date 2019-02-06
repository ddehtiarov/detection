from flask import Flask, Response

app = Flask(__name__)


@app.route('/')
def index():
    from camera2 import VideoCamera
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


if __name__ == '__main__':
    # app.run()
    from camera3 import VideoCamera
    VideoCamera().get_frame()
