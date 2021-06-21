import numpy as np
import cv2
import pyvirtualcam


cascade_frontalface = cv2.CascadeClassifier(
    './haarcascades/haarcascade_frontalface_alt.xml')
cascade_profileface = cv2.CascadeClassifier(
    './haarcascades/haarcascade_profileface.xml')

laughingman_inner = cv2.imread(
    './input/laughingman-inner.png', cv2.IMREAD_UNCHANGED)
laughingman_frame = cv2.imread(
    './input/laughingman-frame.png', cv2.IMREAD_UNCHANGED)
running_mark = cv2.imread('./input/running-mark.jpg')


class LaughingmanApplication():
    def __init__(self):
        self.capture_padding_pixel = 150
        self.capture_resize_scale = 1
        self.laughingman_expansion_pixel = 25
        self.rotation_angle = 0
        self.capture = None
        self.width = 0
        self.height = 0
        self.fps = 0

    def capture_setting(self):
        print('setting start')
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.htight = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        print('setting was complete')

    def start_capture_process(self):
        print('start capture process')
        self.capture = cv2.VideoCapture(0)
        self.capture_setting()
        print('capture process was complete')

    def capture_loop(self):
        print('loop start')
        while self.capture.isOpened():
            ret, frame = self.capture.read()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('confirmed input "q"\nloop finish')
                break
            cv2.imshow('LaughingmanApp', running_mark)

    def capture_release_process(self):
        print('start release process')
        self.capture.release()
        cv2.destroyAllWindows()
        print('release process was complete')

    def run(self):
        self.start_capture_process()
        self.capture_loop()
        self.capture_release_process()


if __name__ == '__main__':
    app = LaughingmanApplication()
    app.run()