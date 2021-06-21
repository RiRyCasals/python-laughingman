import numpy as np
import cv2
import pyvirtualcam
# from functools import lru_cashe


cascade_frontalface = cv2.CascadeClassifier(
    './haarcascades/haarcascade_frontalface_alt.xml')
cascade_profileface = cv2.CascadeClassifier(
    './haarcascades/haarcascade_profileface.xml')

laughingman_inner = cv2.imread(
    './input/laughingman-inner.png', cv2.IMREAD_UNCHANGED)
laughingman_frame = cv2.imread(
    './input/laughingman-frame.png', cv2.IMREAD_UNCHANGED)
running_mark = cv2.imread('./input/running-mark.jpg')

laughingman_center = (int(laughingman_frame.shape[0] / 2),
                      int(laughingman_frame.shape[1] / 2))

class LaughingmanApplication():
    def __init__(self):
        self.capture_padding_pixel = 150
        self.capture_resize_scale = 1
        self.laughingman_expansion_pixel = 25
        self.rotation_angle_degree = 0
        self.capture = None
        self.width = 0
        self.height = 0
        self.fps = 0

    def capture_setting(self):
        print('setting start') #dev
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.htight = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        print('setting was complete') #dev

    def start_capture_process(self):
        print('start capture process') #dev
        self.capture = cv2.VideoCapture(0)
        self.capture_setting()
        print('capture process was complete') #dev

    def capture_loop(self):
        print('loop start') #dev
        capture_padding = 2 * self.capture_padding_pixel
        with pyvirtualcam.Camera(int(self.width * capture_padding),
                                 int(self.height * capture_padding),
                                 int(self.fps)
                                ) as virtual_camera:
            while self.capture.isOpened():
                return_value, frame = self.capture.read()

                if return_value == False:
                    break
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print('confirmed input "q"\nloop finish') #dev
                    break

                cv2.imshow('LaughingmanApp', running_mark)

                frame = cv2.flip(frame, 1)
                face_list = obtaine_face_list(frame)

                if face_list == None:
                    # send(cv2.cvtColor(frame, ))
                    continue

                for (x, y, w, h) in face_list:
                    x -= self.laughingman_expansion_pixel
                    y -= self.laughingman_expansion_pixel
                    w += 2 * self.laughingman_expansion_pixel
                    h += 2 * self.laughingman_expansion_pixel
                    laughingman_inner_resized = cv2.resize(laughingman_inner, (w, h))
                    laughingman_frame_resized = cv2.resize(self.rotation(laughingman_frame), (w, h))

    def capture_release_process(self):
        print('start release process') #dev
        self.capture.release()
        cv2.destroyAllWindows()
        print('release process was complete') #dev

    def run(self):
        self.start_capture_process()
        self.capture_loop()
        self.capture_release_process()

    def obtaine_face_list(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_list = cascade_frontalface.detectMultiScale(frame_gray, minSize=(100,100))
        if len(face_list) != 0:
            return face_list
        face_list = cascade_profileface.detectMultiScale(frame_gray, minSize=(100, 100))        
        if len(face_list) != 0:
            return face_list
        return None
    
    def rotation(self, image):
        rotation_matrix = cv2.getRotationMatrix2D(laughingman_center, self.degree, 1)
        return cv2.warpAffine(image, rotation_matrix, (image.shape[0], image.shape[1]))

    def overlay_images(self):
        pass


if __name__ == '__main__':
    app = LaughingmanApplication()
    app.run()