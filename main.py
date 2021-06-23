import numpy as np
import cv2
import pyvirtualcam


cascade_frontalface = cv2.CascadeClassifier(
    './haarcascades/haarcascade_frontalface_alt.xml')
cascade_profileface = cv2.CascadeClassifier(
    './haarcascades/haarcascade_profileface.xml')

laughingman_frame = cv2.imread(
    './input/laughingman-frame.png', cv2.IMREAD_UNCHANGED)
laughingman_inner = cv2.imread(
    './input/laughingman-inner.png', cv2.IMREAD_UNCHANGED)
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
        print('setting start...')
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.capture.get(cv2.CAP_PROP_FPS))
        print('setting complete')

    def run(self):
        print('application start...')
        self.start_capture_process()
        self.capture_loop()
        self.capture_release_process()
        print('application finish')

    def start_capture_process(self):
        print('capture start...')
        self.capture = cv2.VideoCapture(0)
        self.capture_setting()

    def capture_release_process(self):
        print('capture release...')
        self.capture.release()
        cv2.destroyAllWindows()
        print('release complete')

    def capture_loop(self):
        print('broadcast start...')
        capture_padding = 2 * self.capture_padding_pixel
        with pyvirtualcam.Camera(self.width - 2*self.capture_padding_pixel,
                                 self.height - 2*self.capture_padding_pixel,
                                 self.fps
                                ) as virtual_camera:
            while self.capture.isOpened():
                return_value, frame = self.capture.read()

                if return_value == False:
                    break
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                cv2.imshow('LaughingmanApp', running_mark)

                face_list = self.obtaine_face_list(frame)

                if len(face_list) == 0:
                    self.send_frame(virtual_camera,
                                    frame[self.capture_padding_pixel:self.height-self.capture_padding_pixel,
                                    self.capture_padding_pixel:self.width-self.capture_padding_pixel])
                    continue

                for (x, y, w, h) in face_list:
                    x -= self.laughingman_expansion_pixel
                    y -= self.laughingman_expansion_pixel
                    w += 2 * self.laughingman_expansion_pixel
                    h += 2 * self.laughingman_expansion_pixel

                    if x < 0:
                        x = 0
                    if y < 0:
                        y = 0

                    laughingman_rotated = self.rotation(laughingman_frame)
                    laughingman_frame_resized = cv2.resize(laughingman_rotated, (w, h))
                    laughingman_inner_resized = cv2.resize(laughingman_inner, (w, h))

                    if x+w > self.width:
                        w = self.width - x
                    if y+h > self.height:
                        h = self.height- y

                    frame[y:y+h, x:x+w] = self.overlay_image(frame[y:y+h, x:x+w], laughingman_frame_resized[:h, :w])
                    frame[y:y+h, x:x+w] = self.overlay_image(frame[y:y+h, x:x+w], laughingman_inner_resized[:h, :w])

                self.send_frame(virtual_camera,
                                frame[self.capture_padding_pixel:self.height-self.capture_padding_pixel,
                                      self.capture_padding_pixel:self.width-self.capture_padding_pixel])

        print('camera close...')
        virtual_camera.close()
        print('close complete')

    def obtaine_face_list(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_list = cascade_frontalface.detectMultiScale(frame_gray, minSize=(100,100))
        if len(face_list) != 0:
            return face_list
        face_list = cascade_profileface.detectMultiScale(frame_gray, minSize=(100, 100))
        if len(face_list) != 0:
            return face_list
        face_list = cascade_profileface.detectMultiScale(cv2.flip(frame_gray, 1), minSize=(100, 100))
        if len(face_list) != 0:
            face_list[:, 0] = self.width - face_list[:, 0] - face_list[:, 2]
            return face_list
        return list()
    
    def rotation(self, image):
        rotation_matrix = cv2.getRotationMatrix2D(laughingman_center, self.rotation_angle_degree, 1)
        self.rotation_angle_degree += 1
        if self.rotation_angle_degree >= 360:
            self.rotation_angle_degree = 0
        return cv2.warpAffine(image, rotation_matrix, (image.shape[0], image.shape[1]))

    def overlay_image(self, frame, image):
        image[:, :, 3:] = image[:, :, 3:]/255
        return frame[:, :] * (1 - image[:, :, 3:]) + image[:, :, :3] * (image[:, :, 3:])
    
    def send_frame(self, virtual_camera, frame):
        frame = cv2.flip(frame, 1)
        virtual_camera.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        virtual_camera.sleep_until_next_frame()
        

if __name__ == '__main__':
    app = LaughingmanApplication()
    app.run()