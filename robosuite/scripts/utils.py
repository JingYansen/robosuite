import numpy as np


class Pid:
    def __init__(self, p=0.2, i=0.01, d=0.1):
        self.Kp = p
        self.Ki = i
        self.Kd = d
        self.clear()

    def clear(self):
        self.SetLevel = 0.0
        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0
        self.windup_guard = 100.0
        self.output = 0.0

    def update(self, feedback_value):
        error = self.SetLevel - feedback_value
        #比例环
        self.PTerm = self.Kp * error
        #积分环
        self.ITerm += error
        if (self.ITerm < -self.windup_guard):
            self.ITerm = -self.windup_guard
        elif (self.ITerm > self.windup_guard):
            self.ITerm = self.windup_guard
        #微分环
        delta_error = error-self.last_error
        self.DTerm = delta_error
        self.last_error = error
        #输出
        self.output = self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)

        return self.output


def norm_depth(depth, depth_max=0.99, depth_min=0.85):
    image = (depth - depth_min) / (depth_max - depth_min)
    image = np.clip(image, 0, 1)
    image = np.uint8(image * 255)

    image_shape = image.shape
    image = image.reshape(image_shape[0], image_shape[1], 1)

    return image