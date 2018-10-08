import math as math
import random as rand
import numpy as np


class CartPole:

    def __init__(self):
        self.gravity = 9.8
        self.massCart = 1.0
        self.massPole = 0.1
        self.totalMass = self.massCart + self.massPole
        self.cartWidth = 0.2
        self.cartHeight = 0.1
        self.length = 0.5
        self.poleMoment = self.massPole * self.length
        self.forceMag = 10.0
        self.tau = 0.02

        self.xThreshold = 2.4
        self.theta_threshold = 12 / 360 * 2 * math.pi

        self.x, self.x_dot, self.theta, self.theta_dot = 0, 0, 0, 0
        self.set_random_state()

    def set_random_state(self):
        self.x = rand.random() - 0.5
        self.x_dot = (rand.random() - 0.5) * 1
        self.theta = (rand.random() - 0.5) * 2 * (6 / 360 * 2 * math.pi)
        self.theta_dot = (rand.random() - 0.5) * 0.5

    def get_state_tensor(self):
        return np.array([[self.x, self.x_dot, self.theta, self.theta_dot]])

    def update(self, action):
        force = -self.forceMag
        if action > 0:
            force = self.forceMag

        cos_theta = math.cos(self.theta)
        sin_theta = math.sin(self.theta)

        temp = (force + self.poleMoment * self.theta_dot * self.theta_dot * sin_theta) / self.totalMass
        theta_acc = (self.gravity * sin_theta - cos_theta * temp) / (
                    self.length * (4 / 3 - self.massPole * cos_theta * cos_theta / self.totalMass))
        x_acc = temp - self.poleMoment * theta_acc * cos_theta / self.totalMass

        self.x += self.tau * self.x_dot
        self.x_dot += self.tau * x_acc
        self.theta += self.tau * self.theta_dot
        self.theta_dot += self.tau * theta_acc

        return self.is_done()

    def is_done(self):
        return self.x < -self.xThreshold or self.x > self.xThreshold or \
               self.theta < -self.theta_threshold or self.theta > self.theta_threshold
