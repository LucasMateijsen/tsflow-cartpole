import tensorflow as tf
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
        self.thetaThreshold = 12 / 360 * 2 * math.pi

        self.setRandomState()

    def setRandomState(self):
        self.x = rand.random() - 0.5
        self.xDot = (rand.random() - 0.5) * 1
        self.theta = (rand.random() - 0.5) * 2 * (6 / 360 * 2 * math.pi)
        self.thetaDot = (rand.random() - 0.5) * 0.5

    def getStateTensor(self):
        return np.array([[self.x, self.xDot, self.theta, self.thetaDot]])

    def update(self, action):
        force = -self.forceMag
        if action > 0:
            force = self.forceMag

        cosTheta = math.cos(self.theta)
        sinTheta = math.sin(self.theta)

        temp = (force + self.poleMoment * self.thetaDot * self.thetaDot * sinTheta) / self.totalMass
        thetaAcc = (self.gravity * sinTheta - cosTheta * temp) / (self.length * (4 / 3 - self.massPole * cosTheta * cosTheta / self.totalMass))
        xAcc = temp - self.poleMoment * thetaAcc * cosTheta / self.totalMass
        
        self.x += self.tau * self.xDot
        self.xDot += self.tau * xAcc
        self.theta += self.tau * self.thetaDot
        self.thetaDot += self.tau * thetaAcc

        return self.isDone()

    def isDone(self):
        return self.x < -self.xThreshold or self.x > self.xThreshold or self.theta < -self.thetaThreshold or self.theta > self.thetaThreshold