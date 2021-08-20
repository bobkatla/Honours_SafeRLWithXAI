import random


# Think of now
# Create 2 matrix, one for the Hazard object, one for normal value
# the hazard is for training while the normal val would be for drawing
class Hazard:
    def __init__(self, label, temp, humid, wall):
        self.label = label
        self.temp = temp
        self.humid = humid
        self.wall = wall

    def __str__(self):
        return str((self.label, self.temp, self.humid, self.wall))

    # def __init__(self, t: int):
    #     self.temp = None
    #     self.humid = None
    #     self.wall = None
    #     if type(t) is not int or t < 0 or t > 3:
    #         Exception('Error for input')
    #     if t == 3:
    #         self.temp = 0  # This is a wall
    #     a = random.randint(11, 20)
    #     b = random.randint(1, 9)
    #     self.temp = t * a + b * 0.1 + 9
    #     # So water always under 10, and heat always more than 31
    #     # 0 is water, 1 is normal, 2 is heat

    def get_temp(self):
        return self.temp

    def set_temp(self, t):
        self.temp = t

    def set_humid(self, h):
        self.humid = h

    def set_wall(self, w):
        self.wall = w
