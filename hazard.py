import random


class Hazard:
    def __init__(self, t: int):
        if type(t) is not int or t < 0 or t > 3:
            Exception('Error for input')
        if t == 3:
            self.temp = 0  # This is a wall
        a = random.randint(11, 20)
        b = random.randint(1, 9)
        self.temp = t * a + b * 0.1 + 9
        # So water always under 10, and heat always more than 31
        # 0 is water, 1 is normal, 2 is heat
