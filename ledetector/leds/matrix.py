import board
import neopixel
from typing import Union


class LEDMatrix:
    def __init__(self, n_leds: int, gpio=board.D18):
        self.gpio = gpio
        self.n_leds = n_leds
        self.pixels = neopixel.NeoPixel(self.gpio, self.n_leds)

    def flash_green(self, detected: bool):
        if detected:
            self.pixels.fill((0, 220, 40))
        else:
            self.pixels.fill((225, 0, 0))

    def turn_off(self):
        self.pixels.fill((0, 0, 0))


    def display_detection(self, x: Union[float, None]):
        if x is None:
            self.pixels.fill((0, 0, 0))
        else:
            x_pixel = int(x * self.n_leds)
            for idx in range(0, self.n_leds):
                if x_pixel == idx:
                    self.pixels[idx] = (0, 255, 0)
                else:
                    self.pixels[idx] = (0, 0, 0)