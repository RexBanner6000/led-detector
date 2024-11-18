import board
import neopixel


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
