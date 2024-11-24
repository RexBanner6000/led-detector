from typing import Union
from rpi_ws281x import PixelStrip

import board
import neopixel
import time

# LED strip configuration:
LED_PIN = 18          # GPIO pin connected to the pixels (18 uses PWM!).
LED_FREQ_HZ = 800000  # LED signal frequency in hertz (usually 800khz)
LED_DMA = 10          # DMA channel to use for generating signal (try 10)
LED_BRIGHTNESS = 255  # Set to 0 for darkest and 255 for brightest
LED_INVERT = False    # True to invert the signal (when using NPN transistor level shift)
LED_CHANNEL = 0       # set to '1' for GPIOs 13, 19, 41, 45 or 53


class LEDMatrix:

    def __init__(self, n_leds: int, gpio=board.D18):
        self.gpio = gpio
        self.n_leds = n_leds
        self.pixels = neopixel.NeoPixel(self.gpio, self.n_leds)
        self.strip = PixelStrip(
            self.n_leds, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL
        )
        self.strip.begin()

    def flash_green(self, detected: bool):
        if detected:
            self.pixels.fill((0, 220, 40))
        else:
            self.pixels.fill((225, 0, 0))

    def turn_off(self):
        self.pixels.fill((0, 0, 0))

    def colorWipe(self, wait_ms=50):
        """Wipe color across display a pixel at a time."""
        for i in range(self.strip.numPixels()):
            self.strip.setPixelColorRGB(i, 0, 0, 0)
            self.strip.show()
            time.sleep(wait_ms / 1000.0)

    def display_detection(self, x: Union[float, None]):
        if x is not None:
            for i in range(self.strip.numPixels()):
                if i == int(x * self.strip.numPixels()):
                    self.strip.setPixelColorRGB(i, 0, 255, 0)
                else:
                    self.strip.setPixelColorRGB(i, 255, 0, 0)
        else:
            self.colorWipe()
        self.strip.show()
