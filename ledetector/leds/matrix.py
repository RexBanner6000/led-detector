from typing import Union
from rpi_ws281x import PixelStrip

import board
import neopixel

# LED strip configuration:
LED_COUNT = 16        # Number of LED pixels.
LED_PIN = 18          # GPIO pin connected to the pixels (18 uses PWM!).
# LED_PIN = 10        # GPIO pin connected to the pixels (10 uses SPI /dev/spidev0.0).
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
            LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL
        )
        self.strip.begin()

    def flash_green(self, detected: bool):
        if detected:
            self.pixels.fill((0, 220, 40))
        else:
            self.pixels.fill((225, 0, 0))

    def turn_off(self):
        self.pixels.fill((0, 0, 0))

    def display_detection(self, x: Union[float, None]):

        for i in range(0, self.strip.numPixels()):
            if x is not None:
                if i == int(x * self.strip.numPixels()):
                    self.strip.setPixelColorRGB(i, 128, 128, 128)
            else:
                self.strip.setPixelColorRGB(i, 0, 0, 0)
        self.strip.show()
