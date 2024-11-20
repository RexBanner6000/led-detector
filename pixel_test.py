import board
import neopixel
from time import sleep

pixels = neopixel.NeoPixel(board.D18, 16, pixel_order="RGB")

pixels.fill((255, 0, 0))
sleep(5)
pixels.fill((0, 255, 0))
sleep(5)
pixels.fill((0, 0, 255))
sleep(5)
pixels.fill((0,0,0))
