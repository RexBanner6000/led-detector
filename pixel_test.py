import board
import neopixel
from time import sleep

pixels = neopixel.NeoPixel(board.D18, 4)

pixels.fill((0, 170, 60))
sleep(10)
pixels.fill((255, 0, 0))
sleep(5)
pixels.fill((0,0,0))
