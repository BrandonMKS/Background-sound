import pygame

pygame.init()
# creating window

pygame.display.set_caption("Sound", icontitle="Yes")
size = pygame.display.get_desktop_sizes()
screen = pygame.display.set_mode([100, 50])

from pygame.locals import (
    K_w,
    K_s,
    K_a,
    K_d,
    K_ESCAPE,
    KEYDOWN,
    QUIT, )
pygame.mixer.music.load("song1.mp3")
pygame.mixer.music.set_volume(0.05)


#pygame.mixer.music.play()




running = True
while running:
    pygame.mixer.music.play()
    # screen/display
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    pygame.display.flip()





pygame.quit()
