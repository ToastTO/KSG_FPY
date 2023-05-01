import pygame
pygame.mixer.init()
pygame.mixer.music.load('Z7E8E5U-beep-beep.mp3')
pygame.mixer.music.play()
while pygame.mixer.music.get_busy() == True:
    continue
