import pygame
import cv2
import sys
import os


def redraw_game():
    pygame.display.update()


def game_loop():
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        redraw_game()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    pygame.init()

    resources_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "resources"
        )
    )

    win_size = (640, 360)
    win = pygame.display.set_mode(win_size, pygame.RESIZABLE)
    pygame.display.set_caption("Ground Aircraft Marshalling Simulator")

    icon = pygame.image.load(f"{resources_path}/icon.png")
    pygame.display.set_icon(icon)

    game_loop()