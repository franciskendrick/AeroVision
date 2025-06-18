import pygame
import sys
import os


def redraw():
    win.fill((0, 0, 0))

    win.blit(background, (0, 0))

    pygame.display.update()


def loop():
    global background

    background = pygame.image.load(f"{resources_path}/background.png")
    background = pygame.transform.scale(background, win_size)

    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            elif event.type == pygame.VIDEORESIZE:
                background = pygame.image.load(f"{resources_path}/background.png")
                background = pygame.transform.scale(background, event.dict["size"])

        redraw()
    
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

    loop()