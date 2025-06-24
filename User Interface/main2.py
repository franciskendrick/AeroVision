import pygame
import cv2
import sys
import os
import numpy as np


class Game:
    def __init__(self, win_size):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Could not open camera.")
            pygame.quit()
            sys.exit()

        self.frame_surface = None
        
        self.init_scale(win_size)

    def init_scale(self, win_size):
        base_width, base_height = 640, 360
        scale_w = win_size[0] / base_width
        scale_h = win_size[1] / base_height
        self.scale = min(scale_w, scale_h)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        self.frame_surface = pygame.surfarray.make_surface(frame)

    def draw(self, win):
        win_size = win.get_size()
        win.fill((0, 0, 0))  # Optional: clear screen

        half_width = win_size[0] // 2
        full_height = win_size[1]

        # Vertical divider
        pygame.draw.line(win, (100, 100, 100), (half_width, 0), (half_width, full_height), 2)

        if not self.frame_surface:
            return

        cam_rect = self.frame_surface.get_rect()
        max_width = win_size[0] // 2
        max_height = win_size[1]

        scale = min(max_width / cam_rect.width, max_height / cam_rect.height)
        new_size = (int(cam_rect.width * scale), int(cam_rect.height * scale))
        frame = pygame.transform.smoothscale(self.frame_surface, new_size)

        pos_x = win_size[0] - new_size[0] // 2 - max_width // 2
        pos_y = (win_size[1] - new_size[1]) // 2
        win.blit(frame, (pos_x, pos_y))

        pygame.display.update()

    def release(self):
        self.cap.release()


def game_loop():
    run = True

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            elif event.type == pygame.VIDEORESIZE:
                new_width = max(640, event.w)
                new_height = max(360, event.h)
                new_size = (new_width, new_height)

                pygame.display.set_mode(new_size, pygame.RESIZABLE)
                game.init_scale(new_size)

        game.update_frame()
        game.draw(win)

    game.release()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    pygame.init()

    resources_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "resources")
    )

    win_size = (640, 360)
    win = pygame.display.set_mode(win_size, pygame.RESIZABLE)
    pygame.display.set_caption("Ground Aircraft Marshalling Simulator")

    icon = pygame.image.load(f"{resources_path}/icon.png")
    pygame.display.set_icon(icon)

    game = Game(win_size)
    game_loop()
