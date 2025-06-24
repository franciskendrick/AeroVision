import pygame
import cv2
import sys
import os
import numpy as np


def redraw_game(frame_surface, win_size, win):
    win.fill((0, 0, 0))  # Optional: clear screen with black

    # Split screen in half
    half_width = win_size[0] // 2
    full_height = win_size[1]

    # Draw vertical divider line (optional)
    pygame.draw.line(win, (100, 100, 100), (half_width, 0), (half_width, full_height), 2)

    if frame_surface:
        # Resize frame to fit inside right half (maintain aspect ratio)
        cam_rect = frame_surface.get_rect()
        max_width = win_size[0] // 2
        max_height = win_size[1]

        scale = min(max_width / cam_rect.width, max_height / cam_rect.height)
        new_size = (int(cam_rect.width * scale), int(cam_rect.height * scale))
        frame_surface = pygame.transform.smoothscale(frame_surface, new_size)

        # Center it in the right half
        pos_x = win_size[0] - new_size[0] // 2 - max_width // 2
        pos_y = (win_size[1] - new_size[1]) // 2
        win.blit(frame_surface, (pos_x, pos_y))

    pygame.display.update()


def game_loop():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera.")
        pygame.quit()
        sys.exit()

    run = True
    frame_surface = None

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            elif event.type == pygame.VIDEORESIZE:
                win_size = (max(640, event.w), max(360, event.h))
                pygame.display.set_mode(win_size, pygame.RESIZABLE)

        # Capture frame
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)  # Optional: mirror the feed
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.rot90(frame)  # Rotate if needed
            frame_surface = pygame.surfarray.make_surface(frame)

        win_size = pygame.display.get_surface().get_size()
        redraw_game(frame_surface, win_size, pygame.display.get_surface())

    cap.release()
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
