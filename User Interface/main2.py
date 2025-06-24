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
        self.init_opencv(win_size)
        self.init_rightpanel(win_size)

    def init_scale(self, win_size):
        base_width, base_height = 640, 360
        scale_w = win_size[0] / base_width
        scale_h = win_size[1] / base_height
        self.scale = min(scale_w, scale_h)

    def init_opencv(self, win_size):
        # Use a dummy frame to calculate aspect ratio
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to read dummy frame for init.")
            pygame.quit()
            sys.exit()

        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        dummy_surface = pygame.surfarray.make_surface(frame)

        cam_rect = dummy_surface.get_rect()
        max_width = win_size[0] // 2
        max_height = win_size[1]

        scale = min(max_width / cam_rect.width, max_height / cam_rect.height)
        new_size = (int(cam_rect.width * scale), int(cam_rect.height * scale))
        self.frame_draw_size = new_size

        pos_x = win_size[0] - new_size[0] // 2 - max_width // 2
        pos_y = (win_size[1] - new_size[1]) // 2
        self.frame_draw_pos = (pos_x, pos_y)

    def init_rightpanel(self, win_size):
        cam_top_y = self.frame_draw_pos[1]
        cam_bottom_y = cam_top_y + self.frame_draw_size[1]
        right_x = win_size[0] // 2
        panel_width = win_size[0] // 2
        full_height = win_size[1]

        self.rp_top_rect = pygame.Rect(right_x, 0, panel_width, cam_top_y)
        self.rp_bottom_rect = pygame.Rect(right_x, cam_bottom_y, panel_width, full_height - cam_bottom_y)

        # ── SIGNAL PREDICTION TEXT ──
        font_size = int(self.rp_top_rect.height * 0.4)  # Scales with panel height
        font = pygame.font.SysFont("Franklin Gothic Medium Condensed", font_size, bold=False)

        self.prediction_text = font.render("SIGNAL PREDICTION: NONE", True, (0, 0, 0))

        text_rect = self.prediction_text.get_rect()
        text_x = self.rp_top_rect.centerx - text_rect.width // 2
        text_y = self.rp_top_rect.centery - text_rect.height // 2
        self.prediction_text_pos = (text_x, text_y)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        self.frame_surface = pygame.surfarray.make_surface(frame)

    def draw(self):
        win.fill((0, 0, 0)) 

        if self.frame_surface:
            scaled_frame = pygame.transform.smoothscale(self.frame_surface, self.frame_draw_size)
            win.blit(scaled_frame, self.frame_draw_pos)

            pygame.draw.rect(win, (192, 192, 192), self.rp_top_rect)
            pygame.draw.rect(win, (192, 192, 192), self.rp_bottom_rect)

            win.blit(self.prediction_text, self.prediction_text_pos)

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
                game.init_opencv(new_size)
                game.init_rightpanel(new_size)

        game.update_frame()
        game.draw()

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
