import mediapipe as mp
import numpy as np
import pygame
import cv2
import sys
import os


class Game:
    def __init__(self, win_size):
        self.cap = cv2.VideoCapture(0)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils
        if not self.cap.isOpened():
            print("Could not open camera.")
            pygame.quit()
            sys.exit()

        self.frame_surface = None

        self.init_scale(win_size)
        self.init_opencv(win_size)
        self.init_panels(win_size)

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

    def init_panels(self, win_size):
        cam_top_y = self.frame_draw_pos[1]
        cam_bottom_y = cam_top_y + self.frame_draw_size[1]
        center_x = win_size[0] // 2
        panel_width = win_size[0] // 2
        full_height = win_size[1]

        # RIGHT PANEL (camera side)
        self.rp_top_rect = pygame.Rect(center_x, 0, panel_width, cam_top_y)
        self.rp_bottom_rect = pygame.Rect(center_x, cam_bottom_y, panel_width, full_height - cam_bottom_y)

        # LEFT PANEL (placeholder / visual instructions)
        self.lp_top_rect = pygame.Rect(0, 0, center_x, cam_top_y)

        # ── SIGNAL PREDICTION (right panel) ──
        small_font_size = int(self.rp_top_rect.height * 0.3)
        big_font_size = int(self.rp_top_rect.height * 0.525)

        font_small = pygame.font.SysFont("Franklin Gothic Medium Condensed", small_font_size)
        font_big = pygame.font.SysFont("Franklin Gothic Medium Condensed", big_font_size)

        label_surface = font_small.render("SIGNAL PREDICTION:", True, (0, 0, 0))
        value_surface = font_big.render("NONE", True, (0, 0, 0))

        label_rect = label_surface.get_rect()
        value_rect = value_surface.get_rect()

        total_text_height = label_rect.height + value_rect.height + int(4 * self.scale)

        text_x = self.rp_top_rect.centerx
        text_y_start = self.rp_top_rect.centery - total_text_height // 2

        self.prediction_text_surfaces = [
            (label_surface, (text_x - label_rect.width // 2, text_y_start)),
            (value_surface, (text_x - value_rect.width // 2, text_y_start + label_rect.height + int(2 * self.scale)))
        ]

        # BUTTONS
        franklinsmall_size = int(16 * self.scale)
        franklingothic_small = pygame.font.SysFont("Franklin Gothic Medium Condensed", franklinsmall_size, bold=False)

        button_labels = ["END TRAINING", "START"]
        button_surfaces = [franklingothic_small.render(label, True, (0, 0, 0)) for label in button_labels]

        base_padding_w, base_padding_h = 25, 20
        padding_w = int(base_padding_w * self.scale)
        padding_h = int(base_padding_h * self.scale)

        # Step 1: Find max text width
        text_rects = [surf.get_rect() for surf in button_surfaces]
        max_text_width = max(rect.width for rect in text_rects)

        # Step 2: Shared button dimensions
        button_width = max_text_width + padding_w
        button_height = max(rect.height for rect in text_rects) + padding_h

        # Vertical alignment (example)
        y = int(317 * self.scale)

        self.buttons = {}
        for idx, (label, surface, text_rect) in enumerate(zip(button_labels, button_surfaces, text_rects)):
            if idx == 0:  # Bottom-left of RIGHT panel
                x = self.rp_bottom_rect.left + int(20 * self.scale)
            else:  # Bottom-right of RIGHT panel
                x = self.rp_bottom_rect.right - button_width - int(20 * self.scale)

            button_rect = pygame.Rect(x, y, button_width, button_height)

            text_x = x + (button_width - text_rect.width) // 2
            text_y = y + (button_height - text_rect.height) // 2

            self.buttons[label] = [False, True, surface, (text_x, text_y), button_rect]

        # ── VISUAL INSTRUCTIONS (left panel) ──
     
        vis_font_size = int(self.lp_top_rect.height * 0.55)
        vis_font = pygame.font.SysFont("Franklin Gothic Medium Condensed", vis_font_size, bold=False)

        self.visinstr_text = vis_font.render("VISUAL INSTRUCTIONS", True, (0, 0, 0))
        vis_rect = self.visinstr_text.get_rect()
        self.visinstr_pos = (
            self.lp_top_rect.centerx - vis_rect.width // 2,
            self.lp_top_rect.centery - vis_rect.height // 2
        )

        # Guide
        self.guide_position = [0, self.lp_top_rect.bottom]
        self.guide_videos = {}
        actions = [
            "straight_ahead", "turn_left", "turn_right", "stop", "cut_engine",
            "start_engine", "set_brakes", "chocks_insterted", "all_clear"]

        for action in actions:
            action_dir = os.path.join(f"{resources_path}/guide/{action}")
            
            # frames = [pygame.image.load(f"{action_dir}/{file}").convert() for file in os.listdir(action_dir)]
            frame_paths = sorted(
                [os.path.join(action_dir, f) for f in os.listdir(action_dir) if f.endswith(".jpg")])
            frames = []
            for path in frame_paths:
                img = pygame.image.load(path).convert()
                original_size = img.get_size()
                original_size = (300, 300)
                scaled_size = (int(original_size[0] * self.scale), int(original_size[1] * self.scale))
                img_scaled = pygame.transform.smoothscale(img, scaled_size)
                frames.append(img_scaled)

            self.guide_videos[action] = (frames, tuple(self.guide_position))

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipe Pose processing
        results = self.pose.process(frame_rgb)
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
            )

        # Convert for Pygame (after drawing)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = np.rot90(frame_rgb)
        self.frame_surface = pygame.surfarray.make_surface(frame_rgb)

    def draw(self):
        win.fill((255, 255, 255)) 

        border_width = max(1, round(2 * self.scale))

        if self.frame_surface:     
            scaled_frame = pygame.transform.smoothscale(self.frame_surface, self.frame_draw_size)
            win.blit(scaled_frame, self.frame_draw_pos)

            pygame.draw.rect(win, (192, 192, 192), self.rp_top_rect)
            pygame.draw.rect(win, (119, 163, 200), self.lp_top_rect)

            # Draw guide
            self.draw_guide_animation(win, "")

            pygame.draw.line(win, (0, 0, 0), (self.lp_top_rect.centerx, 0), (self.lp_top_rect.centerx, 360 * self.scale))

            # Draw text
            for surface, pos in self.prediction_text_surfaces:
                win.blit(surface, pos)
            win.blit(self.visinstr_text, self.visinstr_pos)

            # Draw buttons
            for is_hovered, is_open, text, text_pos, btn_rect in self.buttons.values():
                fill = (192, 192, 192) if is_hovered and is_open else \
                    (240, 240, 240) if is_open else (132, 132, 132)
                pygame.draw.rect(win, fill, btn_rect)
                pygame.draw.rect(win, (0, 0, 0), btn_rect, border_width)
                win.blit(text, text_pos)

        pygame.display.update()

    def draw_guide_animation(self, screen, action, frame_delay=5):
        action = "straight_ahead"
        if not hasattr(self, 'guide_frame_counters'):
            self.guide_frame_counters = {}
            self.guide_frame_timers = {}

        frames, pos = self.guide_videos[action]

        # Init counters
        if action not in self.guide_frame_counters:
            self.guide_frame_counters[action] = 0
            self.guide_frame_timers[action] = 0

        idx = self.guide_frame_counters[action]

        # Draw current frame
        screen.blit(frames[idx], pos)

        # Update timer and frame index
        self.guide_frame_timers[action] += 1
        if self.guide_frame_timers[action] >= frame_delay:
            self.guide_frame_counters[action] = (idx + 1) % len(frames)
            self.guide_frame_timers[action] = 0

    def button_over_detection(self, mouse_pos):
        for button in self.buttons.values():
            button[0] = button[4].collidepoint(mouse_pos)

    def button_down_detection(self, mouse_pos):
        for label, (_, is_open, *_, btn_rect) in self.buttons.items():
            if is_open and btn_rect.collidepoint(mouse_pos):
                return label

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
                game.init_panels(new_size)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                btn_label = game.button_down_detection(mouse_pos)
                print(btn_label)

        mouse_pos = pygame.mouse.get_pos()
        game.button_over_detection(mouse_pos)
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
