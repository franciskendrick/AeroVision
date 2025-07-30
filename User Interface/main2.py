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
        self.training_started = False
        self.instruction = "None"
        self.current_action = 0

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
        def setup_layout():
            cam_top_y = self.frame_draw_pos[1]
            cam_bottom_y = cam_top_y + self.frame_draw_size[1]
            center_x = win_size[0] // 2
            panel_width = win_size[0] // 2
            full_height = win_size[1]

            self.rp_top_rect = pygame.Rect(center_x, 0, panel_width, cam_top_y)
            self.rp_bottom_rect = pygame.Rect(center_x, cam_bottom_y, panel_width, full_height - cam_bottom_y)
            self.lp_top_rect = pygame.Rect(0, 0, center_x, cam_top_y)

        def setup_prediction_text():
            small_size = int(self.rp_top_rect.height * 0.3)
            big_size = int(self.rp_top_rect.height * 0.525)
            font_small = pygame.font.SysFont("Franklin Gothic Medium Condensed", small_size)
            font_big = pygame.font.SysFont("Franklin Gothic Medium Condensed", big_size)

            # Text content
            label_left = font_small.render("SIGNAL PREDICTION:", True, (0, 0, 0))
            value_left = font_big.render("NONE", True, (0, 0, 0))
            label_right = font_small.render("PROBABILITY:", True, (0, 0, 0))
            value_right = font_big.render("0%", True, (0, 0, 0))

            # Rects
            label_left_rect = label_left.get_rect()
            value_left_rect = value_left.get_rect()
            label_right_rect = label_right.get_rect()
            value_right_rect = value_right.get_rect()

            spacing = int(2 * self.scale)
            row_height = label_left_rect.height + value_left_rect.height + spacing
            y_start = self.rp_top_rect.centery - row_height // 2

            # Horizontal positions
            margin = int(15 * self.scale)
            left_x = self.rp_top_rect.left + margin
            right_x = self.rp_top_rect.right - margin

            self.prediction_text_surfaces = [
                # Left aligned (label and value)
                (label_left, (left_x, y_start)),
                (value_left, (left_x, y_start + label_left_rect.height + spacing)),

                # Right aligned (label and value)
                (label_right, (right_x - label_right_rect.width, y_start)),
                (value_right, (right_x - value_right_rect.width, y_start + label_right_rect.height + spacing)),
            ]

            # Store references to dynamically update values
            self.predicted_label_surface = value_left
            self.predicted_probability_surface = value_right

        def setup_buttons():
            font_size = int(16 * self.scale)
            font = pygame.font.SysFont("Franklin Gothic Medium Condensed", font_size)
            labels = ["END TRAINING", "START"]
            surfaces = [font.render(text, True, (0, 0, 0)) for text in labels]

            padding_w = int(25 * self.scale)
            padding_h = int(20 * self.scale)
            rects = [s.get_rect() for s in surfaces]
            width = max(r.width for r in rects) + padding_w
            height = max(r.height for r in rects) + padding_h
            y = int(317 * self.scale)

            initial_state = {
                "END TRAINING": self.training_started,
                "START": not self.training_started
            }

            self.buttons = {}
            for i, (label, surface, rect) in enumerate(zip(labels, surfaces, rects)):
                x = self.rp_bottom_rect.left + int(20 * self.scale) if i == 0 else self.rp_bottom_rect.right - width - int(20 * self.scale)
                btn_rect = pygame.Rect(x, y, width, height)
                text_pos = (x + (width - rect.width) // 2, y + (height - rect.height) // 2)
                self.buttons[label] = [False, initial_state[label], surface, text_pos, btn_rect]

        def load_guide_videos():
            self.guide_position = [0, 0]
            self.guide_videos = {}
            self.action_configs = {
                "straight_ahead": {
                    "offset": (54, 10),
                    "size": 450,
                    "frame_delay": 5,
                },
                "turn_left": {
                    "offset": (74, 10),
                    "size": 450,
                    "frame_delay": 5,
                },
                "turn_right": {
                    "offset": (54, 10),
                    "size": 450,
                    "frame_delay": 5,
                },
                "stop": {
                    "offset": (66, 12),
                    "size": 450,
                    "frame_delay": 10,
                },
                "cut_engine": {
                    "offset": (122, 55),
                    "size": 570,
                    "frame_delay": 7,
                },
                "start_engine": {
                    "offset": (56, 0),
                    "size": 430,
                    "frame_delay": 5,
                },
                "set_brakes": {
                    "offset": (62, 0),
                    "size": 450,
                    "frame_delay": 10,
                },
                "chocks_inserted": {
                    "offset": (53, 0),
                    "size": 430,
                    "frame_delay": 7,
                },
                "all_clear": {
                    "offset": (54, 0),
                    "size": 430,
                    "frame_delay": 10,
                },
            }

            self.actions = [
                "start_engine", "straight_ahead", "turn_left", "turn_right",
                "stop", "set_brakes", "cut_engine", "chocks_inserted", "all_clear"
            ]
            self.frame_delays = [self.action_configs[action]["frame_delay"] for action in self.actions]

            for action, cfg in self.action_configs.items():
                offset = cfg["offset"]
                size = cfg["size"]

                dir_path = os.path.join(resources_path, "guide_videos", action)
                frame_paths = sorted([os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".jpg")])
                frames = []

                for path in frame_paths:
                    img = pygame.image.load(path).convert()
                    if action == "turn_left":
                        img = pygame.transform.flip(img, True, False)
                    scaled = pygame.transform.smoothscale(img, (int(size * self.scale), int(size * self.scale)))
                    frames.append(scaled)

                x, y = self.guide_position
                x_offset, y_offset = offset
                self.guide_videos[action] = (
                    frames,
                    ((x - x_offset) * self.scale, (y - y_offset) * self.scale)
                )

        def load_guide_audio():
            self.guide_audio = {}
            audio_dir = os.path.join(resources_path, "guide_audio")

            for action in self.actions:
                path = os.path.join(audio_dir, f"{action}.mp3")
                if os.path.exists(path):
                    try:
                        self.guide_audio[action] = pygame.mixer.Sound(path)
                    except pygame.error as e:
                        print(f"Failed to load audio for '{action}': {e}")
                else:
                    print(f"Missing audio for action: {action}")

        # Execute setup routines
        setup_layout()
        setup_prediction_text()
        setup_buttons()
        load_guide_videos()
        load_guide_audio()
        self.setup_visual_instruction_text(self.instruction)

    def setup_visual_instruction_text(self, instruction):
        # Sizes relative to lp_top_rect
        small_size = int(self.lp_top_rect.height * 0.35)
        big_size = int(self.lp_top_rect.height * 0.75)

        # Fonts
        small_font = pygame.font.SysFont("Franklin Gothic Medium Condensed", small_size)
        big_font = pygame.font.SysFont("Franklin Gothic Medium Condensed", big_size)

        # Render "INSTRUCTIONS"
        title_text = small_font.render("INSTRUCTIONS:", True, (0, 0, 0))
        title_rect = title_text.get_rect()

        # Render instruction (e.g. "Straight Ahead")
        instruction_text = big_font.render(instruction.upper().replace("_", " "), True, (0, 0, 0))
        instruction_rect = instruction_text.get_rect()

        # Positioning
        total_height = title_rect.height + instruction_rect.height
        start_y = self.lp_top_rect.centery - total_height // 2

        self.visinstr_title_text = title_text
        self.visinstr_title_pos = (
            self.lp_top_rect.centerx - title_rect.width // 2,
            start_y
        )

        self.visinstr_instr_text = instruction_text
        self.visinstr_instr_pos = (
            self.lp_top_rect.centerx - instruction_rect.width // 2,
            start_y + title_rect.height
        )

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

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
            # Draw guide
            if self.training_started:
                self.draw_guide_animation(win, self.actions[self.current_action], self.frame_delays[self.current_action])

            scaled_frame = pygame.transform.smoothscale(self.frame_surface, self.frame_draw_size)
            win.blit(scaled_frame, self.frame_draw_pos)

            pygame.draw.rect(win, (192, 192, 192), self.rp_top_rect)
            pygame.draw.rect(win, (119, 163, 200), self.lp_top_rect)

            # Draw text
            for surface, pos in self.prediction_text_surfaces:
                win.blit(surface, pos)
            win.blit(self.visinstr_title_text, self.visinstr_title_pos)
            win.blit(self.visinstr_instr_text, self.visinstr_instr_pos)

            # Draw buttons
            for is_hovered, is_open, text, text_pos, btn_rect in self.buttons.values():
                fill = (192, 192, 192) if is_hovered and is_open else \
                    (240, 240, 240) if is_open else (132, 132, 132)
                pygame.draw.rect(win, fill, btn_rect)
                pygame.draw.rect(win, (0, 0, 0), btn_rect, border_width)
                win.blit(text, text_pos)

        pygame.display.update()

    def draw_guide_animation(self, screen, action, frame_delay):
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

    def play_guide_audio(self):
        action = self.actions[self.current_action]
        if action in self.guide_audio:
            pygame.mixer.stop()  # Stop any currently playing sound
            self.guide_audio[action].play()
        else:
            print(f"No audio loaded for action '{action}'")

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
                print(game.training_started)
                game.init_scale(new_size)
                game.init_opencv(new_size)
                game.init_panels(new_size)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                btn_label = game.button_down_detection(mouse_pos)
                if btn_label == "START":
                    game.training_started = True
                    game.instruction = game.actions[game.current_action]
                    game.setup_visual_instruction_text(game.instruction)
                    game.play_guide_audio()
                    game.buttons["START"][1] = False
                    game.buttons["END TRAINING"][1] = True

                elif btn_label == "END TRAINING":
                    game.training_started = False
                    game.instruction = "None"
                    game.current_action = 0
                    game.setup_visual_instruction_text(game.instruction)
                    pygame.mixer.stop()
                    game.buttons["START"][1] = True
                    game.buttons["END TRAINING"][1] = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    game.current_action += 1
                    if game.current_action >= len(game.actions):
                        game.current_action = 0

                    game.instruction = game.actions[game.current_action]
                    game.setup_visual_instruction_text(game.instruction)
                    game.play_guide_audio()

        mouse_pos = pygame.mouse.get_pos()
        game.button_over_detection(mouse_pos)
        game.update_frame()
        game.draw()

    game.release()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    pygame.init()
    pygame.mixer.init()

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
