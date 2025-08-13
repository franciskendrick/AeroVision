import mediapipe as mp
import numpy as np
import pygame
import cv2
import sys
import os
from keras._tf_keras.keras.models import load_model


class Menu: 
    def __init__(self, win_size):
        self.popup_active = False
        self.game_loading = False
        self.game_initialized = False

        self.init_scale(win_size)
        self.init_menu(win_size)
        self.init_popup(win_size)
        self.init_loading(win_size)

    def init_scale(self, win_size):
        base_width, base_height = 640, 360
        scale_w = win_size[0] / base_width
        scale_h = win_size[1] / base_height
        self.scale = min(scale_w, scale_h)

    def init_menu(self, win_size):
        background = pygame.image.load(f"{resources_path}/background.png")
        self.background = pygame.transform.scale(background, win_size)
    
        garamond_size = int(54 * self.scale)
        franklinbig_size = int(112 * self.scale)
        franklinsmall_size = int(20 * self.scale)
        spacing = int(-5 * self.scale)

        garamond = pygame.font.SysFont("Garamond", garamond_size, bold=True)
        franklingothic_big = pygame.font.SysFont("Franklin Gothic Medium Condensed", franklinbig_size, bold=False)
        franklingothic_small = pygame.font.SysFont("Franklin Gothic Medium Condensed", franklinsmall_size, bold=False)

        self.spacing = spacing

        raw_texts = [
            garamond.render("GROUND", True, (22, 33, 68)),
            garamond.render("AIRCRAFT", True, (22, 33, 68)),
            garamond.render("MARSHALLING", True, (22, 33, 68)),
            franklingothic_big.render("SIMULATOR", True, (66, 140, 226))
        ]

        self.texts = []
        text_heights = sum(text.get_rect().height for text in raw_texts)
        total_spacing = self.spacing * (len(raw_texts) - 1) - (self.spacing * 6)
        start_y = win_size[1] // 2 - (text_heights + total_spacing) // 2

        current_y = start_y
        for text in raw_texts:
            rect = text.get_rect()
            x = win_size[0] // 2 - rect.width // 2
            self.texts.append([text, (x, current_y)])
            current_y += rect.height + self.spacing
        else:
            current_y -= self.spacing * 6

        button_labels = ["CONNECT TO PROTOTYPE", "START"]
        button_surfaces = [franklingothic_small.render(label, True, (0, 0, 0)) for label in button_labels]

        marshalling_surface = self.texts[2][0]
        marshalling_rect = marshalling_surface.get_rect()
        marshalling_x = win_size[0] // 2 - marshalling_rect.width // 2
        marshalling_right = marshalling_x + marshalling_rect.width

        base_padding_w, base_padding_h = 25, 20
        padding_w = int(base_padding_w * self.scale)
        padding_h = int(base_padding_h * self.scale)

        self.buttons = {}
        for idx, (label, surface) in enumerate(zip(button_labels, button_surfaces)):
            text_rect = surface.get_rect()
            if idx == 0:
                open_status = True
                x = marshalling_x
            else:
                open_status = False
                x = marshalling_right - text_rect.width

            y = current_y

            button_rect = pygame.Rect(
                x - (padding_w // 2),
                y - (padding_h // 2),
                text_rect.width + padding_w,
                text_rect.height + padding_h
            )

            text_x = button_rect.x + (button_rect.width - text_rect.width) // 2
            text_y = button_rect.y + (button_rect.height - text_rect.height) // 2

            self.buttons[label] = [False, open_status, surface, (text_x, text_y), button_rect]

    def init_popup(self, win_size):
        popup_lines = [
            "CONNECTED SUCCESSFULLY TO",
            "PROTOTYPE AIRCRAFT!",
            "",
            "PLEASE PRESS",
            "THE START BUTTON TO CONTINUE",
            "ASSESSMENT AND TRAINING."
        ]

        font = pygame.font.SysFont("Franklin Gothic Medium Condensed", int(32 * self.scale))
        self.popup_surfaces = [font.render(line, True, (0, 0, 0)) for line in popup_lines]

        self.popup_width = max(s.get_width() for s in self.popup_surfaces) + int(40 * self.scale)
        self.popup_height = sum(s.get_height() for s in self.popup_surfaces) + int(10 * self.scale) * (len(self.popup_surfaces) - 2) + int(30 * self.scale)

        win_w, win_ht = win_size
        titlebar_height = int(22 * self.scale)
        popup_total_height = self.popup_height + titlebar_height
        vertical_center = (win_ht - popup_total_height) // 2 - (18 * self.scale)

        popup_titlebar_rect = pygame.Rect(
            (win_w - self.popup_width) // 2,
            vertical_center,
            self.popup_width,
            titlebar_height
        )

        popup_rect = pygame.Rect(
            (win_w - self.popup_width) // 2,
            vertical_center + titlebar_height,
            self.popup_width,
            self.popup_height
        )

        bahnschrift = pygame.font.SysFont("Bahnschrift", int(16 * self.scale), bold=False)
        popup_closebutton_text = bahnschrift.render("X", True, (0, 0, 0))
        close_text_rect = popup_closebutton_text.get_rect()

        pad_x = int(8 * self.scale)
        pad_y = int(2 * self.scale)
        button_w = close_text_rect.width + 2 * pad_x
        button_h = close_text_rect.height + 2 * pad_y

        popup_closebutton_rect = pygame.Rect(
            popup_titlebar_rect.right - button_w - int(5 * self.scale),
            popup_titlebar_rect.centery - button_h // 2,
            button_w,
            button_h
        )

        popup_closebutton_text_pos = (
            popup_closebutton_rect.x + (button_w - close_text_rect.width) // 2,
            popup_closebutton_rect.y + (button_h - close_text_rect.height) // 2
        )

        self.popup = [popup_titlebar_rect, popup_rect, popup_closebutton_text, popup_closebutton_rect, popup_closebutton_text_pos]

    def init_loading(self, win_size):
        titlebar_height = int(22 * self.scale)
        font_size = int(64 * self.scale)

        win_w, win_ht = win_size

        # Use same dimensions as the existing popup
        popup_total_height = self.popup_height + titlebar_height
        vertical_center = (win_ht - popup_total_height) // 2 - (18 * self.scale)

        loading_titlebar_rect = pygame.Rect(
            (win_w - self.popup_width) // 2,
            vertical_center,
            self.popup_width,
            titlebar_height
        )

        loading_rect = pygame.Rect(
            (win_w - self.popup_width) // 2,
            vertical_center + titlebar_height,
            self.popup_width,
            self.popup_height
        )

        font = pygame.font.SysFont("Franklin Gothic Medium Condensed", font_size, bold=False)
        loading_surface = font.render("LOADING...", True, (0, 0, 0))

        loading_x = loading_rect.x + (loading_rect.width - loading_surface.get_width()) // 2
        loading_y = loading_rect.y + (loading_rect.height - loading_surface.get_height()) // 2

        self.loading = [loading_titlebar_rect, loading_rect, loading_surface, (loading_x, loading_y)]

    def draw(self, win):
        win.blit(self.background, (0, 0))
        
        border_width = max(1, round(2 * self.scale))

        for is_hovered, is_open, text, text_pos, btn_rect in self.buttons.values():
            fill = (192, 192, 192) if is_hovered and is_open else \
                   (240, 240, 240) if is_open else (132, 132, 132)
            pygame.draw.rect(win, fill, btn_rect)
            pygame.draw.rect(win, (0, 0, 0), btn_rect, border_width)
            win.blit(text, text_pos)

        if not self.popup_active and not self.game_loading:
            for text_surface, pos in self.texts:
                win.blit(text_surface, pos)
        elif self.game_loading:
            loading_titlebar_rect, loading_rect, loading_surface, loading_pos = self.loading
            pygame.draw.rect(win, (255, 255, 255), loading_rect)
            pygame.draw.rect(win, (192, 192, 192), loading_titlebar_rect)
            win.blit(loading_surface, loading_pos)
        elif self.popup_active:
            popup_titlebar_rect, popup_rect, popup_closebutton_text, popup_closebutton_rect, popup_closebutton_text_pos = self.popup
            pygame.draw.rect(win, (255, 255, 255), popup_rect)
            pygame.draw.rect(win, (192, 192, 192), popup_titlebar_rect)
            pygame.draw.rect(win, (162, 162, 162), popup_closebutton_rect)
            win.blit(popup_closebutton_text, popup_closebutton_text_pos)

            cursor_y = popup_rect.y + int(10 * self.scale)
            for idx, surf in enumerate(self.popup_surfaces):
                x = popup_rect.x + (self.popup_width - surf.get_width()) // 2
                win.blit(surf, (x, cursor_y))
                if idx < len(self.popup_surfaces) - 1:
                    cursor_y += surf.get_height() + int(10 * self.scale)

        pygame.display.update()

    def menubutton_down_detection(self, mouse_pos):
        for label, (_, is_open, *_, btn_rect) in self.buttons.items():
            if is_open and btn_rect.collidepoint(mouse_pos):
                return label

    def popupbutton_down_detection(self, mouse_pos):
        if self.popup[3].collidepoint(mouse_pos):
            return True

    def menubutton_over_detection(self, mouse_pos):
        for button in self.buttons.values():
            button[0] = button[4].collidepoint(mouse_pos)


class Game:
    ACTIONS         = ["cut_engine", "start_engine", "stop", "straight_ahead", "turn_left", "turn_right"]
    MODEL_PATH      = r"User Interface/model.h5"
    SEQUENCE_LENGTH = 90
    THRESHOLD       = 0.4

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
        self.current_action = 7
        self.button_states = {
            "END TRAINING": False,
            "START": False
        }
        self.visibility_toggle = "X"
        self.signal_detected = False
        self.assessment_stage = False

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

        self.model = load_model(self.MODEL_PATH)
        self.sequence = []
        self.signal = "NONE"
        self.confidence = 0

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
            value_left = font_big.render(self.signal, True, (0, 0, 0))
            label_right = font_small.render("PROBABILITY:", True, (0, 0, 0))
            value_right = font_big.render(f"{self.confidence * 100:.0f}%", True, (0, 0, 0))

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

        def setup_bookend_buttons():
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
            
            self.buttons = {}
            for i, (label, surface, rect) in enumerate(zip(labels, surfaces, rects)):
                x = self.rp_bottom_rect.left + int(20 * self.scale) if i == 0 else self.rp_bottom_rect.right - width - int(20 * self.scale)
                btn_rect = pygame.Rect(x, y, width, height)
                text_pos = (x + (width - rect.width) // 2, y + (height - rect.height) // 2)
                self.buttons[label] = [False, False, surface, text_pos, btn_rect]

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
                "stop", "set_brakes", "cut_engine", "all_clear"
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

        def load_detection_audio():
            self.detection_audio = {}
            audio_dir = os.path.join(resources_path, "detection_audio")

            for action in self.actions:
                filename = f"{action}_dec.mp3"
                path = os.path.join(audio_dir, filename)
                if os.path.exists(path):
                    try:
                        self.detection_audio[action] = pygame.mixer.Sound(path)
                    except pygame.error as e:
                        print(f"[Detection] Failed to load '{filename}': {e}")
                else:
                    print(f"[Detection] Missing: {filename}")

        def load_instruction_audio():
            self.instruction_audio = {}
            audio_dir = os.path.join(resources_path, "instruction_audio")

            for action in self.actions:
                filename = f"{action}_ins.mp3"
                path = os.path.join(audio_dir, filename)
                if os.path.exists(path):
                    try:
                        self.instruction_audio[action] = pygame.mixer.Sound(path)
                    except pygame.error as e:
                        print(f"[Instruction] Failed to load '{filename}': {e}")
                else:
                    print(f"[Instruction] Missing: {filename}")

        def load_bookends_audio():
            self.bookends_audio = {}
            audio_dir = os.path.join(resources_path, "bookends_audio")
            
            for name in ["introduction", "ending"]:
                filename = f"{name}.mp3"
                path = os.path.join(audio_dir, filename)
                if os.path.exists(path):
                    try:
                        self.bookends_audio[name] = pygame.mixer.Sound(path)
                    except pygame.error as e:
                        print(f"[Bookends] Failed to load '{filename}': {e}")
                else:
                    print(f"[Bookends] Missing: {filename}")

        # Execute setup routines
        setup_layout()
        setup_prediction_text()

        setup_bookend_buttons()
        self.setup_visibility_buttons()

        load_guide_videos()

        load_detection_audio()
        load_instruction_audio()
        load_bookends_audio()

        self.setup_visual_instruction_text(self.instruction)

    def setup_visibility_buttons(self):
        font_size = int(24 * self.scale)
        font = pygame.font.SysFont("Franklin Gothic Medium Condensed", font_size)
        surface = font.render(self.visibility_toggle, True, (0, 0, 0))

        padding_w = int(25 * self.scale)
        padding_h = int(14 * self.scale)
        rect = surface.get_rect()
        width = rect.width + padding_w
        height = rect.height + padding_h
        y = int(317 * self.scale)

        x = self.rp_bottom_rect.centerx - width // 2
        btn_rect = pygame.Rect(x, y, width, height)
        text_pos = (x + (width - rect.width) // 2, y + (height - rect.height) // 2)
        self.visibility_button = [False, True, surface, text_pos, btn_rect]

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

    # Update
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipe Pose processing
        results = self.pose.process(frame_rgb)
        if results.pose_landmarks and self.visibility_toggle == "X":
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
            )

        # Keypoint vector
        keypoints = self.extract_keypoints_full(results)
        self.sequence.append(keypoints)
        if len(self.sequence) > self.SEQUENCE_LENGTH:
            self.sequence.pop(0)

        # Default signal
        signal = ""

        # Pose-based gesture overrides (manual rules)
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            right_wrist_y = lm[15].y
            right_elbow_y = lm[14].y
            left_wrist_y = lm[16].y
            left_hip_y = lm[23].y
            right_shoulder_y = lm[12].y
            right_eye = lm[5].y
            
            if (right_wrist_y < right_eye and right_wrist_y < right_elbow_y and
                    left_wrist_y > left_hip_y):
                signal = "all_clear"
                self.signal = signal
                self.confidence = 1.0
                self.update_prediction_text()
            elif (right_wrist_y < right_shoulder_y and right_wrist_y < right_elbow_y and
                    left_wrist_y > left_hip_y):
                signal = "set_brakes"
                self.signal = signal
                self.confidence = 1.0
                self.update_prediction_text()

        # Prediction
        if signal == "" and len(self.sequence) == self.SEQUENCE_LENGTH:
            input_seq = np.expand_dims(np.array(self.sequence), axis=0)  # shape: (1, 90, 99)
            probs = self.model.predict(input_seq, verbose=0)[0]
            max_idx = np.argmax(probs)
            self.confidence = probs[max_idx]

            if self.confidence > self.THRESHOLD:
                self.signal = self.ACTIONS[max_idx]
                self.update_prediction_text()

        # Convert for Pygame (after drawing)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.flip(frame_rgb, 1)
        frame_rgb = np.rot90(frame_rgb)
        self.frame_surface = pygame.surfarray.make_surface(frame_rgb)

    def extract_keypoints_full(self, results):
        if not results.pose_landmarks:
            return np.zeros(33 * 3)
        return np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()

    def update_prediction_text(self):
        big_size = int(self.rp_top_rect.height * 0.525)
        font_big = pygame.font.SysFont("Franklin Gothic Medium Condensed", big_size)

        # Re-render dynamic values
        formatted_signal = self.signal.replace("_", " ").upper()
        value_left = font_big.render(formatted_signal, True, (0, 0, 0))
        value_right = font_big.render(f"{self.confidence * 100:.0f}%", True, (0, 0, 0))

        # Update stored surfaces
        self.predicted_label_surface = value_left
        self.predicted_probability_surface = value_right

        # Update their positions in the main surfaces list
        spacing = int(2 * self.scale)
        small_size = int(self.rp_top_rect.height * 0.3)
        font_small = pygame.font.SysFont("Franklin Gothic Medium Condensed", small_size)
        label_left_rect = font_small.render("SIGNAL PREDICTION:", True, (0, 0, 0)).get_rect()
        label_right_rect = font_small.render("PROBABILITY:", True, (0, 0, 0)).get_rect()
        row_height = label_left_rect.height + value_left.get_rect().height + spacing
        y_start = self.rp_top_rect.centery - row_height // 2
        margin = int(15 * self.scale)
        left_x = self.rp_top_rect.left + margin
        right_x = self.rp_top_rect.right - margin

        # Update only the value parts (indexes 1 and 3)
        self.prediction_text_surfaces[1] = (value_left, (left_x, y_start + label_left_rect.height + spacing))
        self.prediction_text_surfaces[3] = (value_right, (right_x - value_right.get_width(), y_start + label_right_rect.height + spacing))

    # Draw
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

            is_hovered, is_open, text, text_pos, btn_rect = self.visibility_button
            fill = (192, 192, 192) if is_hovered and is_open else \
                    (240, 240, 240) if is_open else (132, 132, 132)
            pygame.draw.rect(win, fill, btn_rect)
            pygame.draw.rect(win, (0, 0, 0), btn_rect, border_width)
            win.blit(text, text_pos)

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

    # Audio
    def play_detection_audio(self):
        action = self.actions[self.current_action]
        if action in self.detection_audio:
            pygame.mixer.stop()
            self.detection_audio[action].play()
        else:
            print(f"No detection audio loaded for action '{action}'")

    def play_instruction_audio(self):
        action = self.actions[self.current_action]
        if action in self.instruction_audio:
            pygame.mixer.stop()
            self.instruction_audio[action].play()
        else:
            print(f"No instruction audio loaded for action '{action}'")

    def play_bookends_audio(self, name):
        if name in self.bookends_audio:
            pygame.mixer.stop()
            self.bookends_audio[name].play()
        else:
            print(f"No bookend audio loaded for name '{name}'")
    
    def stop_current_audio(self):
        pygame.mixer.stop()

    # Buttons
    def button_over_detection(self, mouse_pos):
        for button in self.buttons.values():
            button[0] = button[4].collidepoint(mouse_pos)

        button = self.visibility_button
        button[0] = button[4].collidepoint(mouse_pos)

    def button_down_detection(self, mouse_pos):
        for label, (_, is_open, *_, btn_rect) in self.buttons.items():
            if is_open and btn_rect.collidepoint(mouse_pos):
                return label
            
        _, is_open, *_, btn_rect = self.visibility_button
        if is_open and btn_rect.collidepoint(mouse_pos):
            return "VISIBILITY"

    # Quit
    def release(self):
        self.cap.release()


class GameOver:
    def __init__(self, win_size, signal_scores):
        self.signal_scores = signal_scores

        self.init(win_size)
    
    def init(self, win_size):
        self.init_scale(win_size)

        self.win_size = win_size

        self.font = pygame.font.SysFont("Franklin Gothic Medium Condensed", int(26 * self.scale))
        self.title_font = pygame.font.SysFont("Franklin Gothic Medium Condensed", int(34 * self.scale), bold=True)

        self.bg_color = (255, 255, 255)
        self.border_color = (0, 0, 0)

        self.prepare_text()
        self.measure_button_dimensions()  # New
        self.calculate_popup_rect()       # Now we can safely calculate height
        self.prepare_buttons()            # Now we can safely position buttons

    def init_scale(self, win_size):
        base_width, base_height = 640, 360
        scale_w = win_size[0] / base_width
        scale_h = win_size[1] / base_height
        self.scale = min(scale_w, scale_h)

    def prepare_text(self):
        self.text_surfaces = []
        total_score = 0

        for label, score in self.signal_scores.items():
            total_score += score
            text = f"{label} . . . . . {int(score * 100)}%"
            surface = self.font.render(text, True, (0, 0, 0))
            self.text_surfaces.append(surface)

        self.overall_score = total_score / len(self.signal_scores)
        self.status = "PASSED" if self.overall_score >= 0.80 else "NEEDS IMPROVEMENT"

        self.overall_surface = self.title_font.render(
            f"OVERALL SCORE: {round(self.overall_score * 5, 2)} / 5.0", True, (0, 0, 0)
        )
        self.status_surface = self.font.render(
            f"Status: {self.status}", True, (0, 128, 0) if self.status == "PASSED" else (200, 0, 0)
        )

    def prepare_buttons(self):
        self.buttons = {}

        y = self.popup_rect.bottom - self.button_height - int(15 * self.scale)
        total_button_width = 2 * self.button_width + self.button_spacing

        x_left = self.popup_rect.centerx - (total_button_width // 2)
        x_right = x_left + self.button_width + self.button_spacing

        for (label, surf, rect), x in zip(self.button_surfaces, [x_left, x_right]):
            btn_rect = pygame.Rect(x, y, self.button_width, self.button_height)
            text_pos = (x + (self.button_width - rect.width) // 2, y + (self.button_height - rect.height) // 2)
            self.buttons[label] = [False, True, surf, text_pos, btn_rect]

    def measure_button_dimensions(self):
        font_size = int(16 * self.scale)
        font = pygame.font.SysFont("Franklin Gothic Medium Condensed", font_size)
        labels = ["Exit", "Retake Mission"]
        surfaces = [font.render(label, True, (0, 0, 0)) for label in labels]

        padding_w = int(25 * self.scale)
        padding_h = int(20 * self.scale)
        rects = [s.get_rect() for s in surfaces]
        
        self.button_width = max(r.width for r in rects) + padding_w
        self.button_height = max(r.height for r in rects) + padding_h
        self.button_spacing = int(20 * self.scale)
        self.button_surfaces = list(zip(labels, surfaces, rects))

    def calculate_popup_rect(self):
        # spacing = int(30 * self.scale)
        # padding_top = int(5 * self.scale)
        # padding_bottom = int(5 * self.scale)
        # extra_spacing = int(25 * self.scale)

        # num_lines = len(self.text_surfaces) + 2  # main lines + overall + status
        # total_text_height = (num_lines * spacing) + extra_spacing
        self.height = int((self.win_size[1]))
        self.width = int((self.win_size[0] // 2))

        self.popup_rect = pygame.Rect(
            0,
            0,
            self.width,
            self.height
        )

    def draw(self, win):
        pygame.draw.rect(win, self.bg_color, self.popup_rect)

        spacing = int(30 * self.scale)
        extra_spacing = int(10 * self.scale)

        cursor_y = self.popup_rect.y + int(20 * self.scale)

        for surf in self.text_surfaces:
            x = self.popup_rect.centerx - surf.get_width() // 2
            win.blit(surf, (x, cursor_y))
            cursor_y += spacing

        cursor_y += extra_spacing
        x = self.popup_rect.centerx - self.overall_surface.get_width() // 2
        win.blit(self.overall_surface, (x, cursor_y))

        cursor_y += spacing
        x = self.popup_rect.centerx - self.status_surface.get_width() // 2
        win.blit(self.status_surface, (x, cursor_y))

        # Draw buttons
        border_width = max(1, round(2 * self.scale))
        for is_hovered, is_open, text, text_pos, btn_rect in self.buttons.values():
            fill = (192, 192, 192) if is_hovered and is_open else \
                (240, 240, 240) if is_open else (132, 132, 132)
            pygame.draw.rect(win, fill, btn_rect)
            pygame.draw.rect(win, (0, 0, 0), btn_rect, border_width)
            win.blit(text, text_pos)

    def button_over_detection(self, mouse_pos):
        for button in self.buttons.values():
            button[0] = button[4].collidepoint(mouse_pos)

    def button_down_detection(self, mouse_pos):
        for label, (_, is_open, *_, btn_rect) in self.buttons.items():
            if is_open and btn_rect.collidepoint(mouse_pos):
                return label


def menu_loop():
    global game
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            elif event.type == pygame.VIDEORESIZE:
                # Enforce minimum size of 640x360
                new_width = max(640, event.w)
                new_height = max(360, event.h)
                new_size = (new_width, new_height)

                pygame.display.set_mode(new_size, pygame.RESIZABLE)

                # Save state
                was_popup_active = menu.popup_active
                start_button_state = menu.buttons["START"][1]  # True if it was open

                # Reinitialize
                menu.init_scale(new_size)
                menu.init_menu(new_size)
                menu.init_popup(new_size)
                menu.init_loading(new_size)

                if menu.game_initialized:
                    game.init_scale(new_size)
                    game.init_opencv(new_size)
                    game.init_panels(new_size)

                # Restore state
                menu.popup_active = was_popup_active
                menu.buttons["START"][1] = start_button_state

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if menu.popup_active:
                    if menu.popupbutton_down_detection(mouse_pos):
                        menu.popup_active = False

                btn_label = menu.menubutton_down_detection(mouse_pos)
                if btn_label == "CONNECT TO PROTOTYPE":
                    if not menu.game_initialized:
                        menu.game_loading = True
                    menu.popup_active = True
                elif btn_label == "START":
                    game_loop()

        mouse_pos = pygame.mouse.get_pos()
        menu.menubutton_over_detection(mouse_pos)
        menu.draw(win)

        if menu.game_loading and not menu.game_initialized:
            current_winsize = pygame.display.get_surface().get_size()
            game = Game(current_winsize)
            menu.game_initialized = True
            menu.game_loading = False
            menu.buttons["START"][1] = True

    pygame.quit()
    sys.exit()


def game_loop():
    game.play_bookends_audio("introduction")
    scores = {
        "Start Engine": 1.00,
        "Straight Ahead": 1.00,
        "Turn Left": 1.00,
        "Turn Right": 1.00,
        "Set Brakes": 1.00,
        "Cut Engines": 1.00,
        "All Clear": 1.00
    }
    gameover = GameOver(win_size, scores)
    game.assessment_stage = False

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

                gameover.init_scale(new_size)
                gameover.init(new_size)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if not game.assessment_stage:
                    btn_label = game.button_down_detection(mouse_pos)
                    if btn_label == "START":
                        game.training_started = True
                        game.instruction = game.actions[game.current_action]
                        game.setup_visual_instruction_text(game.instruction)
                        game.play_instruction_audio()

                        game.buttons["START"][1] = False
                        game.buttons["END TRAINING"][1] = True
                        game.button_states["START"] = False
                        game.button_states["END TRAINING"] = True

                    elif btn_label == "END TRAINING":
                        game.training_started = False
                        game.instruction = "None"
                        game.current_action = 0

                        game.setup_visual_instruction_text(game.instruction)
                        pygame.mixer.stop()

                        game.buttons["START"][1] = True
                        game.buttons["END TRAINING"][1] = False
                        game.button_states["START"] = True
                        game.button_states["END TRAINING"] = False

                    elif btn_label == "VISIBILITY":
                        game.visibility_toggle = "+" if game.visibility_toggle == "X" else "X"

                        game.setup_visibility_buttons()
                else:
                    btn_label = gameover.button_down_detection(mouse_pos)
                    if btn_label == "Exit":
                        run = False

                    elif btn_label == "Retake Mission":
                        # 1) Re-init the Game object to clear out all state:
                        game.__init__(win_size)
                        
                        # 2) Reset any assessment flags
                        game.assessment_stage = False
                        game.training_started = False
                        game.signal_detected = False
                        
                        # 3) Reset UI state to show only the START button
                        game.buttons["START"][1] = False
                        game.buttons["END TRAINING"][1] = False
                        game.button_states["START"] = False
                        game.button_states["END TRAINING"] = False
                        
                        # 4) Reset the “instruction” text to blank (or “None”)
                        game.instruction = "None"
                        game.setup_visual_instruction_text(game.instruction)
                        
                        # 5) (Optionally) play an intro bookend again
                        game.play_bookends_audio("introduction")

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    game.stop_current_audio()
                    if not game.button_states["START"] and not game.training_started:
                        game.buttons["START"][1] = True
                        game.buttons["END TRAINING"][1] = False
                        game.button_states["START"] = True
                        game.button_states["END TRAINING"] = False

        if game.training_started:
            if not pygame.mixer.get_busy() and not game.signal_detected:
                if game.signal == game.actions[game.current_action]:
                    game.play_detection_audio()
                    game.signal_detected = True

            if not pygame.mixer.get_busy() and game.signal_detected:
                game.current_action += 1
                if game.current_action >= len(game.actions):
                    # gameover = GameOver(win_size, scores)
                    gameover.init(win.get_size())
                    game.assessment_stage = True
                    game.training_started = False
                    print("ASSESSMENT ENDED")
                else:
                    game.instruction = game.actions[game.current_action]
                    game.setup_visual_instruction_text(game.instruction)
                    game.play_instruction_audio()
                    game.signal_detected = False
        else:
            if not pygame.mixer.get_busy():
                game.buttons["START"][1] = True
                game.button_states["START"] = True

        game.update_frame()
        game.draw()
        mouse_pos = pygame.mouse.get_pos()
        if not game.assessment_stage:
            game.button_over_detection(mouse_pos)
        else:
            gameover.button_over_detection(mouse_pos)
            gameover.draw(win)
        pygame.display.update()

    game.release()
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

    menu = Menu(win_size)

    menu_loop()
