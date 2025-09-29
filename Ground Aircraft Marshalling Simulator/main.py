import csv
import ctypes
from datetime import datetime
import numpy as np
import os
import pygame
import requests
import sys
import time


def resource_path(relative_path: str) -> str:
    if getattr(sys, 'frozen', False):
        # running in bundle
        base = getattr(sys, '_MEIPASS', None)
        if base:
            return os.path.join(base, relative_path)
        # fallback to executable dir (useful for --onedir)
        return os.path.join(os.path.dirname(sys.executable), relative_path)
    else:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)


# Module-level path used throughout your code (keeps existing variable name to minimize edits)
resources_path = resource_path("resources")


def get_user_data_dir(app_name="GroundAircraftMarshalling"):
    if getattr(sys, 'frozen', False):
        # Whether onefile or onedir, write next to the executable
        return os.path.dirname(sys.executable)
    else:
        # Development mode: write next to the project folder
        return os.path.dirname(os.path.abspath(__file__))


user_data_dir = get_user_data_dir()
os.makedirs(user_data_dir, exist_ok=True)


def save_scores_to_csv(scores, overall_score, status):
    # write scores.csv into a persistent user_data_dir (not the PyInstaller temp folder)
    filename = os.path.join(user_data_dir, "scores.csv")
    file_exists = os.path.isfile(filename)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [now] + [scores.get(key, "") for key in scores.keys()] + [overall_score] + [status]

    with open(filename, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ["Date & Time"] + list(scores.keys()) + ["Overall Score"] + ["Status"]
            writer.writerow(header)
        writer.writerow(row)


def get_score(count, total_score):
    if count == 0:
        overall_pct = 0
    else:
        overall_pct = int(round(total_score / count, 0))

    # Tiered status
    if overall_pct >= 90:
        status = "EXCELLENT"
        color = (0, 128, 0)
    elif overall_pct >= 75:
        status = "GOOD"
        color = (0, 128, 0)
    elif overall_pct >= 50:
        status = "NEEDS IMPROVEMENT"
        color = (200, 140, 0)
    else:
        status = "UNSATISFACTORY"
        color = (200, 0, 0)

    return overall_pct, status, color


def get_pygame_window_pos():
    hwnd = pygame.display.get_wm_info()['window']
    rect = ctypes.wintypes.RECT()
    ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))
    return (rect.left, rect.top)


def pretty_label(key):
    mapping = {
        "start_engine": "Start Engine",
        "straight_ahead": "Straight Ahead",
        "turn_left": "Turn Left",
        "turn_right": "Turn Right",
        "stop": "Stop",
        "set_brakes": "Set Brakes",
        "chocks_inserted": "Chocks Inserted",
        "cut_engine": "Cut Engines",
        "all_clear": "All Clear",
    }
    return mapping[key]


def command_converter(label):
    mapping = {
        "start_engine": "engine_on",
        "straight_ahead": "forward",
        "turn_left": "left",
        "turn_right": "right",
        "stop": "stop",
        "cut_engine": "engine_off",
    }
    try:
        return mapping[label]
    except KeyError:
        return 


def send_command(command, timeout=1.0):
    url = f"{base_url}/{command}"
    try:
        requests.get(url, timeout=timeout)
        print(f"Sent command: {command}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending request to {url}: {e}")
        print("Check that you are connected to the ESP8266's Wi-Fi network.")


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

        button_labels = ["CONNECT TO PROTOTYPE", "REAL-TIME", "TRAINING & ASSESSMENT"]
        button_surfaces = [franklingothic_small.render(label, True, (0, 0, 0)) for label in button_labels]

        marshalling_surface = self.texts[2][0]
        marshalling_rect = marshalling_surface.get_rect()
        marshalling_x = win_size[0] // 2 - marshalling_rect.width // 2
        marshalling_center_x = marshalling_x + marshalling_rect.width // 2

        base_padding_w, base_padding_h = 25, 20
        padding_w = int(base_padding_w * self.scale)
        padding_h = int(base_padding_h * self.scale)

        # spacing between buttons
        spacing = int(15 * self.scale)

        self.buttons = {}
        total_width = sum(surface.get_rect().width + padding_w for surface in button_surfaces) + spacing * (len(button_labels) - 1)
        start_x = marshalling_center_x - total_width // 2

        for idx, (label, surface) in enumerate(zip(button_labels, button_surfaces)):
            text_rect = surface.get_rect()
            x = start_x
            y = current_y

            button_rect = pygame.Rect(
                x - (padding_w // 2),
                y - (padding_h // 2),
                text_rect.width + padding_w,
                text_rect.height + padding_h
            )

            text_x = button_rect.x + (button_rect.width - text_rect.width) // 2
            text_y = button_rect.y + (button_rect.height - text_rect.height) // 2

            # Only the first button starts "open" like before
            self.buttons[label] = [False, (idx == 0), surface, (text_x, text_y), button_rect]

            # move x for next button
            start_x += text_rect.width + padding_w + spacing

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
    ACTIONS = [
        "chocks_inserted", "cut_engine", "start_engine", "stop",
        "straight_ahead", "turn_left", "turn_right"
    ]
    MODEL_PATH = resource_path("model.h5")
    if getattr(sys, 'frozen', False):
        exe_model = os.path.join(os.path.dirname(sys.executable), "model.h5")
        if os.path.exists(exe_model):
            MODEL_PATH = exe_model
    SEQUENCE_LENGTH = 90
    THRESHOLD       = 0.4

    # Scoring parameters (lead's formula)
    PENALTY_RATE    = 5.0   # percent per second
    TMAX            = 20.0  # seconds; cap for TimeToCorrectError; TimetocorrectMAX
    ACCEPT_N        = 5     # consecutive frames required to accept a detection

    def __init__(self, win_size):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
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
        self.button_states = {
            "END TRAINING": False,
            "START": False
        }
        self.visibility_toggle = "X"
        self.signal_detected = False
        self.assessment_stage = False

        # Scoring state
        self.scores = {}
        self.t_prompt = None
        self.t_prompt_end = None
        self.accept_counter = 0
        self.accepted_for_action = False

        # Warning
        self.last_warning_time = None
        self.last_wrong_time = None
        self.warning_played = False
        self.first_warning_played = False
        self.waiting_for_interval = False
        self.audio_end_time = None

        self.init_scale(win_size)
        self.init_opencv(win_size)
        self.init_panels(win_size)

    def init_scale(self, win_size):
        base_width, base_height = 640, 360
        scale_w = win_size[0] / base_width
        scale_h = win_size[1] / base_height
        self.scale = min(scale_w, scale_h)

    def init_opencv(self, win_size):
        # Dummy frame to calculate aspect ratio
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
        self.confidence = 0.0

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
            self.lp_bottom_rect = pygame.Rect(0, cam_bottom_y, panel_width, full_height - cam_bottom_y)

        def setup_prediction_text():
            small_size = int(self.rp_top_rect.height * 0.3)
            big_size = int(self.rp_top_rect.height * 0.525)
            font_small = pygame.font.SysFont("Franklin Gothic Medium Condensed", small_size)
            font_big = pygame.font.SysFont("Franklin Gothic Medium Condensed", big_size)

            label_left = font_small.render("SIGNAL PREDICTION:", True, (0, 0, 0))
            value_left = font_big.render(self.signal, True, (0, 0, 0))
            label_right = font_small.render("PROBABILITY:", True, (0, 0, 0))
            value_right = font_big.render(f"{self.confidence * 100:.0f}%", True, (0, 0, 0))

            label_left_rect = label_left.get_rect()
            value_left_rect = value_left.get_rect()
            label_right_rect = label_right.get_rect()
            value_right_rect = value_right.get_rect()

            spacing = int(2 * self.scale)
            row_height = label_left_rect.height + value_left_rect.height + spacing
            y_start = self.rp_top_rect.centery - row_height // 2

            margin = int(15 * self.scale)
            left_x = self.rp_top_rect.left + margin
            right_x = self.rp_top_rect.right - margin

            self.prediction_text_surfaces = [
                (label_left, (left_x, y_start)),
                (value_left, (left_x, y_start + label_left_rect.height + spacing)),
                (label_right, (right_x - label_right_rect.width, y_start)),
                (value_right, (right_x - value_right_rect.width, y_start + label_right_rect.height + spacing)),
            ]

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
                self.buttons[label] = [False, False, surface, text_pos, btn_rect]  # is_hovered, is_open, text, text_pos, btn_rect

        def load_guide_videos():
            self.guide_position = [0, 0]
            self.guide_videos = {}
            self.action_configs = {
                "straight_ahead": {"offset": (64, 10), "size": 450 * 0.95, "frame_delay": 5},
                "turn_left": {"offset": (43, 10), "size": 450 * 0.95, "frame_delay": 5},
                "turn_right": {"offset": (64, 10), "size": 450 * 0.95, "frame_delay": 5},
                "stop": {"offset": (52, 12), "size": 450 * 0.95, "frame_delay": 10},
                "cut_engine": {"offset": (113, 55), "size": 570 * 0.95, "frame_delay": 7},
                "start_engine": {"offset": (43, 3), "size": 430 * 0.95, "frame_delay": 5},
                "set_brakes": {"offset": (27, 2), "size": 450 * 0.95, "frame_delay": 10},
                "chocks_inserted": {"offset": (38, -7), "size": 430 * 0.92, "frame_delay": 7},
                "all_clear": {"offset": (44, 0), "size": 430 * 0.95, "frame_delay": 10},
            }

            self.actions = [
                "start_engine", "straight_ahead", "turn_left", "turn_right",
                "stop", "set_brakes", "chocks_inserted", "cut_engine", "all_clear"
            ]
            self.frame_delays = [self.action_configs[action]["frame_delay"] for action in self.actions]

            for action, cfg in self.action_configs.items():
                offset = cfg["offset"]
                size = cfg["size"]

                dir_path = os.path.join(resources_path, "guide_videos", action)
                if not os.path.isdir(dir_path):
                    print(f"[GuideVideos] Missing directory: {dir_path}")
                    frame_paths = []
                else:
                    frame_paths = sorted(
                        [os.path.join(dir_path, f) for f in os.listdir(dir_path)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                    )
                frames = []

                for path in frame_paths:
                    img = pygame.image.load(path).convert()
                    if action not in ["turn_left", "set_brakes"]:
                        img = pygame.transform.flip(img, True, False)
                    scaled = pygame.transform.smoothscale(img, (int(size * self.scale), int(size * self.scale)))
                    frames.append(scaled)

                x, y = self.guide_position
                x_offset, y_offset = offset
                self.guide_videos[action] = (
                    frames,
                    ((x - x_offset) * self.scale, (y - y_offset) * self.scale)
                )

            small_size = int(16 * self.scale)
            font_small = pygame.font.SysFont("Franklin Gothic Medium Condensed", small_size)
            self.pilots_pov_text = font_small.render("From Pilot's Point of View", True, (0, 0, 0))
            self.pilots_pov_rect = self.pilots_pov_text.get_rect()
            self.pilots_pov_rect.topleft = (self.lp_top_rect.left + 2, self.lp_top_rect.bottom + 2)

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

        def load_warning_audio():
            filename = "warning_audio.mp3"
            path = os.path.join(resources_path, filename)
            if os.path.exists(path):
                try:
                    self.warning_audio = pygame.mixer.Sound(path)
                except pygame.error as e:
                    print(f"[Detection] Failed to load '{filename}': {e}")
            else:
                print(f"[Detection] Missing: {filename}")

        def load_chockesinserted_video():
            popup_path = os.path.join(resources_path, "chocks_inserted_popup.mp4")
            if os.path.exists(popup_path):
                self.chocks_inserted_video = popup_path
            else:
                self.chocks_inserted_video = None
                print("[Chocks Inserted] Missing: chocks_inserted_popup.mp4")

        def setup_progressbar():
            num_segments = 9
            bar_width = int(300 * self.scale)
            bar_height = int(15 * self.scale)
            gap = int(1 * self.scale)

            total_width = (num_segments * bar_width // num_segments) + (gap * (num_segments - 1))
            start_x = self.lp_bottom_rect.centerx - total_width // 2
            y = self.lp_bottom_rect.y + int(30 * self.scale)

            segment_width = bar_width // num_segments
            self.progress_rects = []
            for i in range(num_segments):
                rect = pygame.Rect(
                    start_x + i * (segment_width + gap),
                    y,
                    segment_width,
                    bar_height
                )
                self.progress_rects.append(rect)

            font_size = int(18 * self.scale)
            self.progress_font = pygame.font.SysFont("Franklin Gothic Medium Condensed", font_size)
            self.progress_text_pos = (self.progress_rects[0].x + int(42 * self.scale), y - int(8 * self.scale))

        def load_bookends_audio():
            self.bookends_audio = {}
            audio_dir = os.path.join(resources_path, "bookends")
            
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

        setup_layout()
        setup_prediction_text()
        setup_bookend_buttons()
        self.setup_visibility_button()
        load_guide_videos()
        load_detection_audio()
        load_instruction_audio()
        load_warning_audio()
        load_chockesinserted_video()
        load_bookends_audio()
        self.setup_visual_instruction_text(self.instruction)
        setup_progressbar()

    def setup_visibility_button(self):
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
        small_size = int(self.lp_top_rect.height * 0.35)
        big_size = int(self.lp_top_rect.height * 0.75)

        small_font = pygame.font.SysFont("Franklin Gothic Medium Condensed", small_size)
        big_font = pygame.font.SysFont("Franklin Gothic Medium Condensed", big_size)

        title_text = small_font.render("INSTRUCTIONS:", True, (0, 0, 0))
        title_rect = title_text.get_rect()

        instruction_text = big_font.render(instruction.upper().replace("_", " "), True, (0, 0, 0))
        instruction_rect = instruction_text.get_rect()

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

    # Detection & Scoring Utilities
    @staticmethod
    def _clamp(x, lo=0.0, hi=100.0):
        return max(lo, min(hi, x))

    def _mark_prompt(self):
        self.t_prompt = time.perf_counter()
        self.accept_counter = 0
        self.accepted_for_action = False
        self.signal_detected = False

    def _maybe_accept_current(self):
        if not self.training_started:
            return
        if self.current_action >= len(self.actions):
            return

        required = self.actions[self.current_action]
        current_time = time.perf_counter()

        # Correct signal resets everything
        if self.signal == required and self.confidence >= self.THRESHOLD:
            self.accept_counter += 1
            self.last_wrong_time = None
            self.first_warning_played = False
            self.waiting_for_interval = False
            self.audio_end_time = None

        else:
            self.accept_counter = 0

            # Start wrong streak
            if self.last_wrong_time is None:
                self.last_wrong_time = current_time
                self.first_warning_played = False
                self.waiting_for_interval = False
                self.audio_end_time = None

            wrong_duration = current_time - self.last_wrong_time

            # First warning after 5s 
            if not self.first_warning_played and wrong_duration >= 5.0:
                self.play_warning_audio()
                self.first_warning_played = True
                self.waiting_for_interval = True
                self.audio_end_time = None  # we’ll detect when it stops

            # Subsequent warnings 
            elif self.first_warning_played:
                if self.waiting_for_interval:
                    # Wait until audio finishes before starting interval
                    if not pygame.mixer.get_busy() and self.audio_end_time is None:
                        self.audio_end_time = current_time  # mark when it actually stopped
                else:
                    # Count interval after audio finished
                    interval = 5.0  
                    if current_time - self.audio_end_time >= interval:
                        self.play_warning_audio()
                        self.waiting_for_interval = True
                        self.audio_end_time = None

                # Once audio is done, allow counting again
                if self.audio_end_time is not None:
                    self.waiting_for_interval = False

        # Normal acceptance flow
        if (not self.accepted_for_action) and (self.accept_counter >= self.ACCEPT_N):
            t_correct = current_time
            if self.t_prompt is None:
                self.t_prompt = t_correct
            time_to_correct = max(0.0, t_correct - self.t_prompt)
            time_to_correct = min(time_to_correct, self.TMAX)
            score = 100.0 - (self.PENALTY_RATE * time_to_correct)
            score = self._clamp(score)
            label = pretty_label(required)
            self.scores[label] = score

            self.accepted_for_action = True
            self.signal_detected = True

            if required == "chocks_inserted":
                self.play_chocksinserted_video(get_pygame_window_pos())

            self.play_detection_audio()

    # Update / Inference
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

        # Heuristic overrides for All Clear and Set Brakes (sets 100% confidence)
        signal = ""
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

        # Model prediction if no heuristic match
        if signal == "" and len(self.sequence) == self.SEQUENCE_LENGTH:
            input_seq = np.expand_dims(np.array(self.sequence), axis=0)  # shape: (1, 90, 99)
            probs = self.model.predict(input_seq, verbose=0)[0]
            max_idx = np.argmax(probs)
            self.confidence = float(probs[max_idx])
            if self.confidence > self.THRESHOLD:
                self.signal = self.ACTIONS[max_idx]
                self.update_prediction_text()

        # Open the detection window only after instruction audio ends
        if self.training_started and (not pygame.mixer.get_busy()) and (not self.signal_detected):
            if self.t_prompt is None:
                self.t_prompt = time.perf_counter()
                self.accept_counter = 0
                self.accepted_for_action = False

            self._maybe_accept_current()

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

        formatted_signal = self.signal.replace("_", " ").upper()
        value_left = font_big.render(formatted_signal, True, (0, 0, 0))
        value_right = font_big.render(f"{self.confidence * 100:.0f}%", True, (0, 0, 0))

        self.predicted_label_surface = value_left
        self.predicted_probability_surface = value_right

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
                for i, rect in enumerate(self.progress_rects):
                    color = (0, 200, 0) if i < self.current_action else (180, 180, 180)  # green if lit
                    pygame.draw.rect(win, color, rect)

                percentage = round((self.current_action / len(self.progress_rects)) * 100)
                text_surface = self.progress_font.render(f"Progress: {percentage}%", True, (0, 0, 0))
                text_rect = text_surface.get_rect(center=self.progress_text_pos)
                win.blit(text_surface, text_rect)

                if self.current_action == 2 or self.current_action == 3:
                    win.blit(self.pilots_pov_text, self.pilots_pov_rect)

            scaled_frame = pygame.transform.smoothscale(self.frame_surface, self.frame_draw_size)
            win.blit(scaled_frame, self.frame_draw_pos)

            pygame.draw.rect(win, (192, 192, 192), self.rp_top_rect)
            pygame.draw.rect(win, (119, 163, 200), self.lp_top_rect)

            for surface, pos in self.prediction_text_surfaces:
                win.blit(surface, pos)
            win.blit(self.visinstr_title_text, self.visinstr_title_pos)
            win.blit(self.visinstr_instr_text, self.visinstr_instr_pos)

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

        if action not in self.guide_frame_counters:
            self.guide_frame_counters[action] = 0
            self.guide_frame_timers[action] = 0

        idx = self.guide_frame_counters[action]

        screen.blit(frames[idx], pos)

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
            snd = self.instruction_audio[action]
            snd.play()
            self.t_prompt = None
            try:
                self.t_prompt_end = time.perf_counter() + float(snd.get_length())
            except Exception:
                self.t_prompt_end = None
        else:
            print(f"No instruction audio loaded for action '{action}'")

    def play_bookends_audio(self, name):
        if name in self.bookends_audio:
            pygame.mixer.stop()
            self.bookends_audio[name].play()
        else:
            print(f"No bookend audio loaded for name '{name}'")

    def play_warning_audio(self):
        pygame.mixer.stop()
        try:
            self.warning_audio.play()
        except pygame.error:
            print(f"No detection audio loaded for 'warning'")

    def stop_current_audio(self):
        pygame.mixer.stop()

    # Videos
    def play_introduction_video(self):
        self.play_bookends_audio("introduction")

        intro_path = getattr(self, "introvid_path", None)
        if not intro_path or not os.path.exists(intro_path):
            intro_path = os.path.join(resources_path, "bookends", "introduction.mp4")
            if not os.path.exists(intro_path):
                print("[Bookends] introduction.mp4 not found.")
                return

        cap = cv2.VideoCapture(intro_path)
        if not cap.isOpened():
            print("[Bookends] Failed to open introduction.mp4")
            return

        fps = 30
        clock = pygame.time.Clock()

        playing = True
        while playing:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    cap.release()
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.VIDEORESIZE:
                    new_width = max(640, event.w)
                    new_height = max(360, event.h)
                    new_size = (new_width, new_height)

                    pygame.display.set_mode((new_width, new_height), pygame.RESIZABLE)

                    game.win_size = new_size
                    game.init_scale(new_size)
                    game.init_opencv(new_size)
                    game.init_panels(new_size)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        game.stop_current_audio()
                        playing = False
                        
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame, 1)
            frame = np.rot90(frame)
            surface = pygame.surfarray.make_surface(frame)

            win_size = win.get_size()
            scaled = pygame.transform.smoothscale(surface, win_size)
            win.blit(scaled, (0, 0))
            pygame.display.update()

            clock.tick(fps)

        cap.release()
        self.introvid_playing = False

    def play_chocksinserted_video(self, pygamewin_pos):
        if not getattr(self, "chocks_inserted_video", None):
            print("[Chocks Inserted] No video loaded.")
            return

        cap = cv2.VideoCapture(self.chocks_inserted_video)
        if not cap.isOpened():
            print(f"[Chocks Inserted] Failed to open video: {self.chocks_inserted_video}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = int(fps) if fps and fps > 0 else 30
        delay = int(1000 / fps)
        
        vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        aspect_ratio = vid_width / vid_height if vid_height != 0 else 1.0

        win_width, win_height = win.get_size()
        scaled_height = win_height
        scaled_width = int(scaled_height * aspect_ratio)

        if scaled_width > win_width:
            scaled_width = win_width
            scaled_height = int(scaled_width / aspect_ratio)

        cv2.namedWindow("Chocks Inserted", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Chocks Inserted", scaled_width, scaled_height)

        x = max(0, pygamewin_pos[0] - scaled_width - 5)
        y = pygamewin_pos[1]
        cv2.moveWindow("Chocks Inserted", x, y)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow("Chocks Inserted", frame)
            if cv2.waitKey(delay) == -1:
                pass

        cap.release()
        cv2.destroyWindow("Chocks Inserted")

    # Buttons
    def button_over_detection(self, mouse_pos):
        for button in self.buttons.values():
            button[0] = button[4].collidepoint(mouse_pos)

        self.visibilitybtn_over_detection(mouse_pos)

    def visibilitybtn_over_detection(self, mouse_pos):
        button = self.visibility_button
        button[0] = button[4].collidepoint(mouse_pos)

    def button_down_detection(self, mouse_pos):
        for label, (_, is_open, *_, btn_rect) in self.buttons.items():
            if is_open and btn_rect.collidepoint(mouse_pos):
                return label
            
        self.visibilitybtn_down_detection(mouse_pos)

    def visibilitybtn_down_detection(self, mouse_pos):
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
        self.measure_button_dimensions()
        self.calculate_popup_rect()
        self.prepare_buttons()

    def init_scale(self, win_size):
        base_width, base_height = 640, 360
        scale_w = win_size[0] / base_width
        scale_h = win_size[1] / base_height
        self.scale = min(scale_w, scale_h)

    def prepare_text(self):
        self.text_surfaces = []

        ordered_labels = [
            "Start Engine", "Straight Ahead", "Turn Left", "Turn Right", "Stop", "Set Brakes", "Chocks Inserted", "Cut Engines", "All Clear"
        ]
        total_score = 0
        count = 0

        # Render rows in consistent order
        formatted_scores = {}
        for label in ordered_labels:
            score = float(self.signal_scores[label])
            total_score += score
            count += 1

            formatted_score = int(round(score))
            formatted_scores[label] = formatted_score

            text = f"{label}: {formatted_score}%"
            surface = self.font.render(text, True, (0, 0, 0))
            self.text_surfaces.append(surface)

        overall_pct, status, color = get_score(count, total_score)
        save_scores_to_csv(formatted_scores, overall_pct, status)

        self.overall_pct = overall_pct
        self.overall_surface = self.title_font.render(
            f"OVERALL SCORE: {self.overall_pct}%", True, (0, 0, 0)
        )
        self.status_surface = self.font.render(
            f"Status: {status}", True, color
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

        spacing = int(28 * self.scale)
        extra_spacing = int(2 * self.scale)

        cursor_y = self.popup_rect.y + int(10 * self.scale)

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


class RealTime:
    ACTIONS = [
        "chocks_inserted", "cut_engine", "start_engine", "stop",
        "straight_ahead", "turn_left", "turn_right"
    ]
    MODEL_PATH = resource_path("model.h5")
    if getattr(sys, 'frozen', False):
        exe_model = os.path.join(os.path.dirname(sys.executable), "model.h5")
        if os.path.exists(exe_model):
            MODEL_PATH = exe_model
    SEQUENCE_LENGTH = 90
    THRESHOLD       = 0.4

    # Scoring parameters (lead's formula)
    PENALTY_RATE    = 5.0   # percent per second
    TMAX            = 20.0  # seconds; cap for TimeToCorrectError; TimetocorrectMAX
    ACCEPT_N        = 5     # consecutive frames required to accept a detection

    def __init__(self, win_size):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils
        if not self.cap.isOpened():
            print("Could not open camera.")
            pygame.quit()
            sys.exit()

        self.realtime_started = False
        self.visibility_toggle = "X"
        self.actions = [
            "start_engine", "straight_ahead", "turn_left", "turn_right",
            "stop", "set_brakes", "chocks_inserted", "cut_engine", "all_clear"
        ]

        self.init_scale(win_size)
        self.init_opencv(win_size)
        self.init_panel(win_size)
        self.init_prediction_text()
        self.init_buttons()
        self.init_visibility_button()

    def init_scale(self, win_size):
        base_width, base_height = 640, 360
        scale_w = win_size[0] / base_width
        scale_h = win_size[1] / base_height
        self.scale = min(scale_w, scale_h)

    def init_opencv(self, win_size):
        # Dummy frame to calculate aspect ratio
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to read dummy frame for init.")
            pygame.quit()
            sys.exit()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        dummy_surface = pygame.surfarray.make_surface(frame)

        cam_rect = dummy_surface.get_rect()
        max_width = win_size[0] // 1.75
        max_height = win_size[1]

        self.model = load_model(self.MODEL_PATH)
        self.sequence = []
        self.signal = "NONE"
        self.confidence = 0.0

        scale = min(max_width / cam_rect.width, max_height / cam_rect.height)
        new_size = (int(cam_rect.width * scale), int(cam_rect.height * scale))
        self.frame_draw_size = new_size

        pos_x = (win_size[0] - new_size[0]) // 2
        pos_y = (win_size[1] - new_size[1]) // 2
        self.frame_draw_pos = (pos_x, pos_y)

    def init_panel(self, win_size):
        background = pygame.image.load(f"{resources_path}/background.png")
        self.background = pygame.transform.scale(background, win_size)

        self.top_panel = pygame.Rect(
            self.frame_draw_pos[0], 0, 
            win_size[0] // 1.75, self.frame_draw_pos[1])
        self.bottom_panel = pygame.Rect(
            self.frame_draw_pos[0], win_size[1] - self.frame_draw_pos[1], 
            win_size[0] // 1.75, self.frame_draw_pos[1])

    def init_prediction_text(self):
        small_size = int(self.top_panel.height * 0.45)
        big_size = int(self.top_panel.height * 0.7875)
        font_small = pygame.font.SysFont("Franklin Gothic Medium Condensed", small_size)
        font_big = pygame.font.SysFont("Franklin Gothic Medium Condensed", big_size)

        label_left = font_small.render("SIGNAL PREDICTION:", True, (0, 0, 0))
        value_left = font_big.render(self.signal, True, (0, 0, 0))
        label_right = font_small.render("PROBABILITY:", True, (0, 0, 0))
        value_right = font_big.render(f"{self.confidence * 100:.0f}%", True, (0, 0, 0))

        label_left_rect = label_left.get_rect()
        value_left_rect = value_left.get_rect()
        label_right_rect = label_right.get_rect()
        value_right_rect = value_right.get_rect()

        spacing = int(2 * self.scale)
        row_height = label_left_rect.height + value_left_rect.height + spacing
        y_start = self.top_panel.centery - row_height // 2

        margin = int(15 * self.scale)
        left_x = self.top_panel.left + margin
        right_x = self.top_panel.right - margin

        self.prediction_text_surfaces = [
            (label_left, (left_x, y_start)),
            (value_left, (left_x, y_start + label_left_rect.height + spacing)),
            (label_right, (right_x - label_right_rect.width, y_start)),
            (value_right, (right_x - value_right_rect.width, y_start + label_right_rect.height + spacing)),
        ]

        self.predicted_label_surface = value_left
        self.predicted_probability_surface = value_right

    def init_buttons(self):
        font_size = int(16 * self.scale)
        font = pygame.font.SysFont("Franklin Gothic Medium Condensed", font_size)
        labels = ["BACK TO MENU", "START REAL-TIME"]
        surfaces = [font.render(text, True, (0, 0, 0)) for text in labels]

        padding_w = int(25 * self.scale)
        padding_h = int(20 * self.scale)
        rects = [s.get_rect() for s in surfaces]
        width = max(r.width for r in rects) + padding_w
        height = max(r.height for r in rects) + padding_h
        y = self.bottom_panel.y + ((self.bottom_panel.height - height) // 2)
        
        self.buttons = {}
        for i, (label, surface, rect) in enumerate(zip(labels, surfaces, rects)):
            x = self.bottom_panel.left + int(20 * self.scale) if i == 0 else self.bottom_panel.right - width - int(20 * self.scale)
            btn_rect = pygame.Rect(x, y, width, height)
            text_pos = (x + (width - rect.width) // 2, y + (height - rect.height) // 2)
            self.buttons[label] = [False, True, surface, text_pos, btn_rect]  # is_hovered, is_open, text, text_pos, btn_rect

    def init_visibility_button(self):
        font_size = int(24 * self.scale)
        font = pygame.font.SysFont("Franklin Gothic Medium Condensed", font_size)
        surface = font.render(self.visibility_toggle, True, (0, 0, 0))

        padding_w = int(25 * self.scale)
        padding_h = int(14 * self.scale)
        rect = surface.get_rect()
        width = rect.width + padding_w
        height = rect.height + padding_h
        y = self.bottom_panel.y + ((self.bottom_panel.height - height) // 2)

        x = self.bottom_panel.centerx - width // 2
        btn_rect = pygame.Rect(x, y, width, height)
        text_pos = (x + (width - rect.width) // 2, y + (height - rect.height) // 2)
        self.visibility_button = [False, True, surface, text_pos, btn_rect]

    # Draw
    def draw(self, win):
        win.blit(self.background, (0, 0))
        border_width = max(1, round(2 * self.scale))

        pygame.draw.rect(win, (192, 192, 192), self.top_panel)
        # pygame.draw.rect(win, (0, 0, 0), self.bottom_panel)

        if self.frame_surface:
            scaled_frame = pygame.transform.smoothscale(self.frame_surface, self.frame_draw_size)
            win.blit(scaled_frame, self.frame_draw_pos)

            for surface, pos in self.prediction_text_surfaces:
                win.blit(surface, pos)

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

    # Update / Interface
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

        # Heuristic overrides for All Clear and Set Brakes (sets 100% confidence)
        signal = ""
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

        # Model prediction if no heuristic match
        if signal == "" and len(self.sequence) == self.SEQUENCE_LENGTH:
            input_seq = np.expand_dims(np.array(self.sequence), axis=0)  # shape: (1, 90, 99)
            probs = self.model.predict(input_seq, verbose=0)[0]
            max_idx = np.argmax(probs)
            self.confidence = float(probs[max_idx])
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
        big_size = int(self.top_panel.height * 0.7875)
        font_big = pygame.font.SysFont("Franklin Gothic Medium Condensed", big_size)

        formatted_signal = self.signal.replace("_", " ").upper()
        value_left = font_big.render(formatted_signal, True, (0, 0, 0))
        value_right = font_big.render(f"{self.confidence * 100:.0f}%", True, (0, 0, 0))

        self.predicted_label_surface = value_left
        self.predicted_probability_surface = value_right

        spacing = int(2 * self.scale)
        small_size = int(self.top_panel.height * 0.45)
        font_small = pygame.font.SysFont("Franklin Gothic Medium Condensed", small_size)
        label_left_rect = font_small.render("SIGNAL PREDICTION:", True, (0, 0, 0)).get_rect()
        label_right_rect = font_small.render("PROBABILITY:", True, (0, 0, 0)).get_rect()
        row_height = label_left_rect.height + value_left.get_rect().height + spacing
        y_start = self.top_panel.centery - row_height // 2
        margin = int(15 * self.scale)
        left_x = self.top_panel.left + margin
        right_x = self.top_panel.right - margin

        self.prediction_text_surfaces[1] = (value_left, (left_x, y_start + label_left_rect.height + spacing))
        self.prediction_text_surfaces[3] = (value_right, (right_x - value_right.get_width(), y_start + label_right_rect.height + spacing))

    # Buttons
    def button_over_detection(self, mouse_pos):
        for button in self.buttons.values():
            button[0] = button[4].collidepoint(mouse_pos)

        self.visibilitybtn_over_detection(mouse_pos)

    def visibilitybtn_over_detection(self, mouse_pos):
        button = self.visibility_button
        button[0] = button[4].collidepoint(mouse_pos)

    def button_down_detection(self, mouse_pos):
        for label, (_, is_open, *_, btn_rect) in self.buttons.items():
            if is_open and btn_rect.collidepoint(mouse_pos):
                return label
            
        self.visibilitybtn_down_detection(mouse_pos)

    def visibilitybtn_down_detection(self, mouse_pos):
        _, is_open, *_, btn_rect = self.visibility_button
        if is_open and btn_rect.collidepoint(mouse_pos):
            return "VISIBILITY"


def menu_loop():
    global game, realtime
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
                start_button_state = menu.buttons["TRAINING & ASSESSMENT"][1]  # True if it was open
                start_button_state = menu.buttons["REAL-TIME"][1]  # True if it was open

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
                menu.buttons["TRAINING & ASSESSMENT"][1] = start_button_state
                menu.buttons["REAL-TIME"][1] = start_button_state

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
                elif btn_label == "REAL-TIME":
                    run = False

                    realtime_loop()
                elif btn_label == "TRAINING & ASSESSMENT":
                    run = False

                    game.play_introduction_video()
                    game_loop()

        mouse_pos = pygame.mouse.get_pos()
        menu.menubutton_over_detection(mouse_pos)
        menu.draw(win)

        if menu.game_loading and not menu.game_initialized:
            global cv2, load_model, mp
            import cv2
            from keras._tf_keras.keras.models import load_model
            import mediapipe as mp

            current_winsize = pygame.display.get_surface().get_size()
            game = Game(current_winsize)
            realtime = RealTime(current_winsize)
            menu.game_initialized = True
            menu.game_loading = False
            menu.buttons["TRAINING & ASSESSMENT"][1] = True
            menu.buttons["REAL-TIME"][1] = True

    pygame.quit()
    sys.exit()


def game_loop():
    global gameover

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
                mouse_pos = pygame.mouse.get_pos()

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

                btn_label = game.visibilitybtn_down_detection(mouse_pos)
                if btn_label == "VISIBILITY":
                    game.visibility_toggle = "+" if game.visibility_toggle == "X" else "X"
                    game.setup_visibility_button()

            elif event.type == pygame.KEYDOWN:  # !!!
                if event.key == pygame.K_SPACE:
                    game.stop_current_audio()
                    if not game.button_states["START"] and not game.training_started:
                        game.buttons["START"][1] = True
                        game.buttons["END TRAINING"][1] = False
                        game.button_states["START"] = True
                        game.button_states["END TRAINING"] = False

        if not game.assessment_stage:
            if game.training_started:
                if not pygame.mixer.get_busy():
                    if game.signal_detected:
                        command = command_converter(game.actions[game.current_action])
                        if command:
                            send_command(command)

                        game.current_action += 1
                        game.accepted_for_action = False
                        
                        if game.current_action >= len(game.actions):
                            game.assessment_stage = True
                            game.training_started = False

                            current_winsize = pygame.display.get_surface().get_size()
                            gameover = GameOver(current_winsize, game.scores)

                            run = False
                            gameover_loop()
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
        game.button_over_detection(mouse_pos)

        pygame.display.update()

    game.release()
    pygame.quit()
    sys.exit()


def gameover_loop():
    global game
    # global cv2, load_model, mp
    # import cv2
    # from keras._tf_keras.keras.models import load_model
    # import mediapipe as mp

    current_winsize = pygame.display.get_surface().get_size()
    # game = Game(current_winsize)  # !!!
    # game.scores = {'Start Engine': 84.15564000002632, 'Straight Ahead': 32.85263000005216, 'Turn Left': 83.62119249999523, 'Turn Right': 94.3293624999933, 'Stop': 90.04987099993741, 'Set Brakes': 96.57333549999748, 'Chocks Inserted': 87.63132899999619, 'Cut Engines': 85.63987800000177, 'All Clear': 81.64309599997068}
    gameover = GameOver(current_winsize, game.scores)

    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                btn_label = gameover.button_down_detection(mouse_pos)
                if btn_label == "Exit":
                    run = False

                    current_winsize = pygame.display.get_surface().get_size()
                    game.release()

                    game.__init__(current_winsize)
                    game.assessment_stage = False
                    game.training_started = False
                    game.signal_detected = False
                    game.buttons["START"][1] = False
                    game.buttons["END TRAINING"][1] = False
                    game.button_states["START"] = False
                    game.button_states["END TRAINING"] = False
                    game.instruction = "None"
                    game.setup_visual_instruction_text(game.instruction)

                    menu.popup_active = False

                    menu_loop()

                elif btn_label == "Retake Mission":
                    current_winsize = pygame.display.get_surface().get_size()
                    game.release()

                    # Re-init Game
                    game.__init__(current_winsize)
                    game.assessment_stage = False
                    game.training_started = False
                    game.signal_detected = False
                    game.buttons["START"][1] = False
                    game.buttons["END TRAINING"][1] = False
                    game.button_states["START"] = False
                    game.button_states["END TRAINING"] = False
                    game.instruction = "None"
                    game.setup_visual_instruction_text(game.instruction)

                    # Back to game loop
                    run = False

                    game.play_introduction_video()
                    game_loop()

                btn_label = game.visibilitybtn_down_detection(mouse_pos)
                if btn_label == "VISIBILITY":
                    game.visibility_toggle = "+" if game.visibility_toggle == "X" else "X"
                    game.setup_visibility_button()

        # Audio
        if not pygame.mixer.get_busy():
            game.buttons["START"][1] = True
            game.button_states["START"] = True
        
        # Buttons
        mouse_pos = pygame.mouse.get_pos()
        game.buttons["START"][1] = False
        game.buttons["END TRAINING"][1] = False
        game.button_states["START"] = False
        game.button_states["END TRAINING"] = False

        game.visibilitybtn_over_detection(mouse_pos)
        gameover.button_over_detection(mouse_pos)

        # Update frame
        game.update_frame()

        game.draw()
        gameover.draw(win)
        pygame.display.update()

    pygame.quit()
    sys.exit()


def realtime_loop():
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

                # Reinitialize
                realtime.init_scale(new_size)
                realtime.init_opencv(new_size)
                realtime.init_panel(new_size)
                realtime.init_prediction_text()
                realtime.init_buttons()
                realtime.init_visibility_button()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()

                btn_label = realtime.button_down_detection(mouse_pos)
                if btn_label == "BACK TO MENU":
                    run = False

                    menu_loop()
                elif btn_label == "START REAL-TIME":
                    realtime.realtime_started = True

                btn_label = realtime.visibilitybtn_down_detection(mouse_pos)
                if btn_label == "VISIBILITY":
                    realtime.visibility_toggle = "+" if realtime.visibility_toggle == "X" else "X"
                    realtime.init_visibility_button()

        if realtime.realtime_started:
            command = command_converter(realtime.signal)
            if command:
                send_command(command, timeout=0.1)  # timeout can be changed

        realtime.update_frame()
        realtime.draw(win)

        mouse_pos = pygame.mouse.get_pos()
        realtime.button_over_detection(mouse_pos)

        pygame.display.update()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    pygame.init()
    pygame.mixer.init()

    win_size = (640, 360)
    win = pygame.display.set_mode(win_size, pygame.RESIZABLE)
    pygame.display.set_caption("Ground Aircraft Marshalling Simulator")

    # resources_path is defined at module level via resource_path()
    icon_path = os.path.join(resources_path, "icon.png")
    if os.path.exists(icon_path):
        try:
            icon = pygame.image.load(icon_path)
            pygame.display.set_icon(icon)
        except Exception as e:
            print(f"Warning: failed to load icon '{icon_path}': {e}")
    else:
        print(f"Warning: icon not found at {icon_path}")

    ESP8266_IP = "192.168.4.1" 
    base_url = f"http://{ESP8266_IP}"

    menu = Menu(win_size)

    menu_loop()
    # realtime_loop()
