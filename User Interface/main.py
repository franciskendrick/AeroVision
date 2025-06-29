import numpy as np
import pygame
import cv2
import sys
import os


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
    def __init__(self, win_size):
        import mediapipe as mp

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

        # ── VISUAL INSTRUCTIONS (left panel) ──
        vis_font_size = int(self.lp_top_rect.height * 0.55)
        vis_font = pygame.font.SysFont("Franklin Gothic Medium Condensed", vis_font_size, bold=False)

        self.visinstr_text = vis_font.render("VISUAL INSTRUCTIONS", True, (0, 0, 0))
        vis_rect = self.visinstr_text.get_rect()
        self.visinstr_pos = (
            self.lp_top_rect.centerx - vis_rect.width // 2,
            self.lp_top_rect.centery - vis_rect.height // 2
        )

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

    def draw(self, win):
        win.fill((0, 0, 0)) 

        if self.frame_surface:
            scaled_frame = pygame.transform.smoothscale(self.frame_surface, self.frame_draw_size)
            win.blit(scaled_frame, self.frame_draw_pos)

            pygame.draw.rect(win, (192, 192, 192), self.rp_top_rect)
            pygame.draw.rect(win, (192, 192, 192), self.rp_bottom_rect)
            pygame.draw.rect(win, (119, 163, 200), self.lp_top_rect)

            for surface, pos in self.prediction_text_surfaces:
                win.blit(surface, pos)
            win.blit(self.visinstr_text, self.visinstr_pos)

        pygame.display.update()

    def release(self):
        self.cap.release()


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

        game.update_frame()
        game.draw(win)

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
