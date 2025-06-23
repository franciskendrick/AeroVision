import pygame
import sys
import os


class Menu: 
    def __init__(self, win_size):
        self.popup_active = False
        self.init_scale(win_size)
        self.init_menu(win_size)
        self.init_popup(win_size)

    def init_scale(self, win_size):
        base_height = 360
        self.scale = win_size[1] / base_height

    def init_menu(self, win_size):
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

        win_w, _ = win_size
        *_, btn_pos, _ = list(self.buttons.values())[0]

        titlebar_height = int(22 * self.scale)
        popup_total_height = self.popup_height + titlebar_height
        vertical_center = btn_pos[1] // 2 - popup_total_height // 2

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

        self.pop_up = [popup_titlebar_rect, popup_rect, popup_closebutton_text, popup_closebutton_rect, popup_closebutton_text_pos]

    def draw(self, win):
        border_width = max(1, round(2 * self.scale))

        for is_hovered, is_open, text, text_pos, btn_rect in self.buttons.values():
            fill = (192, 192, 192) if is_hovered and is_open else \
                   (240, 240, 240) if is_open else (132, 132, 132)
            pygame.draw.rect(win, fill, btn_rect)
            pygame.draw.rect(win, (0, 0, 0), btn_rect, border_width)
            win.blit(text, text_pos)

        if not self.popup_active:
            for text_surface, pos in self.texts:
                win.blit(text_surface, pos)
            return

        popup_titlebar_rect, popup_rect, popup_closebutton_text, popup_closebutton_rect, popup_closebutton_text_pos = self.pop_up
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

    def button_down_detection(self, mouse_pos):
        for label, (_, is_open, *_, btn_rect) in self.buttons.items():
            if is_open and btn_rect.collidepoint(mouse_pos):
                return label

    def button_over_detection(self, mouse_pos):
        for button in self.buttons.values():
            button[0] = button[4].collidepoint(mouse_pos)


def redraw():
    win.blit(background, (0, 0))

    # Draw main texts
    menu.draw(win)

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
                # Enforce minimum size of 640x360
                new_width = max(640, event.w)
                new_height = max(360, event.h)
                new_size = (new_width, new_height)

                pygame.display.set_mode(new_size, pygame.RESIZABLE)

                background = pygame.image.load(f"{resources_path}/background.png")
                background = pygame.transform.scale(background, new_size)

                menu.init_scale(new_size)
                menu.init(new_size)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                btn_label = menu.button_down_detection(mouse_pos)
                if btn_label == "CONNECT TO PROTOTYPE":
                    menu.popup_active = True
                    menu.buttons["START"][1] = True
                else:  # START
                    pass

        mouse_pos = pygame.mouse.get_pos()
        menu.button_over_detection(mouse_pos)

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

    menu = Menu(win_size)

    loop()