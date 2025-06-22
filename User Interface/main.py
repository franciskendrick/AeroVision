import pygame
import sys
import os


class Menu:
    def __init__(self, win_size):
        self.init_scale(win_size)
        self.init(win_size)

    def init_scale(self, win_size):
        base_height = 360
        self.scale = win_size[1] / base_height

    def init(self, win_size):
        garamond_size = int(54 * self.scale)
        franklinbig_size = int(112 * self.scale)
        franklinsmall_size = int(20 * self.scale)
        spacing = int(-5 * self.scale)

        garamond = pygame.font.SysFont("Garamond", garamond_size, bold=True)
        franklingothic_big = pygame.font.SysFont("Franklin Gothic Medium Condensed", franklinbig_size, bold=False)
        franklingothic_small = pygame.font.SysFont("Franklin Gothic Medium Condensed", franklinsmall_size, bold=False)

        self.spacing = spacing

        # Title
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
            self.texts.append([text, (x, current_y)])  # text, position
            current_y += rect.height + self.spacing
        else:
            current_y -= self.spacing * 6  # add space before buttons

        # Buttons
        button_labels = ["CONNECT TO PROTOTYPE", "START"]
        button_surfaces = [franklingothic_small.render(label, True, (0, 0, 0)) for label in button_labels]

        # Align to MARSHALLING
        marshalling_surface = self.texts[2][0]
        marshalling_rect = marshalling_surface.get_rect()
        marshalling_x = win_size[0] // 2 - marshalling_rect.width // 2
        marshalling_right = marshalling_x + marshalling_rect.width

        base_padding_w, base_padding_h = 25, 20
        padding_w = int(base_padding_w * self.scale)
        padding_h = int(base_padding_h * self.scale)

        self.buttons = []
        for i, surface in enumerate(button_surfaces):
            text_rect = surface.get_rect()
            if i == 0:
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

            # Center text within padded rect
            text_x = button_rect.x + (button_rect.width - text_rect.width) // 2
            text_y = button_rect.y + (button_rect.height - text_rect.height) // 2

            self.buttons.append([False, open_status, surface, (text_x, text_y), button_rect])  # is_hovered, is_open, text, text pos, button rect

    def draw(self, win):
        # Draw text
        for text_surface, pos in self.texts:
            win.blit(text_surface, pos)

        # Draw buttons
        border_width = max(1, round(2 * self.scale))
        for is_hovered, is_open, text, text_pos, btn_rect in self.buttons:
            if is_open:
                fill = (192, 192, 192) if is_hovered else (240, 240, 240)
            else:
                fill = (132, 132, 132)
            
            pygame.draw.rect(win, fill, btn_rect)
            pygame.draw.rect(win, (0, 0, 0), btn_rect, border_width)

            win.blit(text, text_pos)

    def button_down_detection(self, mouse_pos):
        for *_, btn_rect in self.buttons:
            if btn_rect.collidepoint(mouse_pos):
                print(True)

    def button_over_detection(self, mouse_pos):
        for button in self.buttons:
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
                menu.button_down_detection(mouse_pos)

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