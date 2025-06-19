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

        self.texts = [
            garamond.render("GROUND", True, (22, 33, 68)),
            garamond.render("AIRCRAFT", True, (22, 33, 68)),
            garamond.render("MARSHALLING", True, (22, 33, 68)),
            franklingothic_big.render("SIMULATOR", True, (66, 140, 226))
        ]

        self.button_texts = [
            franklingothic_small.render("CONNECT TO PROTOTYPE", True, (0, 0, 0)),
            franklingothic_small.render("START", True, (0, 0, 0))
        ]

        # Reference left and right from "MARSHALLING"
        marshalling_rect = self.texts[2].get_rect()
        marshalling_x = win_size[0] // 2 - marshalling_rect.width // 2
        marshalling_right = marshalling_x + marshalling_rect.width

        # Get text and button total height
        text_heights = sum(t.get_rect().height for t in self.texts)
        button_height = max(btn.get_rect().height for btn in self.button_texts)
        total_spacing = self.spacing * (len(self.texts) - 1) - (self.spacing * 6)

        total_height = text_heights + button_height + total_spacing
        start_y = win_size[1] // 2 - total_height // 2

        # TEXT POSITIONS
        self.text_positions = []
        current_y = start_y
        for text in self.texts:
            rect = text.get_rect()
            x = win_size[0] // 2 - rect.width // 2
            self.text_positions.append((x, current_y))
            current_y += rect.height + self.spacing
        else:
            current_y -= self.spacing * 6

        # BUTTON POSITIONS (after SIMULATOR)
        btn1_text = self.button_texts[0]
        btn2_text = self.button_texts[1]
        btn1_rect = btn1_text.get_rect()
        btn2_rect = btn2_text.get_rect()

        # Fixed button height and vertical alignment
        button_y = current_y

        btn1_x = marshalling_x  # align left with MARSHALLING
        btn2_x = marshalling_right - btn2_rect.width  # align right with MARSHALLING

        self.button_text_positions = [
            (btn1_x, button_y),
            (btn2_x, button_y)
        ]

        # Button background rects with padding
        base_padding_w = 25
        base_padding_h = 20
        padding_w = int(base_padding_w * self.scale)
        padding_h = int(base_padding_h * self.scale)

        self.button_rects = [
            pygame.Rect(
                btn1_x - (padding_w // 2),
                button_y - (padding_h // 2),
                btn1_rect.width + padding_w,
                btn1_rect.height + padding_h
            ),
            pygame.Rect(
                btn2_x - (padding_w // 2),
                button_y - (padding_h // 2),
                btn2_rect.width + padding_w,
                btn2_rect.height + padding_h
            )
        ]

    def draw(self, win):
        # Draw texts
        for text, pos in zip(self.texts, self.text_positions):
            win.blit(text, pos)

        # Dynamic border width
        border_width = max(1, round(2 * self.scale))

        # Draw button background rects
        for rect in self.button_rects:
            pygame.draw.rect(win, (255, 255, 255), rect)
            pygame.draw.rect(win, (0, 0, 0), rect, border_width)

        # Draw button texts
        for text, pos in zip(self.button_texts, self.button_text_positions):
            win.blit(text, pos)


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
                background = pygame.image.load(f"{resources_path}/background.png")
                background = pygame.transform.scale(background, event.dict["size"])

                # texts, text_positions, buttons, button_positions, button_rects = init_font(event.dict["size"])
                menu.init_scale(event.dict["size"])
                menu.init(event.dict["size"])

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