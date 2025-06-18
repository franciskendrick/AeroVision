import pygame
import sys
import os


def init_font(win_size):
    # Scale fonts proportionally to height (feel free to tune the scale factors)
    base_height = 360  # your original reference window height
    scale = win_size[1] / base_height
    garamond_size = int(54 * scale)
    franklin_size = int(112 * scale)
    spacing = int(-5 * scale)  # spacing between lines

    garamond = pygame.font.SysFont("Garamond", garamond_size, bold=True)
    franklingothic = pygame.font.SysFont("Franklin Gothic Medium Condensed", franklin_size, bold=False)

    texts = [
        garamond.render("GROUND", True, (22, 33, 68)),
        garamond.render("AIRCRAFT", True, (22, 33, 68)),
        garamond.render("MARSHALLING", True, (22, 33, 68)),
        franklingothic.render("SIMULATOR", True, (66, 140, 226))
    ]

    # Measure total height of stacked text
    total_height = sum(text.get_rect().height for text in texts) + spacing * (len(texts) - 1)
    start_y = win_size[1] // 2 - total_height // 2

    positions = []
    current_y = start_y
    for text in texts:
        rect = text.get_rect()
        x = win_size[0] // 2 - rect.width // 2
        positions.append((x, current_y))
        current_y += rect.height + spacing

    return texts, positions


def redraw():
    win.blit(background, (0, 0))

    for text, pos in zip(texts, positions):
        win.blit(text, pos)

    pygame.display.update()


def loop():
    global background, texts, positions

    texts, positions = init_font(win_size)

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

                texts, positions = init_font(event.dict["size"])

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

    loop()