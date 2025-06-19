import pygame
import sys
import os


def init_font(win_size):
    # Scale fonts proportionally to height
    base_height = 360
    scale = win_size[1] / base_height
    garamond_size = int(54 * scale)
    franklinbig_size = int(112 * scale)
    franklinsmall_size = int(20 * scale)
    spacing = int(-5 * scale)

    garamond = pygame.font.SysFont("Garamond", garamond_size, bold=True)
    franklingothic_big = pygame.font.SysFont("Franklin Gothic Medium Condensed", franklinbig_size, bold=False)
    franklingothic_small = pygame.font.SysFont("Franklin Gothic Medium Condensed", franklinsmall_size, bold=False)

    # Render header texts
    texts = [
        garamond.render("GROUND", True, (22, 33, 68)),
        garamond.render("AIRCRAFT", True, (22, 33, 68)),
        garamond.render("MARSHALLING", True, (22, 33, 68)),
        franklingothic_big.render("SIMULATOR", True, (66, 140, 226)),
    ]

    # Render button texts
    buttons = [
        franklingothic_small.render("CONNECT TO PROTOTYPE", True, (0, 0, 0)),
        franklingothic_small.render("START", True, (0, 0, 0))
    ]

    # Stack the text positions vertically, centered
    total_height = sum(text.get_rect().height for text in texts) + spacing * (len(texts) - 1)
    start_y = win_size[1] // 2 - total_height // 2

    text_positions = []
    current_y = start_y
    for text in texts:
        rect = text.get_rect()
        x = win_size[0] // 2 - rect.width // 2
        text_positions.append((x, current_y))
        current_y += rect.height + spacing

    # Get SIMULATOR's rect to use as horizontal reference
    simulator_rect = texts[-1].get_rect()
    simulator_x = win_size[0] // 2 - simulator_rect.width // 2
    simulator_right = simulator_x + simulator_rect.width

    # Calculate button vertical position
    button_y = current_y  # current_y is already after "SIMULATOR" + spacing

    # Left button
    btn1_text = buttons[0]
    btn1_rect = btn1_text.get_rect()
    btn1_x = simulator_x
    btn1_rect.topleft = (btn1_x, button_y)

    # Right button
    btn2_text = buttons[1]
    btn2_rect = btn2_text.get_rect()
    btn2_x = simulator_right - btn2_rect.width
    btn2_rect.topleft = (btn2_x, button_y)

    button_rects = [
        btn1_rect.inflate(20, 12),  # Add padding for background rect
        btn2_rect.inflate(20, 12)
    ]

    button_positions = [
        (btn1_rect.x, btn1_rect.y),
        (btn2_rect.x, btn2_rect.y)
    ]

    return texts, text_positions, buttons, button_positions, button_rects


def redraw():
    win.blit(background, (0, 0))

    # Draw main texts
    for text, pos in zip(texts, text_positions):
        win.blit(text, pos)

    # Draw buttons with background rectangles
    for button_rect in button_rects:
        pygame.draw.rect(win, (255, 255, 255), button_rect, border_radius=1)  # white background with rounded corners
        pygame.draw.rect(win, (0, 0, 0), button_rect, 2, border_radius=1)     # black border

    for btn_text, btn_pos in zip(buttons, button_positions):
        win.blit(btn_text, btn_pos)

    pygame.display.update()


def loop():
    global background, texts, text_positions, buttons, button_positions, button_rects

    texts, text_positions, buttons, button_positions, button_rects = init_font(win_size)

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

                texts, text_positions, buttons, button_positions, button_rects = init_font(event.dict["size"])

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