from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path("concept_mapping/runs/schematics/ppo_actor_critic_network.png")


def _font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _box(draw, xy, text, fill, outline="#243447", text_fill="#111827", radius=18, font=None):
    x0, y0, x1, y1 = xy
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=2)
    font = font or _font(24, bold=True)
    lines = text.split("\n")
    line_heights = [draw.textbbox((0, 0), line, font=font)[3] for line in lines]
    total_h = sum(line_heights) + (len(lines) - 1) * 8
    y = y0 + ((y1 - y0) - total_h) / 2 - 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        x = x0 + ((x1 - x0) - (bbox[2] - bbox[0])) / 2
        draw.text((x, y), line, font=font, fill=text_fill)
        y += (bbox[3] - bbox[1]) + 8


def _arrow(draw, start, end, color="#334155", width=4):
    draw.line([start, end], fill=color, width=width)
    x0, y0 = start
    x1, y1 = end
    dx, dy = x1 - x0, y1 - y0
    length = max((dx * dx + dy * dy) ** 0.5, 1)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    size = 13
    tip = (x1, y1)
    p1 = (x1 - size * ux + 0.55 * size * px, y1 - size * uy + 0.55 * size * py)
    p2 = (x1 - size * ux - 0.55 * size * px, y1 - size * uy - 0.55 * size * py)
    draw.polygon([tip, p1, p2], fill=color)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    w, h = 1800, 1020
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    title_font = _font(44, bold=True)
    subtitle_font = _font(25)
    label_font = _font(24, bold=True)
    small_font = _font(21)

    draw.text((80, 48), "PPO Actor-Critic Network for Symbolic Craftax", font=title_font, fill="#0F172A")
    draw.text(
        (82, 105),
        "Both actor and critic receive the same 1345-dimensional symbolic observation.",
        font=subtitle_font,
        fill="#475569",
    )

    input_box = (95, 390, 430, 570)
    actor_box = (645, 230, 1105, 420)
    critic_box = (645, 565, 1105, 755)
    logits_box = (1325, 240, 1670, 410)
    value_box = (1325, 575, 1670, 745)

    _box(
        draw,
        input_box,
        "Symbolic state\ns_t ∈ R¹³⁴⁵",
        fill="#E0F2FE",
        outline="#0284C7",
        font=label_font,
    )
    draw.text((105, 592), "7×9×21 map + inventory + health/food/drink/energy + direction + light", font=small_font, fill="#475569")

    _box(
        draw,
        actor_box,
        "Actor MLP\nDense(512) → tanh\nDense(512) → tanh\nDense(512) → tanh",
        fill="#ECFDF3",
        outline="#039855",
        font=small_font,
    )
    _box(
        draw,
        critic_box,
        "Critic MLP\nDense(512) → tanh\nDense(512) → tanh\nDense(512) → tanh",
        fill="#FFF7ED",
        outline="#EA580C",
        font=small_font,
    )
    _box(
        draw,
        logits_box,
        "Actor head\nDense(17)\nlogits → π(a|s)",
        fill="#D1FAE5",
        outline="#047857",
        font=label_font,
    )
    _box(
        draw,
        value_box,
        "Critic head\nDense(1)\nV(s_t)",
        fill="#FFEDD5",
        outline="#C2410C",
        font=label_font,
    )

    _arrow(draw, (430, 470), (645, 325), color="#0F766E")
    _arrow(draw, (430, 490), (645, 660), color="#9A3412")
    _arrow(draw, (1105, 325), (1325, 325), color="#0F766E")
    _arrow(draw, (1105, 660), (1325, 660), color="#9A3412")

    draw.text((515, 290), "same s_t", font=small_font, fill="#0F766E")
    draw.text((515, 675), "same s_t", font=small_font, fill="#9A3412")
    draw.text((1165, 285), "action distribution", font=small_font, fill="#0F766E")
    draw.text((1180, 620), "state value", font=small_font, fill="#9A3412")

    note_box = (96, 820, 1670, 930)
    draw.rounded_rectangle(note_box, radius=16, fill="#F8FAFC", outline="#CBD5E1", width=2)
    draw.text(
        (125, 842),
        "Training signal: PPO uses π(a|s) to choose actions and V(s) to estimate remaining return.",
        font=_font(26, bold=True),
        fill="#111827",
    )
    draw.text(
        (125, 885),
        "Our counterfactual probes edit s_t directly and measure how the critic output V(s_t) changes.",
        font=_font(25),
        fill="#475569",
    )

    img.save(OUT)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
