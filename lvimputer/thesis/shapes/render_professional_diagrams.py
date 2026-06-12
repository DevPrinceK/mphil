from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "imgs"
FONT_REGULAR = Path("C:/Windows/Fonts/segoeui.ttf")
FONT_BOLD = Path("C:/Windows/Fonts/segoeuib.ttf")

INK = "#17202a"
MUTED = "#566573"
LINE = "#607284"
BG = "#f8fafc"
GROUP_FILL = "#ffffff"
GROUP_STROKE = "#d8e0ea"
SHADOW = "#dce4ee"

PALETTE = {
    "data": ("#edf4fb", "#42647f"),
    "process": ("#ecf6f0", "#3f7a5a"),
    "model": ("#fff4df", "#b37219"),
    "hybrid": ("#f7ecf7", "#7f4b86"),
    "eval": ("#fef0ec", "#b64c39"),
    "runtime": ("#edf0ff", "#5363a6"),
    "decision": ("#fff4df", "#b37219"),
    "output": ("#f0f7f7", "#3f7b84"),
    "hidden": ("#fef0ec", "#b64c39"),
    "observed": ("#ecf6f0", "#3f7a5a"),
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    return ImageFont.truetype(str(path), size=size)


FONTS = {
    "title": font(44, True),
    "stage": font(28, True),
    "body": font(27, False),
    "body_bold": font(27, True),
    "small": font(23, False),
    "small_bold": font(23, True),
}


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int
    text: str
    kind: str = "process"
    radius: int = 22

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    @property
    def left(self) -> tuple[int, int]:
        return (self.x, self.y + self.h // 2)

    @property
    def right(self) -> tuple[int, int]:
        return (self.x + self.w, self.y + self.h // 2)

    @property
    def top(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y)

    @property
    def bottom(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h)


def canvas(width: int, height: int, title: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)
    draw.text((70, 52), title, font=FONTS["title"], fill=INK)
    draw.line((70, 118, width - 70, 118), fill="#d9e1ec", width=3)
    return img, draw


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=fnt)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def wrap_line(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    line = words[0]
    for word in words[1:]:
        candidate = f"{line} {word}"
        if text_size(draw, candidate, fnt)[0] <= max_width:
            line = candidate
        else:
            lines.append(line)
            line = word
    lines.append(line)
    return lines


def wrap_text(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    lines: list[str] = []
    for raw in text.split("\n"):
        lines.extend(wrap_line(draw, raw, fnt, max_width))
    return lines


def draw_wrapped_center(
    draw: ImageDraw.ImageDraw,
    xywh: tuple[int, int, int, int],
    text: str,
    fnt: ImageFont.FreeTypeFont,
    fill: str = INK,
    line_gap: int = 8,
) -> None:
    x, y, w, h = xywh
    lines = wrap_text(draw, text, fnt, w - 46)
    heights = [text_size(draw, line, fnt)[1] for line in lines]
    total_h = sum(heights) + line_gap * (len(lines) - 1)
    yy = y + (h - total_h) // 2 - 2
    for line, lh in zip(lines, heights):
        lw, _ = text_size(draw, line, fnt)
        draw.text((x + (w - lw) // 2, yy), line, font=fnt, fill=fill)
        yy += lh + line_gap


def group(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, label: str) -> None:
    draw.rounded_rectangle((x + 6, y + 8, x + w + 6, y + h + 8), radius=30, fill="#eef3f8")
    draw.rounded_rectangle((x, y, x + w, y + h), radius=30, fill=GROUP_FILL, outline=GROUP_STROKE, width=3)
    draw.text((x + 30, y + 22), label, font=FONTS["stage"], fill=INK)


def node(draw: ImageDraw.ImageDraw, box: Box, fnt: ImageFont.FreeTypeFont | None = None) -> None:
    fnt = fnt or FONTS["body"]
    fill, stroke = PALETTE[box.kind]
    x, y, w, h = box.x, box.y, box.w, box.h
    draw.rounded_rectangle((x + 7, y + 8, x + w + 7, y + h + 8), radius=box.radius, fill=SHADOW)
    draw.rounded_rectangle((x, y, x + w, y + h), radius=box.radius, fill=fill, outline=stroke, width=4)
    draw.rounded_rectangle((x, y, x + 13, y + h), radius=box.radius, fill=stroke)
    draw_wrapped_center(draw, (x + 10, y, w - 10, h), box.text, fnt)


def tag(draw: ImageDraw.ImageDraw, x: int, y: int, text: str) -> None:
    tw, th = text_size(draw, text, FONTS["small_bold"])
    draw.rounded_rectangle((x, y, x + tw + 24, y + th + 18), radius=12, fill=BG)
    draw.text((x + 12, y + 8), text, font=FONTS["small_bold"], fill=MUTED)


def diamond(draw: ImageDraw.ImageDraw, box: Box) -> None:
    fill, stroke = PALETTE[box.kind]
    cx, cy = box.center
    points = [(cx, box.y), (box.x + box.w, cy), (cx, box.y + box.h), (box.x, cy)]
    shadow = [(x + 6, y + 8) for x, y in points]
    draw.polygon(shadow, fill=SHADOW)
    draw.polygon(points, fill=fill, outline=stroke)
    draw.line(points + [points[0]], fill=stroke, width=4)
    draw_wrapped_center(draw, (box.x + 22, box.y + 20, box.w - 44, box.h - 40), box.text, FONTS["small_bold"])


def arrow_head(end: tuple[int, int], angle: float, size: int = 18) -> list[tuple[int, int]]:
    x, y = end
    return [
        (x, y),
        (int(x - size * math.cos(angle - math.pi / 7)), int(y - size * math.sin(angle - math.pi / 7))),
        (int(x - size * math.cos(angle + math.pi / 7)), int(y - size * math.sin(angle + math.pi / 7))),
    ]


def poly_arrow(
    draw: ImageDraw.ImageDraw,
    points: Sequence[tuple[int, int]],
    color: str = LINE,
    width: int = 5,
    label: str | None = None,
    label_offset: tuple[int, int] = (0, -34),
) -> None:
    if len(points) < 2:
        return
    draw.line(points, fill=color, width=width, joint="curve")
    x1, y1 = points[-2]
    x2, y2 = points[-1]
    angle = math.atan2(y2 - y1, x2 - x1)
    draw.polygon(arrow_head(points[-1], angle), fill=color)
    if label:
        mid = points[len(points) // 2]
        lx, ly = mid[0] + label_offset[0], mid[1] + label_offset[1]
        tw, th = text_size(draw, label, FONTS["small_bold"])
        draw.rounded_rectangle((lx - 12, ly - 8, lx + tw + 12, ly + th + 10), radius=12, fill=BG)
        draw.text((lx, ly), label, font=FONTS["small_bold"], fill=MUTED)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], **kwargs) -> None:
    poly_arrow(draw, [start, end], **kwargs)


def save(img: Image.Image, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img.save(OUT_DIR / name, quality=95)


def draw_method_pipeline() -> None:
    img, draw = canvas(2500, 1420, "End-to-end methodology pipeline")
    group(draw, 70, 160, 520, 1030, "Data preparation")
    group(draw, 680, 160, 640, 1030, "Imputation methods")
    group(draw, 1410, 160, 1020, 1030, "Evaluation")

    boxes = {
        "data": Box(130, 275, 410, 120, "Timestamped temperature data\ndata/dataset1.csv", "data"),
        "sort": Box(130, 455, 410, 120, "Chronological sorting\ntemperature channel only", "data"),
        "clean": Box(130, 635, 410, 135, "Training-only robust cleaning\nsentinel and local artefact handling", "process"),
        "mask": Box(130, 830, 410, 135, "Controlled 30% masks\nSTART / MIDDLE / END / RANDOM", "process"),
        "stat": Box(740, 280, 500, 120, "Statistical and time-aware\nbaselines", "model"),
        "ml": Box(740, 460, 500, 120, "Lag-feature machine-learning\nbaselines", "model"),
        "prior": Box(740, 640, 500, 120, "Scenario-specific prior\ninterpolation or seasonal KNN", "model"),
        "hybrid": Box(740, 820, 500, 135, "Prior-guided residual\nLSTM-VAE", "hybrid"),
        "complete": Box(1475, 525, 360, 125, "Completed series", "output"),
        "metric": Box(1940, 285, 400, 120, "Masked-point error\nMAE and RMSE", "eval"),
        "plots": Box(1940, 505, 400, 120, "Qualitative reconstruction\nplots", "eval"),
        "runtime": Box(1940, 725, 400, 120, "Imputation-only runtime\ntraining excluded", "runtime"),
        "compare": Box(1775, 965, 470, 130, "Win rate, scenario results\nand method interpretation", "eval"),
    }
    for b in boxes.values():
        node(draw, b)

    for a, b in [("data", "sort"), ("sort", "clean"), ("clean", "mask")]:
        arrow(draw, boxes[a].bottom, boxes[b].top)
    for target in ["stat", "ml", "prior"]:
        poly_arrow(draw, [boxes["mask"].right, (645, boxes["mask"].center[1]), (645, boxes[target].center[1]), boxes[target].left])
    arrow(draw, boxes["prior"].bottom, boxes["hybrid"].top)
    for src in ["stat", "ml", "hybrid"]:
        poly_arrow(draw, [boxes[src].right, (1370, boxes[src].center[1]), (1370, boxes["complete"].center[1]), boxes["complete"].left])
    for target in ["metric", "plots", "runtime"]:
        poly_arrow(draw, [boxes["complete"].right, (1885, boxes["complete"].center[1]), (1885, boxes[target].center[1]), boxes[target].left])
    for src in ["metric", "plots", "runtime"]:
        poly_arrow(draw, [boxes[src].right, (2375, boxes[src].center[1]), (2375, 920), (boxes["compare"].center[0], 920), boxes["compare"].top])
    save(img, "diagram-method-pipeline.png")


def draw_hybrid_prior() -> None:
    img, draw = canvas(2300, 1180, "Hybrid prior construction")
    boxes = {
        "series": Box(90, 500, 360, 130, "Masked temperature series\nvalues x_t and mask m_t", "data"),
        "scenario": Box(535, 455, 300, 220, "Gap\nscenario", "decision"),
        "interp": Box(940, 260, 400, 130, "Time interpolation prior\nlocal temporal anchors", "process"),
        "knn": Box(940, 660, 400, 130, "Seasonal KNN prior\ncalendar-neighbour matching", "process"),
        "calendar": Box(560, 830, 390, 120, "Cyclic calendar features\nsin/cos hour and day", "data"),
        "trend": Box(560, 985, 390, 100, "Normalised trend feature", "data"),
        "prior": Box(1465, 500, 330, 120, "Prior estimate p_t", "model"),
        "filled": Box(1900, 315, 330, 130, "Filled model input\nmissing values use p_t", "model"),
        "residual": Box(1900, 525, 330, 130, "Residual LSTM-VAE\nlearns correction r_t", "hybrid"),
        "hybrid": Box(1900, 735, 330, 130, "Hybrid estimate\nx_hat_t = p_t + r_t", "output"),
    }
    node(draw, boxes["series"])
    diamond(draw, boxes["scenario"])
    for key in ["interp", "knn", "calendar", "trend", "prior", "filled", "residual", "hybrid"]:
        node(draw, boxes[key])
    arrow(draw, boxes["series"].right, boxes["scenario"].left)
    arrow(draw, (boxes["scenario"].x + boxes["scenario"].w, boxes["scenario"].y + 70), boxes["interp"].left)
    arrow(draw, (boxes["scenario"].x + boxes["scenario"].w, boxes["scenario"].y + 150), boxes["knn"].left)
    tag(draw, 810, 335, "MIDDLE / RANDOM")
    tag(draw, 820, 720, "START / END")
    poly_arrow(draw, [boxes["calendar"].right, (900, boxes["calendar"].center[1]), (900, boxes["knn"].bottom[1]), boxes["knn"].bottom])
    poly_arrow(draw, [boxes["trend"].right, (1015, boxes["trend"].center[1]), (1015, boxes["knn"].bottom[1] + 20), (boxes["knn"].center[0], boxes["knn"].bottom[1] + 20), boxes["knn"].bottom])
    poly_arrow(draw, [boxes["interp"].right, (1405, boxes["interp"].center[1]), (1405, boxes["prior"].center[1]), boxes["prior"].left])
    poly_arrow(draw, [boxes["knn"].right, (1405, boxes["knn"].center[1]), (1405, boxes["prior"].center[1]), boxes["prior"].left])
    poly_arrow(draw, [boxes["prior"].right, (1840, boxes["prior"].center[1]), (1840, boxes["filled"].center[1]), boxes["filled"].left])
    arrow(draw, boxes["filled"].bottom, boxes["residual"].top)
    arrow(draw, boxes["residual"].bottom, boxes["hybrid"].top)
    save(img, "diagram-hybrid-prior-construction.png")


def draw_baseline_taxonomy() -> None:
    img, draw = canvas(2500, 1450, "Baseline method taxonomy")
    top = Box(845, 180, 810, 115, "Observed series with the same synthetic mask", "data")
    family = Box(1035, 350, 430, 105, "Baseline imputers", "model")
    node(draw, top)
    node(draw, family)
    arrow(draw, top.bottom, family.top)

    groups = [
        ("Constant statistics", ["Mean", "Median", "Mode"], "model"),
        ("Sampling based", ["Random donor sampling", "Hot-deck sampling"], "model"),
        ("Time-aware rules", ["Forward fill", "LOCF", "LOCB", "Linear interpolation"], "model"),
        ("Lag-feature machine learning", ["Lag values + mask flags, L = 48", "KNN regression", "Ridge regression", "HistGradientBoosting", "IterativeImputer, Bayesian ridge"], "hybrid"),
    ]
    x_positions = [95, 650, 1205, 1760]
    group_boxes: list[tuple[int, int, int, int]] = []
    for x, (title, items, kind) in zip(x_positions, groups):
        group(draw, x, 560, 480, 610, title)
        group_boxes.append((x, 560, 480, 610))
        y = 665
        for item in items:
            b = Box(x + 55, y, 370, 78, item, "process" if kind == "model" else "hybrid", radius=18)
            node(draw, b, FONTS["small"])
            y += 94
        poly_arrow(draw, [family.bottom, (family.center[0], 510), (x + 240, 510), (x + 240, 560)])

    metric = Box(845, 1260, 810, 105, "Masked-point evaluation: MAE and RMSE", "eval")
    node(draw, metric)
    for x, y, w, h in group_boxes:
        poly_arrow(draw, [(x + w // 2, y + h), (x + w // 2, 1218), (metric.center[0], 1218), metric.top])
    save(img, "diagram-baseline-methods-taxonomy.png")


def draw_missingness_scenarios() -> None:
    img, draw = canvas(2500, 1250, "Synthetic missingness scenarios")
    legend_y = 165
    draw.rounded_rectangle((1500, legend_y, 1760, legend_y + 62), radius=18, fill=PALETTE["observed"][0], outline=PALETTE["observed"][1], width=3)
    draw.text((1785, legend_y + 13), "Observed", font=FONTS["small_bold"], fill=INK)
    draw.rounded_rectangle((1950, legend_y, 2210, legend_y + 62), radius=18, fill=PALETTE["hidden"][0], outline=PALETTE["hidden"][1], width=3)
    draw.text((2230, legend_y + 13), "Hidden", font=FONTS["small_bold"], fill=INK)

    start_x, bar_w, bar_h = 470, 1550, 74
    rows = [
        ("START", [(0.00, 0.30, "hidden"), (0.30, 1.00, "observed")]),
        ("MIDDLE", [(0.00, 0.35, "observed"), (0.35, 0.65, "hidden"), (0.65, 1.00, "observed")]),
        ("END", [(0.00, 0.70, "observed"), (0.70, 1.00, "hidden")]),
    ]
    y = 300
    for label, segments in rows:
        draw.text((120, y + 18), label, font=FONTS["stage"], fill=INK)
        draw.rounded_rectangle((start_x, y, start_x + bar_w, y + bar_h), radius=22, fill="#ffffff", outline="#cbd6e2", width=3)
        for a, b, kind in segments:
            fill, stroke = PALETTE[kind]
            x1 = start_x + int(a * bar_w)
            x2 = start_x + int(b * bar_w)
            draw.rounded_rectangle((x1, y, x2, y + bar_h), radius=20, fill=fill, outline=stroke, width=2)
        draw.text((start_x + bar_w + 40, y + 17), "30% masked", font=FONTS["small_bold"], fill=MUTED)
        y += 190

    draw.text((120, y + 18), "RANDOM", font=FONTS["stage"], fill=INK)
    draw.rounded_rectangle((start_x, y, start_x + bar_w, y + bar_h), radius=22, fill=PALETTE["observed"][0], outline=PALETTE["observed"][1], width=3)
    rng_segments = [
        (0.03, 0.045), (0.09, 0.105), (0.16, 0.175), (0.22, 0.24), (0.29, 0.31),
        (0.36, 0.375), (0.43, 0.455), (0.51, 0.525), (0.58, 0.595), (0.64, 0.66),
        (0.71, 0.725), (0.79, 0.81), (0.87, 0.89), (0.94, 0.955),
    ]
    for a, b in rng_segments:
        x1 = start_x + int(a * bar_w)
        x2 = start_x + int(b * bar_w)
        draw.rounded_rectangle((x1, y, x2, y + bar_h), radius=10, fill=PALETTE["hidden"][0], outline=PALETTE["hidden"][1], width=2)
    draw.text((start_x + bar_w + 40, y + 17), "Fixed seed, 30% hidden", font=FONTS["small_bold"], fill=MUTED)

    source = Box(680, 1030, 930, 105, "Same chronological temperature sequence, N = 26,387", "data")
    node(draw, source)
    save(img, "diagram-missingness-scenarios.png")


def draw_architecture() -> None:
    img, draw = canvas(2700, 1500, "Prior-guided residual LSTM-VAE architecture")
    group(draw, 70, 175, 520, 1080, "Prior and input")
    group(draw, 660, 175, 780, 1080, "LSTM-VAE core")
    group(draw, 1510, 175, 770, 1080, "Residual prediction")
    group(draw, 2335, 175, 295, 1080, "Gate")

    boxes = {
        "prior": Box(140, 290, 380, 120, "Prior sequence p_t\ninterpolation or seasonal KNN", "model"),
        "window": Box(140, 590, 380, 150, "Masked window u_1:T\nfilled value, mask, prior, calendar", "data"),
        "enc": Box(720, 590, 330, 120, "Encoder LSTM\nfinal hidden state", "process"),
        "latent": Box(1120, 555, 330, 190, "VAE latent space\nmu, log sigma^2\nreparameterise to z", "model"),
        "context": Box(1575, 300, 375, 120, "Decoder context\nprior + calendar features", "data"),
        "decoder": Box(1575, 590, 375, 140, "Repeat z, concatenate context\nDecoder LSTM + residual head", "hybrid"),
        "pred": Box(2025, 590, 330, 140, "Hybrid prediction\nx_hat_t = p_t + r_t", "output"),
        "gate": Box(2355, 550, 265, 210, "Validation gate\nimproves MAE\nand RMSE?", "decision"),
        "blend": Box(2210, 910, 220, 105, "Use residual\nblend", "eval"),
        "keep": Box(2480, 910, 180, 105, "Keep prior", "eval"),
        "out": Box(2260, 1135, 360, 120, "Final imputed series\nonly hidden points replaced", "output"),
    }
    for key, b in boxes.items():
        if key == "gate":
            diamond(draw, b)
        else:
            node(draw, b, FONTS["small"] if b.w < 250 else FONTS["body"])

    arrow(draw, boxes["prior"].bottom, boxes["window"].top)
    arrow(draw, boxes["window"].right, boxes["enc"].left)
    arrow(draw, boxes["enc"].right, boxes["latent"].left)
    arrow(draw, boxes["latent"].right, boxes["decoder"].left)
    poly_arrow(draw, [boxes["prior"].right, (1515, boxes["prior"].center[1]), boxes["context"].left])
    arrow(draw, boxes["context"].bottom, boxes["decoder"].top)
    arrow(draw, boxes["decoder"].right, boxes["pred"].left)
    poly_arrow(draw, [boxes["prior"].right, (1985, 255), (1985, boxes["pred"].center[1]), boxes["pred"].left])
    arrow(draw, boxes["pred"].right, boxes["gate"].left)
    poly_arrow(draw, [boxes["gate"].bottom, (boxes["gate"].center[0], 855), boxes["blend"].top], label="Yes", label_offset=(20, -22))
    poly_arrow(draw, [boxes["gate"].bottom, (boxes["gate"].center[0], 850), (boxes["keep"].center[0], 850), boxes["keep"].top], label="No", label_offset=(24, -22))
    arrow(draw, boxes["blend"].bottom, boxes["out"].top)
    arrow(draw, boxes["keep"].bottom, boxes["out"].top)
    save(img, "diagram-residual-lstm-vae-architecture.png")


def draw_training_objective() -> None:
    img, draw = canvas(2350, 1320, "Training objective and optimisation")
    boxes = {
        "windows": Box(150, 255, 380, 120, "Sliding training windows\nT = 96, stride = 16", "data"),
        "forward": Box(710, 255, 420, 120, "Residual LSTM-VAE\nforward pass", "hybrid"),
        "pred": Box(1310, 255, 330, 105, "Prediction x_hat_t", "output"),
        "rec": Box(410, 590, 420, 125, "Reconstruction loss\nSmooth L1 on observed points", "eval"),
        "kl": Box(965, 590, 420, 125, "KL divergence\nposterior to standard normal", "eval"),
        "res": Box(1520, 590, 420, 125, "Residual penalty\nlimits unnecessary changes", "eval"),
        "beta": Box(965, 810, 420, 95, "Beta warm-up", "model"),
        "total": Box(850, 1010, 650, 120, "Total objective\nL_rec + beta L_KL + lambda L_res", "model"),
        "adam": Box(1670, 1005, 280, 110, "AdamW", "hybrid"),
        "sched": Box(2020, 880, 260, 105, "LR scheduler", "runtime"),
        "clip": Box(2020, 1050, 260, 105, "Gradient clipping", "runtime"),
        "params": Box(1670, 1200, 610, 95, "Updated model parameters", "output"),
    }
    for b in boxes.values():
        node(draw, b)
    arrow(draw, boxes["windows"].right, boxes["forward"].left)
    arrow(draw, boxes["forward"].right, boxes["pred"].left)
    for key in ["rec", "kl", "res"]:
        poly_arrow(draw, [boxes["pred"].bottom, (boxes["pred"].center[0], 505), (boxes[key].center[0], 505), boxes[key].top])
    arrow(draw, boxes["beta"].top, boxes["kl"].bottom)
    for key in ["rec", "kl", "res"]:
        poly_arrow(draw, [boxes[key].bottom, (boxes[key].center[0], 955), (boxes["total"].center[0], 955), boxes["total"].top])
    arrow(draw, boxes["total"].right, boxes["adam"].left)
    arrow(draw, boxes["sched"].left, boxes["adam"].right)
    arrow(draw, boxes["clip"].left, boxes["adam"].right)
    arrow(draw, boxes["adam"].bottom, boxes["params"].top)
    save(img, "diagram-training-objective.png")


def draw_windowed_stitching() -> None:
    img, draw = canvas(2500, 1360, "Windowed reconstruction and overlap averaging")
    full = Box(120, 270, 410, 120, "Full masked sequence", "data")
    prior = Box(120, 510, 410, 120, "Scenario-specific prior", "model")
    node(draw, full)
    node(draw, prior)
    arrow(draw, full.bottom, prior.top)

    group(draw, 660, 205, 770, 640, "Overlapping windows")
    bar_x, bar_y, bar_w, bar_h = 725, 335, 640, 72
    draw.rounded_rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + bar_h), radius=20, fill=PALETTE["data"][0], outline=PALETTE["data"][1], width=3)
    draw.text((bar_x + 120, bar_y - 45), "length 96, evaluation stride 48", font=FONTS["small_bold"], fill=MUTED)
    colors = [PALETTE["process"], PALETTE["hybrid"], PALETTE["model"]]
    offsets = [0, 145, 290]
    for i, off in enumerate(offsets):
        fill, stroke = colors[i]
        y = 470 + i * 95
        draw.rounded_rectangle((bar_x + off, y, bar_x + off + 300, y + 68), radius=18, fill=fill, outline=stroke, width=3)
        draw.text((bar_x + off + 45, y + 18), f"Window {i + 1} prediction", font=FONTS["small"], fill=INK)
    arrow(draw, prior.right, (660, 525))

    boxes = {
        "collect": Box(1565, 350, 350, 120, "Collect predictions\nby timestamp", "process"),
        "avg": Box(1565, 565, 350, 120, "Overlap average\none value per timestamp", "process"),
        "gate": Box(2030, 500, 300, 190, "Residual gate\nimproves MAE\nand RMSE?", "decision"),
        "blend": Box(1875, 815, 250, 105, "Blend prior and\nresidual prediction", "eval"),
        "keep": Box(2165, 815, 250, 105, "Keep prior\nunchanged", "eval"),
        "final": Box(1950, 1075, 420, 125, "Final imputed series\nobserved raw values retained", "output"),
    }
    for key, b in boxes.items():
        if key == "gate":
            diamond(draw, b)
        else:
            node(draw, b)
    poly_arrow(draw, [(1430, 525), boxes["collect"].left])
    arrow(draw, boxes["collect"].bottom, boxes["avg"].top)
    arrow(draw, boxes["avg"].right, boxes["gate"].left)
    poly_arrow(draw, [boxes["gate"].bottom, (boxes["gate"].center[0], 760), boxes["blend"].top], label="Yes", label_offset=(20, -22))
    poly_arrow(draw, [boxes["gate"].bottom, (boxes["gate"].center[0], 760), (boxes["keep"].center[0], 760), boxes["keep"].top], label="No", label_offset=(24, -22))
    arrow(draw, boxes["blend"].bottom, boxes["final"].top)
    arrow(draw, boxes["keep"].bottom, boxes["final"].top)
    save(img, "diagram-windowed-reconstruction-stitching.png")


def draw_evaluation_runtime() -> None:
    img, draw = canvas(2350, 1350, "Evaluation and imputation-only runtime protocol")
    group(draw, 70, 170, 1420, 1050, "Masked-point accuracy")
    group(draw, 1570, 170, 700, 1050, "Runtime comparison")
    boxes = {
        "raw": Box(150, 270, 380, 115, "Raw temperature sequence\nground truth retained", "data"),
        "mask": Box(150, 470, 380, 115, "Apply synthetic\n30% mask", "process"),
        "obs": Box(150, 670, 380, 115, "Observed input\nseries", "data"),
        "hidden": Box(150, 870, 380, 115, "Hidden index set M\nwhere m_t = 0", "hidden"),
        "base": Box(680, 560, 310, 110, "Baseline\nimputers", "model"),
        "hybrid": Box(680, 760, 310, 125, "Prior-guided residual\nLSTM-VAE", "hybrid"),
        "est": Box(1110, 660, 300, 125, "Estimates at\nhidden indices", "output"),
        "score": Box(1110, 890, 300, 115, "Score only\nhidden points", "eval"),
        "metrics": Box(1110, 1090, 300, 95, "MAE and RMSE", "eval"),
        "methods": Box(1650, 430, 540, 130, "Run each imputer under\nthe same mask", "model"),
        "timer": Box(1650, 685, 540, 130, "Measure imputation step only\ntraining excluded", "runtime"),
        "cost": Box(1650, 940, 540, 130, "Deployment-cost comparison\nby method and scenario", "runtime"),
    }
    for b in boxes.values():
        node(draw, b)
    arrow(draw, boxes["raw"].bottom, boxes["mask"].top)
    arrow(draw, boxes["mask"].bottom, boxes["obs"].top)
    arrow(draw, boxes["obs"].bottom, boxes["hidden"].top)
    poly_arrow(draw, [boxes["obs"].right, (610, boxes["obs"].center[1]), (610, boxes["base"].center[1]), boxes["base"].left])
    poly_arrow(draw, [boxes["obs"].right, (610, boxes["obs"].center[1]), (610, boxes["hybrid"].center[1]), boxes["hybrid"].left])
    arrow(draw, boxes["base"].right, boxes["est"].left)
    arrow(draw, boxes["hybrid"].right, boxes["est"].left)
    arrow(draw, boxes["est"].bottom, boxes["score"].top)
    poly_arrow(draw, [boxes["hidden"].right, (1060, boxes["hidden"].center[1]), boxes["score"].left])
    arrow(draw, boxes["score"].bottom, boxes["metrics"].top)
    arrow(draw, boxes["methods"].bottom, boxes["timer"].top)
    arrow(draw, boxes["timer"].bottom, boxes["cost"].top)
    save(img, "diagram-evaluation-runtime-protocol.png")


def main() -> None:
    draw_method_pipeline()
    draw_missingness_scenarios()
    draw_baseline_taxonomy()
    draw_hybrid_prior()
    draw_architecture()
    draw_training_objective()
    draw_windowed_stitching()
    draw_evaluation_runtime()
    print(f"Wrote professional diagram PNGs to {OUT_DIR}")


if __name__ == "__main__":
    main()
