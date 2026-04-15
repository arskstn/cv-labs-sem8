import numpy as np
import cv2
import pickle
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from pathlib import Path

CONFIGS = [
    {"name": "thresh=1.2 step=16 scale=0.85 iou=0.30", "threshold": 1.2, "step": 16, "pyramid_scale": 0.85, "iou_thresh": 0.30},
    {"name": "thresh=1.4 step=16 scale=0.85 iou=0.30", "threshold": 1.4, "step": 16, "pyramid_scale": 0.85, "iou_thresh": 0.30},
    {"name": "thresh=1.6 step=16 scale=0.85 iou=0.30", "threshold": 1.6, "step": 16, "pyramid_scale": 0.85, "iou_thresh": 0.30},
    {"name": "thresh=1.8 step=16 scale=0.85 iou=0.30", "threshold": 1.8, "step": 16, "pyramid_scale": 0.85, "iou_thresh": 0.30},
    {"name": "thresh=2.0 step=16 scale=0.85 iou=0.30", "threshold": 2.0, "step": 16, "pyramid_scale": 0.85, "iou_thresh": 0.30},
    {"name": "thresh=1.2 step=24 scale=0.80 iou=0.35", "threshold": 1.2, "step": 24, "pyramid_scale": 0.80, "iou_thresh": 0.35},
    {"name": "thresh=1.5 step=24 scale=0.80 iou=0.35", "threshold": 1.5, "step": 24, "pyramid_scale": 0.80, "iou_thresh": 0.35},
    {"name": "thresh=1.8 step=24 scale=0.80 iou=0.35", "threshold": 1.8, "step": 24, "pyramid_scale": 0.80, "iou_thresh": 0.35},
    {"name": "thresh=1.5 step=12 scale=0.75 iou=0.25", "threshold": 1.5, "step": 12, "pyramid_scale": 0.75, "iou_thresh": 0.25},
    {"name": "thresh=2.0 step=12 scale=0.75 iou=0.25", "threshold": 2.0, "step": 12, "pyramid_scale": 0.75, "iou_thresh": 0.25},
]

WIN_H = 62
WIN_W = 47
SUPPORTED_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def to_gray(image_float):
    if len(image_float.shape) == 3:
        return (0.299 * image_float[:, :, 0] +
                0.587 * image_float[:, :, 1] +
                0.114 * image_float[:, :, 2]) * 255.0
    return image_float * 255.0


def compute_gradients(gray):
    Ix = np.zeros_like(gray)
    Iy = np.zeros_like(gray)
    Ix[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) / 2.0
    Iy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) / 2.0
    magnitude = np.sqrt(Ix**2 + Iy**2)
    angle     = np.degrees(np.arctan2(Iy, Ix)) % 180.0
    return magnitude, angle


def build_cell_histograms(magnitude, angle, cell_size=8, num_bins=9):
    h, w  = magnitude.shape
    n_cy  = h // cell_size
    n_cx  = w // cell_size
    hists = np.zeros((n_cy, n_cx, num_bins))
    for cy in range(n_cy):
        for cx in range(n_cx):
            r0, r1 = cy * cell_size, (cy + 1) * cell_size
            c0, c1 = cx * cell_size, (cx + 1) * cell_size
            hist, _ = np.histogram(
                angle[r0:r1, c0:c1], bins=num_bins,
                range=(0, 180), weights=magnitude[r0:r1, c0:c1]
            )
            hists[cy, cx] = hist
    return hists


def normalize_blocks(cell_hists, block_size=2):
    n_cy, n_cx, num_bins = cell_hists.shape
    descriptor = []
    for by in range(n_cy - block_size + 1):
        for bx in range(n_cx - block_size + 1):
            block = cell_hists[by:by + block_size, bx:bx + block_size].flatten()
            norm  = np.sqrt(np.sum(block**2) + 1e-6)
            descriptor.append(block / norm)
    return np.concatenate(descriptor)


def hog_descriptor(image_float, cell_size=8, num_bins=9, block_size=2):
    gray     = to_gray(image_float)
    mag, ang = compute_gradients(gray)
    hists    = build_cell_histograms(mag, ang, cell_size, num_bins)
    return normalize_blocks(hists, block_size)


def image_pyramid(image_uint8, scale=0.85, min_size=64):
    img    = image_uint8.copy()
    factor = 1.0
    while True:
        yield img, factor
        h, w   = img.shape[:2]
        new_h  = int(h * scale)
        new_w  = int(w * scale)
        if new_h < min_size or new_w < min_size:
            break
        img    = cv2.resize(img, (new_w, new_h))
        factor *= scale


def iou(boxA, boxB):
    r0 = max(boxA[0], boxB[0]); c0 = max(boxA[1], boxB[1])
    r1 = min(boxA[2], boxB[2]); c1 = min(boxA[3], boxB[3])
    inter = max(0, r1 - r0) * max(0, c1 - c0)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def nms(detections, iou_thresh=0.3):
    if not detections:
        return []
    detections = sorted(detections, key=lambda x: x[0], reverse=True)
    kept = []
    while detections:
        best = detections.pop(0)
        kept.append(best)
        detections = [d for d in detections if iou(best[1:], d[1:]) < iou_thresh]
    return kept


def detect_and_classify(image_rgb, detector, gender_clf, cfg):
    detections = []
    for img_scaled, factor in image_pyramid(image_rgb, scale=cfg["pyramid_scale"]):
        h_s, w_s = img_scaled.shape[:2]
        if h_s < WIN_H or w_s < WIN_W:
            continue
        for r in range(0, h_s - WIN_H + 1, cfg["step"]):
            for c in range(0, w_s - WIN_W + 1, cfg["step"]):
                patch   = img_scaled[r:r + WIN_H, c:c + WIN_W]
                patch_f = patch.astype(np.float32) / 255.0
                desc    = hog_descriptor(patch_f).reshape(1, -1)
                score   = detector.decision_function(desc)[0]
                if score > cfg["threshold"]:
                    r0 = int(r / factor); c0 = int(c / factor)
                    r1 = int((r + WIN_H) / factor); c1 = int((c + WIN_W) / factor)
                    detections.append((score, r0, c0, r1, c1))

    detections = nms(detections, cfg["iou_thresh"])

    results = []
    for score, r0, c0, r1, c1 in detections:
        crop = image_rgb[r0:r1, c0:c1]
        if crop.size == 0:
            continue
        face_r  = cv2.resize(crop, (WIN_W, WIN_H))
        face_f  = face_r.astype(np.float32) / 255.0
        desc    = hog_descriptor(face_f).reshape(1, -1)
        gender  = gender_clf.predict(desc)[0]
        results.append((r0, c0, r1, c1, int(gender), float(score)))
    return results


def draw_detections_on_ax(ax, image_rgb, results, cfg_name):
    ax.imshow(image_rgb)
    ax.axis('off')
    ax.set_title(cfg_name, fontsize=6, pad=3, color='white',
                 bbox=dict(facecolor='#1a1a2e', edgecolor='none', pad=2))

    for (r0, c0, r1, c1, gender, score) in results:
        color = '#4fc3f7' if gender == 0 else '#f48fb1'
        label = f"{'M' if gender == 0 else 'F'} {score:.2f}"
        rect = patches.Rectangle(
            (c0, r0), c1 - c0, r1 - r0,
            linewidth=1.2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(c0 + 2, r0 - 3, label, fontsize=5, color=color,
                bbox=dict(facecolor='black', alpha=0.5, pad=1, edgecolor='none'))

    n_m = sum(1 for *_, g, _s in results if g == 0)
    n_f = sum(1 for *_, g, _s in results if g == 1)
    ax.text(0.02, 0.98, f"Faces: {len(results)}  M:{n_m} F:{n_f}",
            transform=ax.transAxes, fontsize=5.5, color='yellow', va='top',
            bbox=dict(facecolor='black', alpha=0.6, pad=1.5, edgecolor='none'))


def save_config_png(images_rgb, img_names, results_col, cfg, out_path):
    n_imgs = len(images_rgb)
    CELL_W, CELL_H = 3.2, 2.6
    LABEL_W = 1.0

    fig = plt.figure(figsize=(LABEL_W + CELL_W, n_imgs * CELL_H + 0.5), facecolor='#0d0d1a')
    gs = GridSpec(n_imgs, 2, figure=fig,
                  width_ratios=[LABEL_W / CELL_W, 1.0],
                  wspace=0.03, hspace=0.15,
                  left=0.01, right=0.99, top=0.94, bottom=0.02)

    fig.suptitle(cfg['name'], color='white', fontsize=8,
                 y=0.98, va='top', fontfamily='monospace')

    for i in range(n_imgs):
        ax_lbl = fig.add_subplot(gs[i, 0])
        ax_lbl.set_facecolor('#0d0d1a')
        ax_lbl.axis('off')
        ax_lbl.text(0.95, 0.5, img_names[i], transform=ax_lbl.transAxes,
                    fontsize=6, color='#a0c4ff', va='center', ha='right',
                    fontfamily='monospace')

        ax = fig.add_subplot(gs[i, 1])
        ax.set_facecolor('#0d0d1a')
        draw_detections_on_ax(ax, images_rgb[i], results_col[i], cfg['name'])

    fig.savefig(str(out_path), dpi=120, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)


def save_combined_png(images_rgb, img_names, results_grid, out_path):
    n_imgs = len(images_rgb)
    n_cfgs = len(CONFIGS)
    CELL_W, CELL_H = 3.2, 2.6
    LABEL_W = 1.0

    fig = plt.figure(figsize=(LABEL_W + n_cfgs * CELL_W, n_imgs * CELL_H + 0.6), facecolor='#0d0d1a')
    gs = GridSpec(n_imgs, 1 + n_cfgs, figure=fig,
                  width_ratios=[LABEL_W / CELL_W] + [1.0] * n_cfgs,
                  wspace=0.04, hspace=0.15,
                  left=0.01, right=0.99, top=0.97, bottom=0.01)

    fig.suptitle(
        f'HOG Face Detector — {n_imgs} images × {n_cfgs} configs  |  Blue=Male  Pink=Female',
        color='white', fontsize=10, y=0.99, va='top', fontfamily='monospace'
    )

    for i in range(n_imgs):
        ax_lbl = fig.add_subplot(gs[i, 0])
        ax_lbl.set_facecolor('#0d0d1a')
        ax_lbl.axis('off')
        ax_lbl.text(0.95, 0.5, img_names[i], transform=ax_lbl.transAxes,
                    fontsize=6, color='#a0c4ff', va='center', ha='right',
                    fontfamily='monospace')

        for j, cfg in enumerate(CONFIGS):
            ax = fig.add_subplot(gs[i, j + 1])
            ax.set_facecolor('#0d0d1a')
            draw_detections_on_ax(ax, images_rgb[i], results_grid[i][j], cfg['name'])

    fig.savefig(str(out_path), dpi=120, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)


def main():
    script_dir = Path(__file__).parent.resolve()
    totest_dir = script_dir / 'totest'
    out_dir    = script_dir / 'results'
    out_dir.mkdir(exist_ok=True)

    if not totest_dir.exists():
        print(f"[ERROR] Папка 'totest' не найдена: {totest_dir}")
        sys.exit(1)

    image_paths = sorted([p for p in totest_dir.iterdir()
                          if p.suffix.lower() in SUPPORTED_EXT])
    if not image_paths:
        print("[ERROR] В папке totest не найдено изображений")
        sys.exit(1)

    print(f"Найдено изображений: {len(image_paths)}")

    for fname in ['detector_model.pkl', 'gender_model.pkl']:
        if not (script_dir / fname).exists():
            print(f"[ERROR] {fname} не найден в {script_dir}")
            sys.exit(1)

    with open(script_dir / 'detector_model.pkl', 'rb') as f:
        detector = pickle.load(f)
    with open(script_dir / 'gender_model.pkl', 'rb') as f:
        gender_clf = pickle.load(f)

    print("Модели загружены.")

    images_rgb = []
    img_names  = []
    for p in image_paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            print(f"  [WARN] Не удалось прочитать {p.name}, пропускаем.")
            continue
        h, w = bgr.shape[:2]
        if max(h, w) > 640:
            s = 640 / max(h, w)
            bgr = cv2.resize(bgr, (int(w * s), int(h * s)))
        images_rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        img_names.append(p.name)

    n_imgs = len(images_rgb)
    n_cfgs = len(CONFIGS)
    results_grid = [[None] * n_cfgs for _ in range(n_imgs)]

    for j, cfg in enumerate(CONFIGS):
        print(f"\n[{j+1}/{n_cfgs}] {cfg['name']}")
        for i, (img_rgb, name) in enumerate(zip(images_rgb, img_names)):
            try:
                res = detect_and_classify(img_rgb, detector, gender_clf, cfg)
            except Exception as e:
                print(f"  [WARN] {name}: {e}")
                res = []
            results_grid[i][j] = res
            print(f"  {name}: {len(res)} face(s)  " +
                  ", ".join(f"({'M' if g==0 else 'F'} {s:.2f})" for *_, g, s in res))

        config_png = out_dir / f"config_{j+1:02d}.png"
        save_config_png(images_rgb, img_names,
                        [results_grid[i][j] for i in range(n_imgs)],
                        cfg, config_png)
        print(f"  => Сохранено: {config_png.name}")

    combined_png = script_dir / 'batch_results.png'
    print(f"\nСохраняем общий график -> {combined_png} ...")
    save_combined_png(images_rgb, img_names, results_grid, combined_png)

    print("\n── Статистика ──")
    for j, cfg in enumerate(CONFIGS):
        total = sum(len(results_grid[i][j]) for i in range(n_imgs))
        m = sum(sum(1 for *_, g, _s in results_grid[i][j] if g == 0) for i in range(n_imgs))
        f = sum(sum(1 for *_, g, _s in results_grid[i][j] if g == 1) for i in range(n_imgs))
        print(f"  [{j+1:2d}] {cfg['name']:42s}  faces={total:3d}  M={m:3d}  F={f:3d}")

    print(f"\nГотово. Индивидуальные PNG: {out_dir}/  |  Общий: {combined_png}")


if __name__ == '__main__':
    main()