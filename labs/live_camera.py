import numpy as np
import cv2
import pickle
import sys
import os

CAM_INDEX     = 2      # индекс камеры (2 = встроенная MacBook в данной конфигурации)
DET_THRESHOLD = 1.2    # порог уверенности детектора
STEP          = 24     # шаг скользящего окна (пикс.)
SCALE_DOWN    = 0.25    # уменьшение входного кадра для скорости
WIN_H         = 62     # высота окна (должна совпадать с высотой патчей LFW при resize=0.5)
WIN_W         = 47     # ширина окна
PYRAMID_SCALE = 0.75   # коэффициент уменьшения пирамиды масштабов
IOU_THRESH    = 0.3    # порог IoU для NMS
# ──────────────────────────────────────────────────────────────────────────────


# ── HOG-дескриптор (дублируем из ноутбука — скрипт самодостаточен) ────────────

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
    gray      = to_gray(image_float)
    mag, ang  = compute_gradients(gray)
    hists     = build_cell_histograms(mag, ang, cell_size, num_bins)
    return normalize_blocks(hists, block_size)


# ── Детектор ──────────────────────────────────────────────────────────────────

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


def detect_and_classify(image_uint8, detector, gender_clf,
                        win_h, win_w, step, scale, threshold, iou_thresh):
    detections = []
    for img_scaled, factor in image_pyramid(image_uint8, scale=scale):
        h_s, w_s = img_scaled.shape[:2]
        if h_s < win_h or w_s < win_w:
            continue
        for r in range(0, h_s - win_h + 1, step):
            for c in range(0, w_s - win_w + 1, step):
                patch   = img_scaled[r:r + win_h, c:c + win_w]
                patch_f = patch.astype(np.float32) / 255.0
                desc    = hog_descriptor(patch_f).reshape(1, -1)
                score   = detector.decision_function(desc)[0]
                if score > threshold:
                    r0 = int(r / factor); c0 = int(c / factor)
                    r1 = int((r + win_h) / factor); c1 = int((c + win_w) / factor)
                    detections.append((score, r0, c0, r1, c1))

    detections = nms(detections, iou_thresh)

    results = []
    for score, r0, c0, r1, c1 in detections:
        crop = image_uint8[r0:r1, c0:c1]
        if crop.size == 0:
            continue
        face_r  = cv2.resize(crop, (win_w, win_h))
        face_f  = face_r.astype(np.float32) / 255.0
        desc    = hog_descriptor(face_f).reshape(1, -1)
        gender  = gender_clf.predict(desc)[0]
        results.append((r0, c0, r1, c1, int(gender)))
    return results


def draw_results(frame_bgr, results, threshold):
    img = frame_bgr.copy()
    for r0, c0, r1, c1, gender in results:
        # Синий = мужчина, Розовый = женщина (BGR!)
        color = (255, 60, 60) if gender == 0 else (147, 20, 255)
        label = 'Male' if gender == 0 else 'Female'
        cv2.rectangle(img, (c0, r0), (c1, r1), color, 2)
        cv2.putText(img, label, (c0, r0 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    # Инфо в углу
    cv2.putText(img, f'Threshold: {threshold:.2f}  [+/-] to adjust',
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(img, f'Faces: {len(results)}   [Q] to quit',
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return img


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global DET_THRESHOLD

    # Загружаем модели
    for fname in ['detector_model.pkl', 'gender_model.pkl']:
        if not os.path.exists(fname):
            print(f'[ERROR] Файл {fname} не найден.')
            print('  Сначала выполните ячейки 4.1–4.4 в ноутбуке lab4.ipynb')
            sys.exit(1)

    with open('detector_model.pkl', 'rb') as f:
        detector = pickle.load(f)
    with open('gender_model.pkl', 'rb') as f:
        gender_clf = pickle.load(f)

    print('Модели загружены.')
    print(f'Открываем камеру {CAM_INDEX}...')

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f'[ERROR] Не удалось открыть камеру с индексом {CAM_INDEX}')
        sys.exit(1)

    cv2.namedWindow('HOG Face Detector', cv2.WINDOW_NORMAL)
    print('Управление: Q — выход, + — поднять порог, - — снизить порог')

    threshold = DET_THRESHOLD

    while True:
        ret, frame = cap.read()
        if not ret:
            print('[ERROR] Не удалось получить кадр с камеры')
            break

        # Уменьшаем для скорости
        h, w   = frame.shape[:2]
        small  = cv2.resize(frame, (int(w * SCALE_DOWN), int(h * SCALE_DOWN)))

        # Детектируем (small — BGR, передаём BGR, внутри конвертируем в float)
        small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results   = detect_and_classify(
            small_rgb, detector, gender_clf,
            WIN_H, WIN_W,
            step=STEP,
            scale=PYRAMID_SCALE,
            threshold=threshold,
            iou_thresh=IOU_THRESH
        )

        # Переводим координаты обратно в полный масштаб
        results_full = [
            (int(r0 / SCALE_DOWN), int(c0 / SCALE_DOWN),
             int(r1 / SCALE_DOWN), int(c1 / SCALE_DOWN), g)
            for r0, c0, r1, c1, g in results
        ]

        frame_drawn = draw_results(frame, results_full, threshold)
        cv2.imshow('HOG Face Detector', frame_drawn)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('+') or key == ord('='):
            threshold = min(threshold + 0.05, 3.0)
            print(f'Порог: {threshold:.2f}')
        elif key == ord('-'):
            threshold = max(threshold - 0.05, -1.0)
            print(f'Порог: {threshold:.2f}')

    cap.release()
    cv2.destroyAllWindows()
    print('Завершено.')


if __name__ == '__main__':
    main()
