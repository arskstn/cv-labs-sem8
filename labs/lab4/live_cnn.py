import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
import os
import sys
from collections import deque

CAM_INDEX      = 2
THRESHOLD      = 0.84
STEP           = 20
SCALE_DOWN     = 0.2
WIN            = 64
MIN_BOX        = 45
PYR_SCALE      = 0.80
IOU_THRESH     = 0.32
CROP_PAD       = 0.15
EMO_SMOOTH_N   = 10
EMO_MIN_CONF   = 0.42

EMOTIONS    = ['angry', 'happy', 'sad', 'surprised']
EMO_COLORS  = {
    'angry':     (80,  80,  255),
    'happy':     (80,  220, 80),
    'sad':       (220, 120, 60),
    'surprised': (60,  200, 220),
}
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else
                       'cpu' if not torch.cuda.is_available() else 'cuda')


class FaceDetectorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class EmotionCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(2)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.45),
            nn.Linear(128 * 2 * 2, num_classes),
        )

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


infer_det_tf = T.Compose([T.ToPILImage(), T.ToTensor(), T.Normalize(MEAN, STD)])
infer_emo_tf = T.Compose([T.ToPILImage(), T.Grayscale(1), T.ToTensor(), T.Normalize([0.5], [0.5])])


def iou(a, b):
    r0 = max(a[0], b[0]); c0 = max(a[1], b[1])
    r1 = min(a[2], b[2]); c1 = min(a[3], b[3])
    inter = max(0, r1 - r0) * max(0, c1 - c0)
    ua = (a[2]-a[0])*(a[3]-a[1]); ub = (b[2]-b[0])*(b[3]-b[1])
    return inter / (ua + ub - inter) if (ua + ub - inter) > 0 else 0.0


def nms(dets, thresh):
    dets = sorted(dets, key=lambda x: x[0], reverse=True)
    kept = []
    while dets:
        best = dets.pop(0); kept.append(best)
        dets = [d for d in dets if iou(best[1:], d[1:]) < thresh]
    return kept


def padded_crop(img_rgb, r0, c0, r1, c1, pad=CROP_PAD):
    h, w = img_rgb.shape[:2]
    dh   = int((r1 - r0) * pad)
    dw   = int((c1 - c0) * pad)
    r0p  = max(0, r0 - dh)
    c0p  = max(0, c0 - dw)
    r1p  = min(h, r1 + dh)
    c1p  = min(w, c1 + dw)
    return img_rgb[r0p:r1p, c0p:c1p]


@torch.no_grad()
def detect(img_rgb, det_model, emo_model, threshold):
    dets = []
    img  = img_rgb.copy()
    factor = 1.0
    while True:
        h, w = img.shape[:2]
        if h < WIN or w < WIN:
            break
        for r in range(0, h - WIN + 1, STEP):
            for c in range(0, w - WIN + 1, STEP):
                p  = img[r:r+WIN, c:c+WIN]
                t  = infer_det_tf(p).unsqueeze(0).to(DEVICE)
                sc = torch.sigmoid(det_model(t)).item()
                if sc > threshold:
                    r0i, c0i = int(r/factor), int(c/factor)
                    r1i, c1i = int((r+WIN)/factor), int((c+WIN)/factor)
                    if (r1i - r0i) >= MIN_BOX and (c1i - c0i) >= MIN_BOX:
                        dets.append((sc, r0i, c0i, r1i, c1i))
        nh, nw = int(h * PYR_SCALE), int(w * PYR_SCALE)
        if nh < 64 or nw < 64:
            break
        img = cv2.resize(img, (nw, nh))
        factor *= PYR_SCALE

    results = []
    for sc, r0, c0, r1, c1 in nms(dets, IOU_THRESH):
        crop = padded_crop(img_rgb, r0, c0, r1, c1)
        if crop.size == 0:
            continue
        face  = cv2.resize(crop, (WIN, WIN))
        t     = infer_emo_tf(face).unsqueeze(0).to(DEVICE)
        probs = torch.softmax(emo_model(t), 1)[0].cpu().numpy()
        results.append((r0, c0, r1, c1, probs, sc))
    return results


def draw(frame, smoothed, threshold):
    out = frame.copy()
    for r0, c0, r1, c1, probs, _ in smoothed:
        best_i = probs.argmax()
        conf   = probs[best_i]
        if conf >= EMO_MIN_CONF:
            emo = EMOTIONS[best_i]
            col = EMO_COLORS[emo]
            label = f'{emo} {conf:.2f}'
        else:
            col   = (180, 180, 180)
            label = f'? {conf:.2f}'
        cv2.rectangle(out, (c0, r0), (c1, r1), col, 2)
        cv2.putText(out, label, (c0, max(r0 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        bar_y = r1 + 4
        bar_w = c1 - c0
        for i, (p, name) in enumerate(zip(probs, EMOTIONS)):
            bw  = int(p * bar_w)
            bcol = EMO_COLORS[name]
            cv2.rectangle(out, (c0, bar_y + i*8), (c0 + bw, bar_y + i*8 + 6), bcol, -1)
    cv2.putText(out, f'thr={threshold:.2f}  +/- adjust  Q quit',
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(out, f'faces: {len(smoothed)}',
                (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return out


def load_model(cls, path):
    if not os.path.exists(path):
        print(f'[ERROR] {path} not found. Run lab4_nn.ipynb first.')
        sys.exit(1)
    m = cls().to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    return m


def main():
    det_model = load_model(FaceDetectorCNN, '../lab4-trained/face_detector_cnn.pth')
    emo_model = load_model(lambda: EmotionCNN(len(EMOTIONS)), '../lab4-trained/emotion_cnn_noadded.pth')

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f'[ERROR] camera {CAM_INDEX} unavailable')
        sys.exit(1)

    cv2.namedWindow('CNN face + emotion', cv2.WINDOW_NORMAL)
    threshold = THRESHOLD

    emo_buf = deque(maxlen=EMO_SMOOTH_N)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w  = frame.shape[:2]
        small = cv2.resize(frame, (int(w * SCALE_DOWN), int(h * SCALE_DOWN)))
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        res_small = detect(rgb, det_model, emo_model, threshold)

        inv = 1.0 / SCALE_DOWN
        results_full = [(int(r0*inv), int(c0*inv), int(r1*inv), int(c1*inv), probs, sc)
                        for r0, c0, r1, c1, probs, sc in res_small]

        if results_full:
            emo_buf.append(results_full[0][4])
        smooth_probs = np.mean(emo_buf, axis=0) if emo_buf else None

        smoothed = []
        for i, (r0, c0, r1, c1, probs, sc) in enumerate(results_full):
            p = smooth_probs if (i == 0 and smooth_probs is not None) else probs
            smoothed.append((r0, c0, r1, c1, p, sc))

        cv2.imshow('CNN face + emotion', draw(frame, smoothed, threshold))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break
        elif key in (ord('+'), ord('=')):
            threshold = min(threshold + 0.05, 0.99)
        elif key == ord('-'):
            threshold = max(threshold - 0.05, 0.1)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
