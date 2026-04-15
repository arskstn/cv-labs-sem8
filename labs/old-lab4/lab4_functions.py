def to_gray(image_float):
    if len(image_float.shape) == 3:
        return (0.299*image_float[:,:,0] +
                0.587*image_float[:,:,1] +
                0.114*image_float[:,:,2]) * 255.0
    return image_float * 255.0

def compute_gradients(gray):
    Ix = np.zeros_like(gray)
    Iy = np.zeros_like(gray)
    Ix[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) / 2.0
    Iy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) / 2.0
    magnitude = np.sqrt(Ix**2 + Iy**2)
    angle     = np.degrees(np.arctan2(Iy, Ix)) % 180.0
    return Ix, Iy, magnitude, angle

def build_cell_histograms(magnitude, angle, cell_size=8, num_bins=9):
    h, w   = magnitude.shape
    n_cy   = h // cell_size
    n_cx   = w // cell_size
    bw     = 180.0 / num_bins
    hists  = np.zeros((n_cy, n_cx, num_bins))

    for cy in range(n_cy):
        for cx in range(n_cx):
            r0, r1 = cy*cell_size, (cy+1)*cell_size
            c0, c1 = cx*cell_size, (cx+1)*cell_size
            cell_mag = magnitude[r0:r1, c0:c1]
            cell_ang = angle[r0:r1, c0:c1]
            hist, _ = np.histogram(cell_ang, bins=num_bins,
                                   range=(0, 180), weights=cell_mag)
            hists[cy, cx] = hist
    return hists

def normalize_blocks(cell_hists, block_size=2):
    n_cy, n_cx, num_bins = cell_hists.shape
    descriptor = []
    for by in range(n_cy - block_size + 1):
        for bx in range(n_cx - block_size + 1):
            block = cell_hists[by:by+block_size, bx:bx+block_size].flatten()
            norm  = np.sqrt(np.sum(block**2) + 1e-6)
            descriptor.append(block / norm)
    return np.concatenate(descriptor)

def hog_descriptor(image_float, cell_size=8, num_bins=9, block_size=2):
    gray  = to_gray(image_float)
    _, _, magnitude, angle = compute_gradients(gray)
    hists = build_cell_histograms(magnitude, angle, cell_size, num_bins)
    return normalize_blocks(hists, block_size)

def make_negatives(all_imgs, img_h, img_w, n_total):
    n_imgs   = len(all_imgs)
    per_type = n_total // 6 + 1
    negatives = []

    idx = np.random.permutation(n_imgs)

    for i in range(per_type):
        patch = all_imgs[idx[i % n_imgs]].copy()
        negatives.append(patch[::-1, :, :])

    for i in range(per_type):
        patch = all_imgs[idx[(i + per_type) % n_imgs]].copy()
        shift = max(1, img_h // 3)

        shifted = np.zeros_like(patch)
        shifted[shift:, :, :] = patch[:img_h - shift, :, :]
        shifted[:shift, :, :]  = np.random.rand(shift, img_w, 3).astype(np.float32)
        negatives.append(shifted)


    for i in range(per_type):
        patch = all_imgs[idx[(i + 2*per_type) % n_imgs]].copy()

        rotated = np.transpose(patch, (1, 0, 2))
        rotated = rotated[:img_h, :img_w, :] if rotated.shape[0] >= img_h and rotated.shape[1] >= img_w \
                  else np.pad(rotated, ((0, max(0, img_h-rotated.shape[0])),
                                        (0, max(0, img_w-rotated.shape[1])),
                                        (0, 0)), mode='edge')[:img_h, :img_w, :]
        negatives.append(rotated)


    for i in range(per_type):
        patch = all_imgs[idx[(i + 3*per_type) % n_imgs]].copy()
        negatives.append(patch[::-1, ::-1, :])


    for i in range(per_type):

        src = all_imgs[idx[(i + 4*per_type) % n_imgs]]
        big = np.kron(src, np.ones((2, 2, 1)))
        h_b, w_b = big.shape[:2]

        r0 = h_b - img_h
        c0 = w_b - img_w
        negatives.append(np.clip(big[r0:r0+img_h, c0:c0+img_w, :], 0, 1).astype(np.float32))

    for i in range(per_type):
        noise = np.random.rand(img_h, img_w, 3).astype(np.float32)
        negatives.append(noise)

    return negatives[:n_total]


def sliding_window(image_uint8, win_h, win_w, step=16):
    h, w = image_uint8.shape[:2]
    for r in range(0, h - win_h + 1, step):
        for c in range(0, w - win_w + 1, step):
            patch = image_uint8[r:r+win_h, c:c+win_w]
            yield r, c, patch


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
    r0 = max(boxA[0], boxB[0]);  c0 = max(boxA[1], boxB[1])
    r1 = min(boxA[2], boxB[2]);  c1 = min(boxA[3], boxB[3])
    inter = max(0, r1-r0) * max(0, c1-c0)
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
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
        detections = [
            d for d in detections
            if iou(best[1:], d[1:]) < iou_thresh
        ]
    return kept


def detect_and_classify(image_uint8, detector, gender_clf,
                        win_h, win_w, step=16, scale=0.85,
                        det_threshold=0.5, iou_thresh=0.3):
    detections = []

    for img_scaled, factor in image_pyramid(image_uint8, scale=scale):
        for r, c, patch in sliding_window(img_scaled, win_h, win_w, step):
            patch_f = patch.astype(np.float32) / 255.0
            desc    = hog_descriptor(patch_f).reshape(1, -1)
            score   = detector.decision_function(desc)[0]
            if score > det_threshold:
                r0 = int(r / factor);  c0 = int(c / factor)
                r1 = int((r+win_h) / factor);  c1 = int((c+win_w) / factor)
                detections.append((score, r0, c0, r1, c1))

    detections = nms(detections, iou_thresh)

    results = []
    for score, r0, c0, r1, c1 in detections:
        face_crop = image_uint8[r0:r1, c0:c1]
        if face_crop.size == 0:
            continue
        face_resized = cv2.resize(face_crop, (win_w, win_h))
        face_f  = face_resized.astype(np.float32) / 255.0
        desc    = hog_descriptor(face_f).reshape(1, -1)
        gender  = gender_clf.predict(desc)[0]  # 0=муж, 1=жен
        results.append((r0, c0, r1, c1, gender))

    return results

def draw_results(image_uint8, results):
    img = image_uint8.copy()
    for (r0, c0, r1, c1, gender) in results:
        color  = [255, 0, 0] if gender == 0 else [0, 0, 255]
        label  = 'M' if gender == 0 else 'F'

        img[r0, c0:c1] = color
        img[r1, c0:c1] = color
        img[r0:r1, c0] = color
        img[r0:r1, c1] = color

        cv2.putText(img, label, (c0+2, r0+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img