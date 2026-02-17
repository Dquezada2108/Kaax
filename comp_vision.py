import cv2
import numpy as np

img = cv2.imread("plant.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

h, w = img.shape[:2]

cv2.namedWindow("Mask")
cv2.namedWindow("Preview")

def nothing(x):
    pass

# ---- Sliders ----
cv2.createTrackbar("Clusters", "Mask", 4, 8, nothing)
cv2.createTrackbar("RedBias", "Mask", 50, 100, nothing)
cv2.createTrackbar("LabThreshold", "Mask", 30, 100, nothing)
cv2.createTrackbar("Grow", "Mask", 3, 10, nothing)

while True:
    K = max(2, cv2.getTrackbarPos("Clusters", "Mask"))
    red_bias = cv2.getTrackbarPos("RedBias", "Mask") / 100.0
    lab_mult = cv2.getTrackbarPos("LabThreshold", "Mask") / 10.0
    grow = cv2.getTrackbarPos("Grow", "Mask")

    pixels = img.reshape((-1,3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    labels = labels.flatten()

    # --- Select Sargassum cluster ---
    scores = []
    for c in centers:
        r, g, b = c
        score = (r - b) * red_bias + g * 0.3
        scores.append(score)

    plant_cluster = np.argmax(scores)

    mask = (labels == plant_cluster).astype(np.uint8)
    mask = mask.reshape(h, w) * 255

    # --- Extract pixels ---
    plant_pixels = img[mask > 0]
    if len(plant_pixels) < 100:
        continue

    lab = cv2.cvtColor(plant_pixels.reshape(-1,1,3).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1,3)

    centroid = np.mean(lab, axis=0)
    distances = np.linalg.norm(lab - centroid, axis=1)
    sigma = np.std(distances)
    threshold = sigma * lab_mult

    # --- Rebuild mask using Lab distance ---
    lab_full = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    dist_full = np.linalg.norm(lab_full - centroid, axis=2)
    mask = (dist_full < threshold).astype(np.uint8) * 255

    # --- Region grow ---
    kernel = np.ones((7,7), np.uint8)
    for _ in range(grow):
        mask = cv2.dilate(mask, kernel)
        mask = cv2.bitwise_and(mask, mask)

    preview = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow("Mask", mask)
    cv2.imshow("Preview", cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) == 27:  # ESC
        break

cv2.destroyAllWindows()
