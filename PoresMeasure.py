import torch
import numpy as np
import cv2
from model import PointNetLike
from scipy.interpolate import interp1d
from predict import resample_contour, normalize_contour, compute_scalars, predict_single_contour, getContourList
from extract_to_data import getContour
from area import contourArea
import time
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "pointnet_pore_classifier.pth" 
NUM_POINTS = 64

print("Loading model...")
model = PointNetLike(num_classes=2, num_points=NUM_POINTS)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("Model loaded successfully.")

IMAGE_PATH = "images\\6.JPG" 
poreContours = []
time1 = time.time()
try:

    contours_list = getContourList(IMAGE_PATH)
    print(f"Found {len(contours_list)} contours in the image.")
    for i, cnt in enumerate(contours_list):
        try:
            if len(cnt) < 10: # 跳过太小的轮廓，反正孔洞也不可能这么小
                continue
            cls, conf = predict_single_contour(model, cnt, DEVICE, NUM_POINTS)
            print(f"Contour {i}: Predicted class={cls} ({'Pore' if cls==1 else 'Noise/Fiber'}), Confidence={conf:.4f}")
            if cls == 1:
                poreContours.append(cnt.astype(np.int32))
        except Exception as e:
             print(f"Failed to predict contour {i}: {e}")


except FileNotFoundError as e:
    print(e)
time2 = time.time()

print("time used: " + str(time2 - time1))

refinedContours = getContour(IMAGE_PATH)
refinedContours_int32 = [cnt.astype(np.int32) for cnt in refinedContours]
img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
mask_color = cv2.cvtColor(np.zeros_like(img), cv2.COLOR_GRAY2BGR)
cv2.drawContours(mask_color, refinedContours_int32, -1, (128, 128, 128), 1) 
cv2.drawContours(mask_color, poreContours, -1, (0, 0, 255), 2) 

areas = []
for contour in poreContours:
    area, closed, filled_col, filled_row = contourArea(contour, threshold = 0.1)
    areas.append(int(area))
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        centreGravity = (cx, cy)
    cv2.putText(mask_color, f"{int(area)}", centreGravity, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


cv2.imshow("Detected Pores", mask_color)
cv2.waitKey(0)
cv2.destroyAllWindows() 


if not areas:
    print("No areas provided. Cannot plot histogram.")
else:
    print(f"Plotting distribution histogram for {len(areas)} pores.")

    # --- Histogram ---
    fig, ax = plt.subplots(figsize=(10, 6))
    n, bins, patches = ax.hist(areas, bins="auto", color='lightcoral', edgecolor='black', alpha=0.7)

    ax.set_xlabel('Pore Area (pixels²)')
    ax.set_ylabel('Number of Pores')
    ax.set_title('Distribution Histogram of Pore Areas')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    for i in range(len(n)):
        if n[i] > 0:
            ax.text(bin_centers[i], n[i] + max(n)*0.01,
                    str(int(n[i])), ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

 
    print("\n--- Pore Area Distribution Statistics ---")
    print(f"Total number of pores detected: {len(areas)}")
    print(f"Min pore area: {np.min(areas):.2f} pixels²")
    print(f"Max pore area: {np.max(areas):.2f} pixels²")
    print(f"Mean pore area: {np.mean(areas):.2f} pixels²")
    print(f"Median pore area: {np.median(areas):.2f} pixels²")
    print(f"Standard deviation of pore area: {np.std(areas):.2f} pixels²")
