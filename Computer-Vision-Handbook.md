# 🧠 Computer Vision: Complete A–Z Handbook & Cheat Sheet

> **The only CV reference you'll ever need** — Theory, Math, Code, Models, Pipelines, Tips & Tricks.  
> Covers: Classical CV → Deep Learning → State-of-the-Art (2024–25)

---

## 📚 Table of Contents

1. [Fundamentals & Image Basics](#1-fundamentals--image-basics)
2. [Image Processing & Filtering](#2-image-processing--filtering)
3. [Feature Detection & Description](#3-feature-detection--description)
4. [Classical Object Detection](#4-classical-object-detection)
5. [Deep Learning for CV — Core Concepts](#5-deep-learning-for-cv--core-concepts)
6. [Convolutional Neural Networks (CNNs)](#6-convolutional-neural-networks-cnns)
7. [Object Detection — Deep Learning](#7-object-detection--deep-learning)
8. [Image Segmentation](#8-image-segmentation)
9. [Image Classification](#9-image-classification)
10. [Object Tracking](#10-object-tracking)
11. [Pose Estimation](#11-pose-estimation)
12. [Face Recognition & Analysis](#12-face-recognition--analysis)
13. [Optical Flow & Motion Analysis](#13-optical-flow--motion-analysis)
14. [Depth Estimation & 3D Vision](#14-depth-estimation--3d-vision)
15. [Image Generation & GANs](#15-image-generation--gans)
16. [Vision Transformers & Modern Architectures](#16-vision-transformers--modern-architectures)
17. [Multimodal Vision-Language Models](#17-multimodal-vision-language-models)
18. [OCR & Document Understanding](#18-ocr--document-understanding)
19. [Medical Imaging](#19-medical-imaging)
20. [Video Understanding](#20-video-understanding)
21. [Data Augmentation](#21-data-augmentation)
22. [Transfer Learning & Fine-Tuning](#22-transfer-learning--fine-tuning)
23. [Model Evaluation & Metrics](#23-model-evaluation--metrics)
24. [Model Optimization & Deployment](#24-model-optimization--deployment)
25. [Libraries & Frameworks Quick Reference](#25-libraries--frameworks-quick-reference)
26. [Datasets Encyclopedia](#26-datasets-encyclopedia)
27. [Loss Functions Reference](#27-loss-functions-reference)
28. [Architecture Cheat Sheet](#28-architecture-cheat-sheet)
29. [CV Interview Q&A](#29-cv-interview-qa)
30. [End-to-End Project Templates](#30-end-to-end-project-templates)

---

## 1. Fundamentals & Image Basics

### 1.1 What is an Image?

An image is a 2D (or 3D) array of pixel values:

| Type | Shape | Values |
|---|---|---|
| Grayscale | `(H, W)` | 0–255 |
| RGB | `(H, W, 3)` | 0–255 per channel |
| RGBA | `(H, W, 4)` | + Alpha (transparency) |
| Float | `(H, W, C)` | 0.0–1.0 (normalized) |
| HDR | `(H, W, C)` | > 1.0 float |

### 1.2 Color Spaces

| Space | Channels | Use Case |
|---|---|---|
| **RGB** | Red, Green, Blue | General, display |
| **BGR** | Blue, Green, Red | OpenCV default |
| **HSV** | Hue, Saturation, Value | Color filtering/thresholding |
| **HSL** | Hue, Saturation, Lightness | UI, color picking |
| **LAB** | Lightness, A, B | Perceptually uniform |
| **YCrCb** | Luma, Cr, Cb | Skin detection, compression |
| **Grayscale** | Intensity | Preprocessing |
| **XYZ** | CIE XYZ | Color science |

```python
import cv2
import numpy as np

# Load image
img_bgr = cv2.imread("image.jpg")

# Color space conversions
img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
img_hsv   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
img_lab   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

# Normalize to [0, 1]
img_float = img_bgr.astype(np.float32) / 255.0

# Back to uint8
img_uint8 = (img_float * 255).astype(np.uint8)

print(f"Shape: {img_bgr.shape}, dtype: {img_bgr.dtype}")
print(f"Min: {img_bgr.min()}, Max: {img_bgr.max()}")
```

### 1.3 Image I/O

```python
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# --- OpenCV ---
img = cv2.imread("image.jpg")                    # BGR
img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.imread("image.jpg", cv2.IMREAD_UNCHANGED)  # with alpha
cv2.imwrite("output.jpg", img)
cv2.imwrite("output.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 9])

# --- PIL / Pillow ---
pil_img = Image.open("image.jpg")
pil_img = pil_img.convert("RGB")          # ensure RGB
pil_img.save("output.jpg", quality=95)

# PIL <-> NumPy
np_img = np.array(pil_img)
pil_back = Image.fromarray(np_img)

# PIL <-> OpenCV
cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# --- Matplotlib display ---
plt.imshow(img_rgb)
plt.axis("off")
plt.title("Image")
plt.show()

# Display multiple
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(img_rgb);      axes[0].set_title("RGB")
axes[1].imshow(img_gray, cmap="gray"); axes[1].set_title("Gray")
axes[2].imshow(img_hsv);      axes[2].set_title("HSV")
plt.tight_layout(); plt.show()
```

### 1.4 Basic Image Operations

```python
import cv2
import numpy as np

img = cv2.imread("image.jpg")
h, w, c = img.shape

# --- Resize ---
resized = cv2.resize(img, (640, 480))
resized = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)  # half size
# Interpolation methods:
# INTER_NEAREST  - fastest, pixelated (downscaling)
# INTER_LINEAR   - bilinear (default, good balance)
# INTER_CUBIC    - bicubic (better quality, slower)
# INTER_AREA     - best for shrinking
# INTER_LANCZOS4 - best quality for upscaling

# --- Crop ---
crop = img[100:300, 200:400]    # [y1:y2, x1:x2]

# --- Flip ---
flip_h = cv2.flip(img, 1)  # horizontal
flip_v = cv2.flip(img, 0)  # vertical
flip_b = cv2.flip(img, -1) # both

# --- Rotate ---
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle=45, scale=1.0)
rotated = cv2.warpAffine(img, M, (w, h))

# --- Padding ---
padded = cv2.copyMakeBorder(img, 10, 10, 10, 10,
                             cv2.BORDER_CONSTANT, value=(0,0,0))
# Border types: BORDER_REFLECT, BORDER_REPLICATE, BORDER_WRAP

# --- Channel operations ---
b, g, r = cv2.split(img)
merged = cv2.merge([b, g, r])

# --- Bitwise operations ---
mask = np.zeros(img.shape[:2], dtype=np.uint8)
cv2.circle(mask, (w//2, h//2), 100, 255, -1)
masked = cv2.bitwise_and(img, img, mask=mask)

# --- Drawing ---
cv2.rectangle(img, (x1,y1), (x2,y2), color=(0,255,0), thickness=2)
cv2.circle(img, (cx,cy), radius=30, color=(255,0,0), thickness=-1)
cv2.line(img, (0,0), (w,h), color=(0,0,255), thickness=2)
cv2.putText(img, "Hello CV", (50,50), cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1, color=(255,255,0), thickness=2)

# --- Histograms ---
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
# Equalize histogram (grayscale)
eq = cv2.equalizeHist(gray)
# CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl = clahe.apply(gray)
```

### 1.5 Geometric Transformations

```python
import cv2
import numpy as np

img = cv2.imread("image.jpg")
h, w = img.shape[:2]

# --- Affine Transform (3 point pairs) ---
src_pts = np.float32([[0,0],[w-1,0],[0,h-1]])
dst_pts = np.float32([[50,10],[w-80,20],[10,h-80]])
M_affine = cv2.getAffineTransform(src_pts, dst_pts)
affine = cv2.warpAffine(img, M_affine, (w, h))

# --- Perspective Transform (4 point pairs) ---
src_pts4 = np.float32([[0,0],[w,0],[w,h],[0,h]])
dst_pts4 = np.float32([[50,30],[w-50,20],[w-30,h-20],[30,h-30]])
M_persp = cv2.getPerspectiveTransform(src_pts4, dst_pts4)
persp = cv2.warpPerspective(img, M_persp, (w, h))

# Inverse perspective
M_inv = np.linalg.inv(M_persp)
back = cv2.warpPerspective(persp, M_inv, (w, h))

# --- Document dewarping / bird's eye view ---
def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0]-bl[0])**2) + ((br[1]-bl[1])**2))
    widthB = np.sqrt(((tr[0]-tl[0])**2) + ((tr[1]-tl[1])**2))
    maxW = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0]-br[0])**2) + ((tr[1]-br[1])**2))
    heightB = np.sqrt(((tl[0]-bl[0])**2) + ((tl[1]-bl[1])**2))
    maxH = max(int(heightA), int(heightB))
    dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxW, maxH))
```

---

## 2. Image Processing & Filtering

### 2.1 Smoothing & Blurring

```python
import cv2
import numpy as np

img = cv2.imread("image.jpg")

# Average blur
avg = cv2.blur(img, (5,5))

# Gaussian blur (most common, removes high-freq noise)
gauss = cv2.GaussianBlur(img, (5,5), sigmaX=0)

# Median blur (removes salt-and-pepper noise, preserves edges)
median = cv2.medianBlur(img, 5)

# Bilateral filter (preserves edges, very slow)
bilateral = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

# Custom kernel convolution
kernel = np.ones((5,5), np.float32) / 25
custom = cv2.filter2D(img, -1, kernel)

# --- Sharpen ---
sharpen_kernel = np.array([[ 0,-1, 0],
                            [-1, 5,-1],
                            [ 0,-1, 0]])
sharp = cv2.filter2D(img, -1, sharpen_kernel)

# Unsharp mask
blurred = cv2.GaussianBlur(img, (0,0), 3)
unsharp = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
```

### 2.2 Edge Detection

```python
import cv2
import numpy as np

gray = cv2.cvtColor(cv2.imread("image.jpg"), cv2.COLOR_BGR2GRAY)

# Sobel (gradient magnitude)
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel  = np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)

# Scharr (better than Sobel for small kernels)
scharrx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
scharry = cv2.Scharr(gray, cv2.CV_64F, 0, 1)

# Laplacian (second derivative, finds all edges)
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))

# Canny (best general-purpose edge detector)
# auto thresholds using Otsu
gray_blur = cv2.GaussianBlur(gray, (5,5), 0)
_, otsu  = cv2.threshold(gray_blur, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
high_t   = int(otsu)
low_t    = int(0.4 * high_t)
canny    = cv2.Canny(gray_blur, low_t, high_t)

# --- Contour detection ---
contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)
# Draw all contours
canvas = img.copy()
cv2.drawContours(canvas, contours, -1, (0,255,0), 2)

# Filter by area
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 500:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(canvas, (x,y), (x+w,y+h), (255,0,0), 2)
        # Moments & centroid
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
        # Bounding shapes
        (mx,my),(ma,mb),angle = cv2.minAreaRect(cnt)
        (ex,ey),er = cv2.minEnclosingCircle(cnt)
        hull = cv2.convexHull(cnt)
        epsilon = 0.02*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
```

### 2.3 Morphological Operations

```python
import cv2
import numpy as np

gray = cv2.cvtColor(cv2.imread("img.jpg"), cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

kernel = np.ones((5,5), np.uint8)
kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
kernel_cross   = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))

# Core operations
eroded  = cv2.erode(binary, kernel, iterations=1)    # shrinks bright
dilated = cv2.dilate(binary, kernel, iterations=1)   # grows bright

# Derived
opened  = cv2.morphologyEx(binary, cv2.MORPH_OPEN,    kernel)  # erode→dilate (removes noise)
closed  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,   kernel)  # dilate→erode (fills holes)
grad    = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT,kernel)  # edges
tophat  = cv2.morphologyEx(binary, cv2.MORPH_TOPHAT,  kernel)  # bright spots
blackhat= cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT,kernel)  # dark spots
```

### 2.4 Thresholding

```python
import cv2
import numpy as np

gray = cv2.cvtColor(cv2.imread("img.jpg"), cv2.COLOR_BGR2GRAY)

# Global thresholding
_, thresh_binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
_, thresh_inv    = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
_, thresh_trunc  = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
_, thresh_tozero = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)

# Otsu's thresholding (automatic)
_, otsu = cv2.threshold(gray, 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Adaptive thresholding (good for varying lighting)
adaptive_mean   = cv2.adaptiveThreshold(gray, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
adaptive_gauss  = cv2.adaptiveThreshold(gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# HSV color thresholding (e.g., detect green)
hsv = cv2.cvtColor(cv2.imread("img.jpg"), cv2.COLOR_BGR2HSV)
lower = np.array([40, 40, 40])
upper = np.array([80, 255, 255])
mask_green = cv2.inRange(hsv, lower, upper)
result = cv2.bitwise_and(img, img, mask=mask_green)
```

---

## 3. Feature Detection & Description

### 3.1 Corner Detection

```python
import cv2
import numpy as np

gray = cv2.cvtColor(cv2.imread("img.jpg"), cv2.COLOR_BGR2GRAY)
gray_f = np.float32(gray)

# Harris Corner Detector
harris = cv2.cornerHarris(gray_f, blockSize=2, ksize=3, k=0.04)
harris = cv2.dilate(harris, None)
img[harris > 0.01 * harris.max()] = [0, 0, 255]

# Shi-Tomasi (Good Features to Track)
corners = cv2.goodFeaturesToTrack(gray, maxCorners=100,
                                   qualityLevel=0.01,
                                   minDistance=10)
corners = np.int0(corners)
for c in corners:
    x, y = c.ravel()
    cv2.circle(img, (x,y), 3, (0,255,0), -1)
```

### 3.2 SIFT, ORB, AKAZE, BRISK

```python
import cv2

img1 = cv2.imread("img1.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("img2.jpg", cv2.IMREAD_GRAYSCALE)

# --- SIFT (Scale-Invariant Feature Transform) ---
# Best accuracy, patented (free in OpenCV 4.4+)
sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.04)
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Visualize keypoints
vis = cv2.drawKeypoints(img1, kp1, None,
      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# --- ORB (Oriented FAST + Rotated BRIEF) ---
# Fast, free, good for real-time
orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8)
kp1_orb, des1_orb = orb.detectAndCompute(img1, None)

# --- AKAZE ---
akaze = cv2.AKAZE_create()
kp, des = akaze.detectAndCompute(img1, None)

# --- BRISK ---
brisk = cv2.BRISK_create()
kp, des = brisk.detectAndCompute(img1, None)
```

### 3.3 Feature Matching

```python
import cv2
import numpy as np

# BFMatcher (Brute Force)
# For SIFT/SURF (float descriptors) → L2 norm
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
top_matches = matches[:50]

# For ORB/BRISK (binary descriptors) → Hamming norm
bf_ham = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf_ham.match(des1_orb, des2_orb)

# KNN Matching + Lowe's Ratio Test (best practice)
bf_knn = cv2.BFMatcher(cv2.NORM_L2)
knn_matches = bf_knn.knnMatch(des1, des2, k=2)
good = []
for m, n in knn_matches:
    if m.distance < 0.75 * n.distance:  # Lowe's ratio = 0.75
        good.append(m)

# FLANN Matcher (faster for large sets)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
knn_matches = flann.knnMatch(des1, des2, k=2)

# Draw matches
result = cv2.drawMatches(img1, kp1, img2, kp2,
                          good, None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# --- Homography from matches ---
if len(good) >= 4:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    # H is 3x3 transformation matrix
    h, w = img1.shape
    pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, H)
```

---

## 4. Classical Object Detection

### 4.1 Haar Cascade Classifier

```python
import cv2

# Load pre-trained cascades
face_cascade    = cv2.CascadeClassifier(cv2.data.haarcascades +
                                         'haarcascade_frontalface_default.xml')
eye_cascade     = cv2.CascadeClassifier(cv2.data.haarcascades +
                                         'haarcascade_eye.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                         'haarcascade_profileface.xml')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,    # image pyramid step size
    minNeighbors=5,     # higher = fewer false positives
    minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE
)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.circle(img, (x+ex+ew//2, y+ey+eh//2), ew//2, (0,255,0), 2)
```

### 4.2 HOG + SVM (Histogram of Oriented Gradients)

```python
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC

# --- Built-in OpenCV HOG person detector ---
hog_detector = cv2.HOGDescriptor()
hog_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Detect people
rects, weights = hog_detector.detectMultiScale(
    img,
    winStride=(4, 4),
    padding=(8, 8),
    scale=1.05
)

for (x,y,w,h) in rects:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

# --- Custom HOG feature extraction ---
def extract_hog(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 128))
    features, _ = hog(resized,
                       orientations=9,
                       pixels_per_cell=(8,8),
                       cells_per_block=(2,2),
                       visualize=True,
                       feature_vector=True)
    return features

# Training pipeline (conceptual)
# X_train = [extract_hog(img) for img in train_images]
# y_train = [1, 0, 1, ...]  # 1=person, 0=background
# svm = LinearSVC(C=0.01)
# svm.fit(X_train, y_train)
```

### 4.3 Template Matching

```python
import cv2
import numpy as np

img = cv2.imread("scene.jpg", cv2.IMREAD_GRAYSCALE)
template = cv2.imread("template.jpg", cv2.IMREAD_GRAYSCALE)
h, w = template.shape

# Methods: TM_CCOEFF_NORMED (best), TM_CCORR_NORMED, TM_SQDIFF_NORMED
result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Single match
top_left = max_loc
bottom_right = (top_left[0]+w, top_left[1]+h)
cv2.rectangle(img, top_left, bottom_right, 255, 2)

# Multiple matches
threshold = 0.8
locations = np.where(result >= threshold)
for pt in zip(*locations[::-1]):
    cv2.rectangle(img, pt, (pt[0]+w, pt[1]+h), 0, 2)

# Multi-scale template matching
def multi_scale_match(img, template, scales=np.linspace(0.2, 1.0, 20)):
    found = None
    for scale in scales:
        resized = cv2.resize(template, (0,0), fx=scale, fy=scale)
        if resized.shape[0] > img.shape[0] or resized.shape[1] > img.shape[1]:
            continue
        result = cv2.matchTemplate(img, resized, cv2.TM_CCOEFF_NORMED)
        _, maxVal, _, maxLoc = cv2.minMaxLoc(result)
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, scale)
    return found
```

---

## 5. Deep Learning for CV — Core Concepts

### 5.1 PyTorch Image Basics

```python
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

# --- PIL → Tensor ---
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),                             # → [C, H, W], float [0,1]
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])   # ImageNet normalization
])
img = Image.open("img.jpg").convert("RGB")
tensor = transform(img)          # (3, 224, 224)
batch  = tensor.unsqueeze(0)     # (1, 3, 224, 224)

# --- Tensor → NumPy/PIL ---
def tensor_to_numpy(tensor):
    # unnormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    t = tensor * std + mean
    return t.permute(1,2,0).numpy()        # (H, W, 3)

def tensor_to_pil(tensor):
    return T.ToPILImage()(tensor.clamp(0,1))

# --- Common transforms ---
train_transform = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.RandomRotation(degrees=15),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# --- GPU handling ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
batch = batch.to(device)
```

### 5.2 DataLoader Setup

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from PIL import Image
import os, glob

# --- Custom Dataset ---
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.files     = glob.glob(os.path.join(root_dir, "**/*.jpg"), recursive=True)
        self.labels    = [int(f.split("/")[-2]) for f in self.files]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# --- Standard DataLoaders ---
train_ds = datasets.ImageFolder("data/train", transform=train_transform)
val_ds   = datasets.ImageFolder("data/val",   transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                           num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False,
                           num_workers=4, pin_memory=True)

# Check class mapping
print(train_ds.class_to_idx)

# --- Weighted sampler (class imbalance) ---
from torch.utils.data import WeightedRandomSampler
class_counts = [100, 500, 250]  # samples per class
weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
sample_weights = weights[train_ds.targets]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
loader_balanced = DataLoader(train_ds, batch_size=32, sampler=sampler)
```

### 5.3 Training Loop Template

```python
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100. * correct / total

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100. * correct / total

# Full training
model     = MyModel().to(device)
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler    = torch.cuda.amp.GradScaler()  # mixed precision

best_acc = 0
for epoch in range(num_epochs):
    # Mixed precision training
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

    val_loss, val_acc = validate(model, val_loader, criterion, device)
    scheduler.step()

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")

    print(f"Epoch {epoch+1}: Val Loss={val_loss:.4f} Acc={val_acc:.2f}%")
```

---

## 6. Convolutional Neural Networks (CNNs)

### 6.1 CNN Building Blocks

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Convolutional Layer ---
# nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups, dilation)

# Standard convolution
conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)  # same padding

# Depthwise separable convolution (MobileNet style)
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise  = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch)
        self.pointwise  = nn.Conv2d(in_ch, out_ch, 1)
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

# Dilated convolution (large receptive field, no downsampling)
dilated = nn.Conv2d(64, 64, 3, padding=2, dilation=2)

# Transposed convolution (upsampling)
deconv = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)  # 2x upscale

# --- Normalization ---
bn  = nn.BatchNorm2d(64)    # most common
gn  = nn.GroupNorm(8, 64)   # good for small batches
ln  = nn.LayerNorm([64,28,28])
in_ = nn.InstanceNorm2d(64) # style transfer

# --- Activation Functions ---
relu    = nn.ReLU(inplace=True)
lrelu   = nn.LeakyReLU(0.1, inplace=True)
gelu    = nn.GELU()         # transformers
silu    = nn.SiLU()         # Swish, modern CNNs
mish    = nn.Mish()         # YOLO models
prelu   = nn.PReLU()        # learnable slope

# --- Pooling ---
maxpool = nn.MaxPool2d(2, 2)
avgpool = nn.AvgPool2d(2, 2)
gadap   = nn.AdaptiveAvgPool2d((1, 1))   # global avg pool → (B,C,1,1)
gmax    = nn.AdaptiveMaxPool2d((1, 1))

# --- Standard CNN Block ---
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)
```

### 6.2 ResNet Building Blocks

```python
import torch.nn as nn

# Basic Residual Block (ResNet-18/34)
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)

# Bottleneck Block (ResNet-50/101/152)
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(out_ch, out_ch*4, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_ch*4)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample:
            identity = self.downsample(x)
        return self.relu(out + identity)
```

### 6.3 Loading Pretrained Models

```python
import torch
import torchvision.models as models

# --- Load pretrained models ---
resnet50  = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
effnet_b0 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
vit_b16   = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
convnext  = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
swin      = models.swin_b(weights=models.Swin_B_Weights.DEFAULT)

# --- Modify for custom classes ---
num_classes = 10
# ResNet
resnet50.fc = nn.Linear(resnet50.fc.in_features, num_classes)

# EfficientNet
effnet_b0.classifier[1] = nn.Linear(
    effnet_b0.classifier[1].in_features, num_classes)

# ViT
vit_b16.heads.head = nn.Linear(vit_b16.heads.head.in_features, num_classes)

# --- Feature extraction (freeze backbone) ---
for param in resnet50.parameters():
    param.requires_grad = False
for param in resnet50.fc.parameters():
    param.requires_grad = True  # only train head

# --- timm library (500+ pretrained models) ---
import timm

model = timm.create_model("convnext_base", pretrained=True, num_classes=10)
model = timm.create_model("efficientnet_b4", pretrained=True, num_classes=10)
model = timm.create_model("vit_large_patch16_224", pretrained=True, num_classes=10)
model = timm.create_model("swin_large_patch4_window7_224", pretrained=True, num_classes=10)

# Get data config for a timm model
data_config = timm.data.resolve_model_data_config(model)
transform = timm.data.create_transform(**data_config, is_training=False)
```

---

## 7. Object Detection — Deep Learning

### 7.1 Detection Concepts

| Concept | Description |
|---|---|
| **Anchor boxes** | Pre-defined bounding box shapes/scales |
| **IoU** | Intersection over Union — overlap metric |
| **NMS** | Non-Maximum Suppression — remove duplicate boxes |
| **FPN** | Feature Pyramid Network — multi-scale features |
| **RPN** | Region Proposal Network (Faster RCNN) |
| **One-stage** | YOLO, SSD, RetinaNet — direct regression |
| **Two-stage** | RCNN family — propose then classify |
| **Anchor-free** | FCOS, CenterNet, DETR — no anchor boxes |

```python
# IoU calculation
def compute_iou(box1, box2):
    """box format: [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2-x1) * max(0, y2-y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - intersection
    return intersection / (union + 1e-6)

# Non-Maximum Suppression
def nms(boxes, scores, iou_threshold=0.5):
    """boxes: (N,4) [x1,y1,x2,y2], scores: (N,)"""
    import numpy as np
    indices = np.argsort(scores)[::-1]
    keep = []
    while len(indices):
        i = indices[0]
        keep.append(i)
        ious = np.array([compute_iou(boxes[i], boxes[j])
                         for j in indices[1:]])
        indices = indices[1:][ious < iou_threshold]
    return keep

# Soft-NMS (keeps overlapping boxes with reduced scores)
# torchvision.ops.nms, batched_nms available
import torchvision
boxes_t  = torch.tensor(boxes, dtype=torch.float32)
scores_t = torch.tensor(scores, dtype=torch.float32)
keep     = torchvision.ops.nms(boxes_t, scores_t, iou_threshold=0.5)
```

### 7.2 YOLOv8 (Ultralytics)

```python
from ultralytics import YOLO
import cv2

# --- Inference ---
model = YOLO("yolov8n.pt")   # n=nano, s=small, m=medium, l=large, x=xlarge
# Also: yolov8n-seg.pt, yolov8n-pose.pt, yolov8n-obb.pt

results = model("image.jpg")
results = model("video.mp4", stream=True)  # streaming for video

for r in results:
    boxes  = r.boxes.xyxy.cpu().numpy()    # [x1,y1,x2,y2]
    confs  = r.boxes.conf.cpu().numpy()    # confidence scores
    cls    = r.boxes.cls.cpu().numpy()     # class indices
    names  = r.names                        # {0: 'person', ...}
    annotated = r.plot()                   # draw boxes
    cv2.imshow("Result", annotated)

# --- Training ---
model = YOLO("yolov8n.pt")
results = model.train(
    data="coco128.yaml",       # dataset config
    epochs=100,
    imgsz=640,
    batch=16,
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    device="0",                # GPU id
    workers=8,
    project="runs/detect",
    name="exp1",
    augment=True,
    mixup=0.1,
    copy_paste=0.1,
    degrees=10.0,
    scale=0.5
)

# --- Validation ---
metrics = model.val()
print(metrics.box.map)       # mAP50-95
print(metrics.box.map50)     # mAP50

# --- Export ---
model.export(format="onnx")     # ONNX
model.export(format="tflite")   # TFLite
model.export(format="engine")   # TensorRT

# --- Custom dataset YAML ---
# data.yaml
# path: /path/to/dataset
# train: images/train
# val: images/val
# nc: 3  (number of classes)
# names: ['cat', 'dog', 'bird']
```

### 7.3 Faster RCNN / Mask RCNN (torchvision)

```python
import torch
import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
    maskrcnn_resnet50_fpn,
    retinanet_resnet50_fpn,
    ssdlite320_mobilenet_v3_large
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# --- Inference ---
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
model.eval()
transform = torchvision.transforms.ToTensor()
img_t = transform(pil_img).unsqueeze(0)
with torch.no_grad():
    predictions = model(img_t)

pred = predictions[0]
boxes  = pred['boxes'].numpy()   # shape (N,4)
labels = pred['labels'].numpy()  # shape (N,)
scores = pred['scores'].numpy()  # shape (N,)

# Filter by confidence
keep = scores > 0.5

# --- Custom Faster RCNN ---
def get_faster_rcnn(num_classes):
    model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# --- Custom Mask RCNN ---
def get_mask_rcnn(num_classes):
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    # Box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # Mask predictor
    in_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, num_classes)
    return model

# --- Detection training loop ---
model = get_faster_rcnn(num_classes=4).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005,
                             momentum=0.9, weight_decay=0.0005)
# Target format for training:
# [{'boxes': tensor([[x1,y1,x2,y2],...]), 'labels': tensor([1,2,...])}]
model.train()
for images, targets in train_loader:
    images  = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
    loss_dict = model(images, targets)
    losses = sum(loss_dict.values())
    optimizer.zero_grad(); losses.backward(); optimizer.step()
```

### 7.4 DETR — Detection Transformer

```python
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests

# Load model
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model     = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Inference
image = Image.open("image.jpg")
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# Post-process
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(
    outputs, threshold=0.9, target_sizes=target_sizes)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    print(f"{model.config.id2label[label.item()]}: {score:.3f} @ {box.tolist()}")
```

---

## 8. Image Segmentation

### 8.1 Segmentation Types

| Type | Description | Output | Example Models |
|---|---|---|---|
| **Semantic** | Class per pixel, no instances | `(H,W)` label map | FCN, DeepLab, SegFormer |
| **Instance** | Each object separately | Mask per instance | Mask RCNN, SOLO |
| **Panoptic** | Semantic + Instance unified | `(H,W)` + instance IDs | Panoptic FPN, MaskFormer |
| **Interactive** | User prompts | SAM, SAM2 |

### 8.2 Semantic Segmentation

```python
# --- torchvision FCN / DeepLab ---
from torchvision.models.segmentation import (
    fcn_resnet101,
    deeplabv3_resnet101,
    deeplabv3_mobilenet_v3_large,
    lraspp_mobilenet_v3_large
)
import torch
import torchvision.transforms as T

model = deeplabv3_resnet101(weights="DEFAULT")
model.eval()

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
img_t = transform(pil_img).unsqueeze(0)
with torch.no_grad():
    output = model(img_t)['out']          # (1, num_classes, H, W)
    pred   = output.argmax(1).squeeze()   # (H, W)

# Colorize
palette = torch.randint(0, 256, (21, 3), dtype=torch.uint8)
color_map = palette[pred]  # (H, W, 3)

# --- SegFormer (HuggingFace) ---
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import torch, torch.nn.functional as F

processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
model     = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")

image = Image.open("img.jpg")
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits                            # (1, num_labels, H/4, W/4)
upsampled = F.interpolate(logits, size=image.size[::-1],
                           mode="bilinear", align_corners=False)
pred = upsampled.argmax(dim=1).squeeze().numpy()   # (H, W)
```

### 8.3 SAM (Segment Anything Model)

```python
# pip install segment-anything
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import numpy as np
import cv2

# Load SAM
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam.to(device="cuda")

# --- Prompted prediction (points / boxes) ---
predictor = SamPredictor(sam)
image = cv2.cvtColor(cv2.imread("img.jpg"), cv2.COLOR_BGR2RGB)
predictor.set_image(image)

# Point prompt
point_coords = np.array([[500, 375]])
point_labels = np.array([1])   # 1=foreground, 0=background
masks, scores, logits = predictor.predict(
    point_coords=point_coords,
    point_labels=point_labels,
    multimask_output=True   # returns 3 masks
)
best_mask = masks[scores.argmax()]   # (H, W) bool

# Box prompt
box = np.array([425, 600, 700, 875])  # [x1, y1, x2, y2]
masks, scores, _ = predictor.predict(
    box=box,
    multimask_output=False
)

# --- Automatic mask generation (everything) ---
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.95,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2
)
masks = mask_generator.generate(image)
# masks is list of dicts: {segmentation, area, bbox, predicted_iou, ...}

# Visualize
def show_masks(image, masks):
    colors = np.random.randint(0, 255, (len(masks), 3))
    overlay = image.copy()
    for mask, color in zip(masks, colors):
        overlay[mask['segmentation']] = color
    return cv2.addWeighted(image, 0.5, overlay, 0.5, 0)

# --- SAM2 (2024) ---
# pip install sam2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
sam2 = build_sam2("sam2_hiera_large.yaml", "sam2_hiera_large.pt")
predictor2 = SAM2ImagePredictor(sam2)
```

### 8.4 U-Net Architecture

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, features=[64,128,256,512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()
        self.pool  = nn.MaxPool2d(2, 2)
        for f in features:
            self.downs.append(DoubleConv(in_ch, f)); in_ch = f
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f*2, f, 2, 2))
            self.ups.append(DoubleConv(f*2, f))
        self.final = nn.Conv2d(features[0], out_ch, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x)
        skips.reverse()
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i//2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i+1](x)
        return self.final(x)

# Usage
model  = UNet(in_ch=3, out_ch=1).to(device)
output = model(torch.randn(1,3,572,572).to(device))
# For binary segmentation: nn.Sigmoid() + BCEWithLogitsLoss
# For multi-class:         nn.Softmax(dim=1) + CrossEntropyLoss
```

---

## 9. Image Classification

### 9.1 End-to-End Classification Pipeline

```python
import torch, torchvision, timm
from torchvision import datasets
import torch.nn as nn

# --- Complete pipeline ---
# 1. Data
train_ds = datasets.ImageFolder("data/train", transform=train_transform)
val_ds   = datasets.ImageFolder("data/val",   transform=val_transform)
train_loader = DataLoader(train_ds, 32, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   32, shuffle=False, num_workers=4)
num_classes  = len(train_ds.classes)

# 2. Model
model = timm.create_model("convnext_small", pretrained=True, num_classes=num_classes)
model = model.to(device)

# 3. Loss & optimizer
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# 4. Training
for epoch in range(50):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc     = validate(model, val_loader, criterion, device)
    scheduler.step()
    print(f"Ep {epoch}: Train {train_acc:.1f}% | Val {val_acc:.1f}%")

# 5. Inference
model.eval()
def predict_image(model, pil_img, class_names):
    tensor = val_transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        topk   = probs.topk(5)
    return [(class_names[i], p.item()) for i, p in zip(topk.indices, topk.values)]
```

### 9.2 Architecture Comparison Table

| Model | Top-1 (ImgNet) | Params | Speed | Notes |
|---|---|---|---|---|
| ResNet-50 | 76.1% | 25M | Fast | Classic baseline |
| ResNet-101 | 77.4% | 45M | Medium | |
| EfficientNet-B0 | 77.1% | 5.3M | Fast | Smallest efficient |
| EfficientNet-B7 | 84.3% | 66M | Slow | |
| ConvNeXt-S | 83.1% | 50M | Fast | Modern CNN |
| ConvNeXt-B | 83.8% | 89M | Medium | |
| ViT-B/16 | 81.1% | 86M | Medium | Pure transformer |
| ViT-L/16 | 82.6% | 307M | Slow | |
| Swin-B | 83.5% | 88M | Fast | Window attention |
| DeiT-B | 81.8% | 86M | Medium | Distilled ViT |
| EfficientNetV2-M | 85.1% | 54M | Fast | 2021 best |
| MaxViT-B | 86.7% | 120M | Medium | 2022 SOTA |

---

## 10. Object Tracking

### 10.1 OpenCV Trackers

```python
import cv2

# Available trackers
TRACKER_TYPES = {
    'CSRT':     cv2.TrackerCSRT_create,        # Accurate, slower
    'KCF':      cv2.TrackerKCF_create,         # Fast, less accurate
    'MOSSE':    cv2.legacy.TrackerMOSSE_create, # Fastest
    'MedianFlow': cv2.legacy.TrackerMedianFlow_create,
    'GOTURN':   cv2.TrackerGOTURN_create,      # DNN-based
    'DaSiamRPN': cv2.TrackerDaSiamRPN_create,  # Best accuracy
    'Nano':     cv2.TrackerNano_create,
    'Vit':      cv2.TrackerVit_create,         # ViT-based
}

tracker = cv2.TrackerCSRT_create()
cap = cv2.VideoCapture("video.mp4")
ret, frame = cap.read()

# Select ROI
bbox = cv2.selectROI("Select", frame, False)
tracker.init(frame, bbox)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    success, bbox = tracker.update(frame)
    if success:
        x,y,w,h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
    else:
        cv2.putText(frame, "LOST", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) == ord('q'): break

# Multi-object tracking
multi_tracker = cv2.legacy.MultiTracker_create()
for bbox in bboxes:
    multi_tracker.add(cv2.TrackerCSRT_create(), frame, bbox)
success, boxes = multi_tracker.update(frame)
```

### 10.2 SORT & DeepSORT

```python
# pip install filterpy scikit-image lap
# SORT: Simple Online and Realtime Tracking
from sort import Sort

tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

cap = cv2.VideoCapture("video.mp4")
detector = YOLO("yolov8n.pt")

while True:
    ret, frame = cap.read()
    if not ret: break
    results = detector(frame, verbose=False)[0]
    detections = results.boxes.data.cpu().numpy()  # [x1,y1,x2,y2,conf,cls]
    dets_xyxy = detections[:, :5]  # [x1,y1,x2,y2,conf]

    tracked = tracker.update(dets_xyxy)  # returns [x1,y1,x2,y2,id]
    for *xyxy, track_id in tracked:
        x1,y1,x2,y2 = [int(v) for v in xyxy]
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"ID:{int(track_id)}", (x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

# DeepSORT adds appearance features (ReID)
# pip install deep-sort-realtime
from deep_sort_realtime.deepsort_tracker import DeepSort
tracker_ds = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0,
                       max_cosine_distance=0.3, nn_budget=None,
                       embedder="mobilenet", half=True)
tracks = tracker_ds.update_tracks(detections_list, frame=frame)
# detections_list: list of ([x1,y1,w,h], confidence, class)
```

### 10.3 ByteTrack / BoTrack (State-of-Art)

```python
# pip install boxmot
from boxmot import ByteTrack, BoTrack, StrongSORT, OcSort

tracker = ByteTrack()
# or
tracker = BoTrack()

while True:
    ret, frame = cap.read()
    if not ret: break
    results = detector(frame)[0]
    dets = results.boxes.data.cpu()  # N x (x1,y1,x2,y2,conf,cls)

    tracks = tracker.update(dets, frame)
    # tracks: N x (x1,y1,x2,y2,id,conf,cls,idx)
    for track in tracks:
        x1,y1,x2,y2 = track[:4].astype(int)
        tid = int(track[4])
        cv2.rectangle(frame, (x1,y1),(x2,y2),(255,0,0),2)
        cv2.putText(frame, f"ID {tid}", (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0),2)
```

---

## 11. Pose Estimation

### 11.1 YOLOv8 Pose

```python
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n-pose.pt")

# COCO keypoints (17 points):
# 0:nose 1:left_eye 2:right_eye 3:left_ear 4:right_ear
# 5:left_shoulder 6:right_shoulder 7:left_elbow 8:right_elbow
# 9:left_wrist 10:right_wrist 11:left_hip 12:right_hip
# 13:left_knee 14:right_knee 15:left_ankle 16:right_ankle

SKELETON = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),
            (8,10),(11,12),(5,11),(6,12),(11,13),(12,14),(13,15),(14,16)]

results = model("video.mp4", stream=True)
for r in results:
    kps = r.keypoints.xy.cpu().numpy()  # (num_people, 17, 2)
    confs = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None
    frame = r.orig_img.copy()
    for person_kps in kps:
        for i, (x,y) in enumerate(person_kps):
            if x > 0 and y > 0:
                cv2.circle(frame, (int(x),int(y)), 4, (0,255,0), -1)
        for a, b in SKELETON:
            xa,ya = int(person_kps[a][0]), int(person_kps[a][1])
            xb,yb = int(person_kps[b][0]), int(person_kps[b][1])
            if xa>0 and ya>0 and xb>0 and yb>0:
                cv2.line(frame, (xa,ya),(xb,yb),(255,0,0),2)
```

### 11.2 MediaPipe Pose / Hands / Face

```python
import mediapipe as mp
import cv2

# --- Pose ---
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

with mp_pose.Pose(min_detection_confidence=0.7,
                  min_tracking_confidence=0.7,
                  model_complexity=1) as pose:
    frame = cv2.cvtColor(cv2.imread("img.jpg"), cv2.COLOR_BGR2RGB)
    results = pose.process(frame)
    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks,
                                mp_pose.POSE_CONNECTIONS)
        # Access landmarks
        lm = results.pose_landmarks.landmark
        left_shoulder  = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        print(f"LS: ({left_shoulder.x:.2f}, {left_shoulder.y:.2f})")

# --- Hands ---
mp_hands = mp.solutions.hands
with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
    results = hands.process(frame)
    if results.multi_hand_landmarks:
        for hand_lm in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
            # 21 landmarks: 0=WRIST, 1-4=THUMB, 5-8=INDEX, ...
            wrist = hand_lm.landmark[mp_hands.HandLandmark.WRIST]

# --- Face Mesh ---
mp_face_mesh = mp.solutions.face_mesh
with mp_face_mesh.FaceMesh(max_num_faces=2,
                             refine_landmarks=True,
                             min_detection_confidence=0.7) as face_mesh:
    results = face_mesh.process(frame)
    if results.multi_face_landmarks:
        for face_lm in results.multi_face_landmarks:
            mp_draw.draw_landmarks(frame, face_lm,
                                    mp_face_mesh.FACEMESH_TESSELATION)
            # 478 landmarks (with refined iris landmarks)

# --- Holistic (Pose+Hands+Face together) ---
mp_holistic = mp.solutions.holistic
with mp_holistic.Holistic(min_detection_confidence=0.5) as holistic:
    results = holistic.process(frame)
    pose_lm  = results.pose_landmarks
    left_lm  = results.left_hand_landmarks
    right_lm = results.right_hand_landmarks
    face_lm  = results.face_landmarks
```

### 11.3 Angle Calculation & Gesture Recognition

```python
import numpy as np
import math

def calc_angle(a, b, c):
    """Angle at point b, given points a, b, c as (x,y)"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
              np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def calc_distance(p1, p2):
    return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

# Rep counter example (bicep curl)
counter, stage = 0, None
def count_rep(angle):
    global counter, stage
    if angle < 30:
        stage = "up"
    if angle > 160 and stage == "up":
        stage = "down"; counter += 1
    return counter, stage

# Finger counting (hand landmarks)
def count_fingers(hand_landmarks):
    fingers = []
    tip_ids = [4, 8, 12, 16, 20]  # thumb tip, index, middle, ring, pinky
    # Thumb (check x)
    fingers.append(1 if hand_landmarks[4][0] > hand_landmarks[3][0] else 0)
    # Other fingers (check y — lower y = higher on image = extended)
    for i in range(1, 5):
        tip = hand_landmarks[tip_ids[i]][1]
        mid = hand_landmarks[tip_ids[i]-2][1]
        fingers.append(1 if tip < mid else 0)
    return sum(fingers)
```

---

## 12. Face Recognition & Analysis

### 12.1 Face Detection

```python
import cv2

# --- OpenCV DNN face detector (more accurate than Haar) ---
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

def detect_faces_dnn(img, conf_thresh=0.5):
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300,300),
                                   (104.0,177.0,123.0))
    net.setInput(blob)
    dets = net.forward()
    faces = []
    for i in range(dets.shape[2]):
        conf = dets[0,0,i,2]
        if conf > conf_thresh:
            box = dets[0,0,i,3:7] * np.array([w,h,w,h])
            x1,y1,x2,y2 = box.astype(int)
            faces.append((x1,y1,x2-x1,y2-y1,conf))
    return faces

# --- MTCNN ---
from mtcnn import MTCNN
detector = MTCNN()
results = detector.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
for r in results:
    x,y,w,h = r['box']
    keypoints = r['keypoints']  # left_eye, right_eye, nose, ...
    confidence = r['confidence']

# --- RetinaFace ---
from retinaface import RetinaFace
faces = RetinaFace.detect_faces("img.jpg")
# Returns dict with 'face_1', 'face_2', ...
```

### 12.2 Face Recognition

```python
# --- face_recognition library ---
import face_recognition
import numpy as np
import cv2

# Encode known faces
known_image = face_recognition.load_image_file("known.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# Compare
unknown_image = face_recognition.load_image_file("unknown.jpg")
unknown_encodings = face_recognition.face_encodings(unknown_image)
for enc in unknown_encodings:
    results = face_recognition.compare_faces([known_encoding], enc, tolerance=0.6)
    distance = face_recognition.face_distance([known_encoding], enc)
    print(f"Match: {results[0]}, Distance: {distance[0]:.4f}")

# Real-time recognition
known_encodings_db = []  # preloaded encodings
known_names_db     = []  # corresponding names

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb, model="hog")  # or "cnn"
    face_encodings = face_recognition.face_encodings(rgb, face_locations)
    for (top,right,bottom,left), enc in zip(face_locations, face_encodings):
        dists = face_recognition.face_distance(known_encodings_db, enc)
        name  = known_names_db[dists.argmin()] if dists.min() < 0.5 else "Unknown"
        cv2.rectangle(frame,(left,top),(right,bottom),(0,255,0),2)
        cv2.putText(frame, name,(left,top-10), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
    cv2.imshow("Recognition", frame)
    if cv2.waitKey(1) == ord('q'): break

# --- DeepFace (multiple backends) ---
from deepface import DeepFace

# Verification
result = DeepFace.verify("img1.jpg", "img2.jpg",
                          model_name="ArcFace",  # Facenet, VGG-Face, ArcFace, Dlib
                          detector_backend="retinaface")
print(result["verified"], result["distance"])

# Recognition (find in database)
dfs = DeepFace.find("query.jpg", db_path="face_db/",
                     model_name="ArcFace", enforce_detection=False)

# Analysis
obj = DeepFace.analyze("img.jpg", actions=["age","gender","race","emotion"])
print(obj[0])  # {'age': 25, 'gender': 'Man', 'dominant_race': ...}
```

### 12.3 Face Alignment & Anti-Spoofing

```python
# Face alignment using 5 landmarks
def align_face(image, landmarks, output_size=(112, 112)):
    """landmarks: dict with left_eye, right_eye, nose, mouth_left, mouth_right"""
    dst_pts = np.float32([
        [38.29, 51.70],[73.53, 51.50],[56.02, 71.74],
        [41.55, 92.37],[70.72, 92.20]
    ])
    src_pts = np.float32([landmarks["left_eye"], landmarks["right_eye"],
                           landmarks["nose"],     landmarks["mouth_left"],
                           landmarks["mouth_right"]])
    M = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]
    aligned = cv2.warpAffine(image, M, output_size)
    return aligned

# Face quality & liveness check (basic)
def check_face_quality(face_roi):
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)
    return {
        "sharpness": laplacian_var,
        "too_blurry": laplacian_var < 100,
        "brightness": brightness,
        "too_dark": brightness < 40,
        "too_bright": brightness > 220
    }
```

---

## 13. Optical Flow & Motion Analysis

### 13.1 Dense Optical Flow

```python
import cv2
import numpy as np

cap = cv2.VideoCapture("video.mp4")
ret, prev = cap.read()
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(prev)
hsv[..., 1] = 255

while True:
    ret, curr = cap.read()
    if not ret: break
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    # Farneback dense optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2,
        flags=0
    )
    # flow shape: (H, W, 2) — x and y flow

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2     # hue = direction
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Motion mask (moving objects)
    motion_mask = (mag > 2).astype(np.uint8) * 255
    prev_gray = curr_gray
    cv2.imshow("Flow", rgb)
    if cv2.waitKey(30) == ord('q'): break
```

### 13.2 Sparse Optical Flow (Lucas-Kanade)

```python
import cv2
import numpy as np

cap = cv2.VideoCapture("video.mp4")
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Detect good features to track
p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=200, qualityLevel=0.01,
                               minDistance=7, blockSize=7)

lk_params = dict(winSize=(15,15), maxLevel=2,
                  criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 10, 0.03))

mask = np.zeros_like(old_frame)  # for drawing trails
colors = np.random.randint(0, 255, (200, 3))

while True:
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Track features
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)
    good_new = p1[st==1]
    good_old = p0[st==1]

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel().astype(int)
        c, d = old.ravel().astype(int)
        mask  = cv2.line(mask, (a,b),(c,d), colors[i%200].tolist(), 2)
        frame = cv2.circle(frame, (a,b), 4, colors[i%200].tolist(), -1)

    output = cv2.add(frame, mask)
    cv2.imshow("LK Flow", output)
    if cv2.waitKey(30) == ord('q'): break
    old_gray = gray.copy()
    p0 = good_new.reshape(-1,1,2)

    # Periodically re-detect features
    if len(p0) < 50:
        p0 = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01,
                                       minDistance=7, blockSize=7)
        mask = np.zeros_like(old_frame)
```

### 13.3 Background Subtraction

```python
import cv2

# MOG2 (Mixture of Gaussians 2) — best general purpose
mog2 = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=16, detectShadows=True)

# KNN (K-Nearest Neighbors)
knn = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400.0)

cap = cv2.VideoCapture("video.mp4")
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

while True:
    ret, frame = cap.read()
    if not ret: break
    fg_mask = mog2.apply(frame)             # 0=bg, 127=shadow, 255=fg
    # Remove shadows
    fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]
    # Clean up noise
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Find moving objects
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("Motion", frame)
    if cv2.waitKey(30) == ord('q'): break
```

---

## 14. Depth Estimation & 3D Vision

### 14.1 Stereo Vision

```python
import cv2
import numpy as np

# Camera calibration (get intrinsics & extrinsics)
# Calibration with chessboard
objpoints = []  # 3D real world points
imgpoints_l = []  # 2D image points (left camera)
imgpoints_r = []  # 2D image points (right camera)

chessboard = (9, 6)  # inner corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((chessboard[0]*chessboard[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1,2)

for fname_l, fname_r in image_pairs:
    gray_l = cv2.cvtColor(cv2.imread(fname_l), cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(cv2.imread(fname_r), cv2.COLOR_BGR2GRAY)
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, chessboard, None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, chessboard, None)
    if ret_l and ret_r:
        objpoints.append(objp)
        corners_l = cv2.cornerSubPix(gray_l, corners_l, (11,11), (-1,-1), criteria)
        corners_r = cv2.cornerSubPix(gray_r, corners_r, (11,11), (-1,-1), criteria)
        imgpoints_l.append(corners_l)
        imgpoints_r.append(corners_r)

# Calibrate each camera
_, K_l, D_l, _, _ = cv2.calibrateCamera(objpoints, imgpoints_l, gray_l.shape[::-1], None, None)
_, K_r, D_r, _, _ = cv2.calibrateCamera(objpoints, imgpoints_r, gray_r.shape[::-1], None, None)

# Stereo calibration
_, K_l, D_l, K_r, D_r, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_l, imgpoints_r, K_l, D_l, K_r, D_r,
    gray_l.shape[::-1], flags=cv2.CALIB_FIX_INTRINSIC)

# Rectification
R_l,R_r,P_l,P_r,Q,_,_ = cv2.stereoRectify(K_l,D_l,K_r,D_r,
                                              gray_l.shape[::-1],R,T)

# Disparity computation
stereo = cv2.StereoBM_create(numDisparities=96, blockSize=11)
# Or better:
stereo = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=112, blockSize=5,
    P1=8*3*5**2, P2=32*3*5**2, disp12MaxDiff=1,
    uniquenessRatio=10, speckleWindowSize=100, speckleRange=32)
disparity = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0

# Depth from disparity: depth = (focal * baseline) / disparity
depth = cv2.reprojectImageTo3D(disparity, Q)
```

### 14.2 Monocular Depth Estimation

```python
# --- MiDaS (Intel) ---
import torch
from torchvision import transforms

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")  # or DPT_Large
midas.eval().to(device)

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform  # or dpt_transform

img = cv2.cvtColor(cv2.imread("img.jpg"), cv2.COLOR_BGR2RGB)
input_batch = transform(img).to(device)
with torch.no_grad():
    prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1), size=img.shape[:2],
        mode="bicubic", align_corners=False).squeeze()
depth_map = prediction.cpu().numpy()

# --- Depth Anything V2 (2024 SOTA) ---
from transformers import pipeline
pipe = pipeline(task="depth-estimation",
                model="depth-anything/Depth-Anything-V2-Large-hf")
depth = pipe("img.jpg")["depth"]

# --- ZoeDepth (metric depth) ---
repo = "isl-org/ZoeDepth"
model = torch.hub.load(repo, "ZoeD_NK", pretrained=True).to(device).eval()
from zoedepth.utils.misc import pil_to_batched_tensor
img_pil = Image.open("img.jpg").convert("RGB")
X = pil_to_batched_tensor(img_pil).to(device)
depth_metric = model.infer(X)  # actual meters
```

### 14.3 Point Cloud & 3D Reconstruction

```python
import open3d as o3d
import numpy as np

# Load/create point cloud
pcd = o3d.io.read_point_cloud("cloud.ply")
pcd = o3d.io.read_point_cloud("cloud.pcd")

# From depth + color
depth = o3d.io.read_image("depth.png")
color = o3d.io.read_image("color.jpg")
rgbd  = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color, depth, depth_scale=1000.0, depth_trunc=3.0,
    convert_rgb_to_intensity=False)
camera = o3d.camera.PinholeCameraIntrinsic(640, 480, 525, 525, 319.5, 239.5)
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera)

# Basic processing
pcd = pcd.voxel_down_sample(voxel_size=0.02)     # downsample
pcd, inliers = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                      radius=0.1, max_nn=30))

# RANSAC plane segmentation
plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                          ransac_n=3, num_iterations=1000)
plane_pcd = pcd.select_by_index(inliers)
rest_pcd  = pcd.select_by_index(inliers, invert=True)

# Clustering
labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=10))
max_label = labels.max()
colors = plt.get_cmap("tab20")(labels/(max_label if max_label>0 else 1))
pcd.colors = o3d.utility.Vector3dVector(colors[:,:3])

# Mesh reconstruction
pcd.orient_normals_consistent_tangent_plane(100)
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

# Visualize
o3d.visualization.draw_geometries([pcd])
```

---

## 15. Image Generation & GANs

### 15.1 GAN Concepts

```python
import torch
import torch.nn as nn

# --- Generator ---
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_size=64, channels=3):
        super().__init__()
        self.net = nn.Sequential(
            # latent → feature map
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128,  64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.ConvTranspose2d( 64, channels, 4, 2, 1, bias=False),
            nn.Tanh()   # output in [-1, 1]
        )
    def forward(self, z):
        return self.net(z.view(-1,100,1,1))

# --- Discriminator ---
class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x).view(-1)

# --- DCGAN Training ---
G = Generator().to(device)
D = Discriminator().to(device)
opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5,0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5,0.999))
bce   = nn.BCELoss()

for real_imgs, _ in dataloader:
    real_imgs = real_imgs.to(device)
    B = real_imgs.size(0)
    real_labels = torch.ones(B).to(device)
    fake_labels = torch.zeros(B).to(device)

    # Train D
    opt_D.zero_grad()
    z    = torch.randn(B, 100).to(device)
    fake = G(z).detach()
    d_real = D(real_imgs); d_fake = D(fake)
    loss_D = bce(d_real, real_labels) + bce(d_fake, fake_labels)
    loss_D.backward(); opt_D.step()

    # Train G
    opt_G.zero_grad()
    z    = torch.randn(B, 100).to(device)
    fake = G(z)
    loss_G = bce(D(fake), real_labels)  # fool D
    loss_G.backward(); opt_G.step()
```

### 15.2 Stable Diffusion & Diffusion Models

```python
# pip install diffusers transformers accelerate
from diffusers import (StableDiffusionPipeline, StableDiffusionImg2ImgPipeline,
                        StableDiffusionInpaintPipeline, ControlNetModel,
                        StableDiffusionControlNetPipeline, DPMSolverMultistepScheduler)
import torch
from PIL import Image

# --- Text-to-Image ---
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()

image = pipe(
    prompt="a futuristic city at night, neon lights, ultra detailed, 8k",
    negative_prompt="blurry, low quality, ugly, bad anatomy",
    height=512, width=512,
    num_inference_steps=20,
    guidance_scale=7.5,
    num_images_per_prompt=4
).images[0]

# --- Image-to-Image ---
img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
init_img = Image.open("sketch.jpg").convert("RGB").resize((512,512))
result = img2img(
    prompt="a beautiful oil painting",
    image=init_img,
    strength=0.75,   # 0=keep original, 1=ignore original
    guidance_scale=7.5
).images[0]

# --- Inpainting ---
inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16).to("cuda")
mask = Image.open("mask.png").convert("RGB")
result = inpaint(prompt="a red sports car", image=init_img,
                  mask_image=mask, num_inference_steps=50).images[0]

# --- SDXL (higher quality) ---
from diffusers import StableDiffusionXLPipeline
pipe_xl = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16, use_safetensors=True).to("cuda")
image = pipe_xl("astronaut in space, photorealistic").images[0]

# --- ControlNet ---
from diffusers.utils import load_image
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe_cn = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet,
    torch_dtype=torch.float16).to("cuda")
control_image = load_image("canny_edge.png")
result = pipe_cn("beautiful landscape", image=control_image,
                  num_inference_steps=20).images[0]
```

---

## 16. Vision Transformers & Modern Architectures

### 16.1 ViT — Vision Transformer

```python
import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    """Split image into patches and embed them."""
    def __init__(self, img_size=224, patch_size=16, in_ch=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, N, D)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim*3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2,0,3,1,4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = self.dropout(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1,2).reshape(B, N, D)
        return self.proj(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim*mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim*mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.blocks = nn.Sequential(*[TransformerBlock(embed_dim, num_heads)
                                       for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B,-1,-1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x[:,0])   # CLS token
        return self.head(x)

# ViT variants:
# ViT-Ti: embed=192, depth=12, heads=3
# ViT-S:  embed=384, depth=12, heads=6
# ViT-B:  embed=768, depth=12, heads=12
# ViT-L:  embed=1024,depth=24, heads=16
# ViT-H:  embed=1280,depth=32, heads=16
```

### 16.2 Modern Architecture Summary

```
ConvNeXt (2022) — pure CNN, ViT-inspired design
├── Depthwise conv with large kernel (7x7)
├── Inverted bottleneck (like MobileNetV2)
├── Layer Scale + Stochastic Depth
└── GELU activation, fewer normalization layers

Swin Transformer (2021)
├── Shifted window attention (linear complexity)
├── Hierarchical feature maps (like CNN)
├── Better for dense prediction tasks
└── W-MSA + SW-MSA alternating

MaxViT (2022)
├── Multi-axis attention: local + global
├── Grid-partitioned self-attention
└── Best accuracy-efficiency trade-off

DeiT (Data-efficient ViT)
├── Knowledge distillation from CNN teacher
├── Distillation token (like CLS token)
└── Trains well without large datasets

EfficientNetV2 (2021)
├── Smaller & faster than EfficientNet
├── Progressive training (small→large resolution)
└── Fused-MBConv for early layers
```

---

## 17. Multimodal Vision-Language Models

### 17.1 CLIP

```python
import torch
import clip
from PIL import Image

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
# Also: "ViT-L/14", "ViT-B/16", "RN50", "RN101"

# --- Zero-shot classification ---
image = preprocess(Image.open("img.jpg")).unsqueeze(0).to(device)
labels = ["a dog", "a cat", "a car", "a tree"]
text   = clip.tokenize(labels).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features  = model.encode_text(text)
    # Normalize
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features  /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

for label, prob in zip(labels, similarity[0]):
    print(f"{label}: {prob:.2%}")

# --- Image retrieval (semantic search) ---
def embed_images(image_paths):
    features = []
    for path in image_paths:
        img = preprocess(Image.open(path)).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(img)
            feat /= feat.norm(dim=-1, keepdim=True)
        features.append(feat)
    return torch.cat(features)

image_features = embed_images(all_image_paths)

def search_by_text(query, image_features, top_k=5):
    text   = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_feat = model.encode_text(text)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
    sims = (image_features @ text_feat.T).squeeze()
    return sims.topk(top_k).indices
```

### 17.2 BLIP-2, LLaVA, Qwen-VL

```python
# --- BLIP-2 (image captioning + VQA) ---
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to("cuda")

image = Image.open("img.jpg").convert("RGB")

# Captioning
inputs = processor(image, return_tensors="pt").to("cuda", torch.float16)
generated = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(generated[0], skip_special_tokens=True))

# VQA
question = "What objects are in this image?"
inputs = processor(image, question, return_tensors="pt").to("cuda", torch.float16)
generated = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(generated[0], skip_special_tokens=True))

# --- LLaVA (Large Language & Vision Assistant) ---
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16).to("cuda")

from transformers import LlavaNextProcessor
prompt = "[INST] <image>\nDescribe this image in detail. [/INST]"
image = Image.open("img.jpg")
inputs = processor(prompt, image, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))

# --- Qwen2-VL (2024 SOTA) ---
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.float16).to("cuda")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

messages = [{"role": "user", "content": [
    {"type": "image", "image": "img.jpg"},
    {"type": "text",  "text": "What is in this image?"}
]}]
text = processor.apply_chat_template(messages, add_generation_prompt=True)
image_inputs, _ = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0]))
```

---

## 18. OCR & Document Understanding

### 18.1 Tesseract OCR

```python
import pytesseract
import cv2
from PIL import Image
import numpy as np

# Configure path (Windows)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

img = cv2.imread("doc.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Preprocess for OCR
def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    # Threshold
    _, thresh = cv2.threshold(enhanced, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

preprocessed = preprocess_for_ocr(img)

# Basic OCR
text = pytesseract.image_to_string(preprocessed,
                                    lang="eng",
                                    config="--psm 6")  # uniform block of text
# PSM modes:
# 3 = fully automatic (default)
# 6 = uniform block of text
# 11 = sparse text
# 13 = raw line

# OCR with bounding boxes
data = pytesseract.image_to_data(preprocessed, output_type=pytesseract.Output.DICT)
for i, word in enumerate(data['text']):
    if data['conf'][i] > 60 and word.strip():
        x,y,w,h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
        cv2.putText(img, word, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)

# OCR with bounding boxes as dataframe
import pandas as pd
df = pytesseract.image_to_data(preprocessed, output_type=pytesseract.Output.DATAFRAME)
df = df[df['conf'] > 60].dropna()
```

### 18.2 EasyOCR & PaddleOCR

```python
# --- EasyOCR ---
import easyocr
import cv2

reader = easyocr.Reader(['en', 'hi'], gpu=True)  # multi-language
results = reader.readtext("image.jpg")
# results: [([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], text, confidence), ...]

img = cv2.imread("image.jpg")
for (bbox, text, conf) in results:
    if conf > 0.5:
        pts = np.array(bbox, np.int32).reshape((-1,1,2))
        cv2.polylines(img, [pts], True, (0,255,0), 2)
        cv2.putText(img, f"{text} ({conf:.2f})", tuple(bbox[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

# --- PaddleOCR (best accuracy) ---
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image

ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=True)
result = ocr.ocr("image.jpg", cls=True)

for line in result[0]:
    bbox = line[0]    # [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
    text = line[1][0]
    conf = line[1][1]
    print(f"'{text}' (conf={conf:.2f})")

# Visualize with PaddleOCR's draw_ocr
image = Image.open("image.jpg").convert("RGB")
boxes  = [line[0] for line in result[0]]
texts  = [line[1][0] for line in result[0]]
scores = [line[1][1] for line in result[0]]
im_show = draw_ocr(image, boxes, texts, scores)
im_show = Image.fromarray(im_show)
```

### 18.3 TrOCR & Document AI

```python
# TrOCR (Transformer OCR — HuggingFace)
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")

image = Image.open("line.jpg").convert("RGB")
pixel_values = processor(image, return_tensors="pt").pixel_values
generated = model.generate(pixel_values)
text = processor.batch_decode(generated, skip_special_tokens=True)[0]
print(text)

# Donut (Document Understanding Transformer) — no OCR needed
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

image = Image.open("document.jpg").convert("RGB")
task_prompt = "<s_docvqa><s_question>{}</s_question><s_answer>".format("What is the total amount?")
decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False,
                                         return_tensors="pt").input_ids
pixel_values = processor(image, return_tensors="pt").pixel_values
outputs = model.generate(pixel_values, decoder_input_ids=decoder_input_ids,
                          max_length=model.decoder.config.max_position_embeddings)
seq = processor.batch_decode(outputs.tolist())[0]
seq = processor.token2json(seq)
print(seq)
```

---

## 19. Medical Imaging

### 19.1 DICOM Handling

```python
import pydicom
import numpy as np
import cv2
from pathlib import Path

# Load DICOM
dcm = pydicom.dcmread("scan.dcm")

# Metadata
print(dcm.PatientName, dcm.StudyDate, dcm.Modality)
print(dcm.Rows, dcm.Columns, dcm.SliceThickness)

# Pixel data → Hounsfield Units (CT)
def dcm_to_hu(dcm):
    pixel = dcm.pixel_array.astype(np.int16)
    # Apply slope/intercept
    slope = float(dcm.RescaleSlope) if hasattr(dcm, 'RescaleSlope') else 1.0
    intercept = float(dcm.RescaleIntercept) if hasattr(dcm, 'RescaleIntercept') else 0.0
    hu = (pixel * slope + intercept).astype(np.int16)
    return hu

# Window/level (W/L) for visualization
def apply_windowing(hu, window_center, window_width):
    lower = window_center - window_width // 2
    upper = window_center + window_width // 2
    clipped = np.clip(hu, lower, upper)
    normalized = (clipped - lower) / (upper - lower)
    return (normalized * 255).astype(np.uint8)

hu = dcm_to_hu(dcm)
# Common windows:
lung_img   = apply_windowing(hu, -600, 1500)   # Lung window
brain_img  = apply_windowing(hu,  40,  80)     # Brain window
bone_img   = apply_windowing(hu,  300, 1500)   # Bone window
abdomen_img= apply_windowing(hu,  60,  400)    # Abdomen window

# Load 3D CT volume
def load_ct_volume(dcm_dir):
    slices = [pydicom.dcmread(p) for p in sorted(Path(dcm_dir).glob("*.dcm"))]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    volume = np.stack([dcm_to_hu(s) for s in slices], axis=0)  # (Z, H, W)
    spacing = [float(slices[0].SliceThickness),
               float(slices[0].PixelSpacing[0]),
               float(slices[0].PixelSpacing[1])]  # mm
    return volume, spacing
```

### 19.2 Medical Image Segmentation

```python
# MONAI (Medical Open Network for AI)
from monai.networks.nets import UNet, UNETR, SwinUNETR
from monai.losses import DiceLoss, DiceCELoss, FocalLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, ScaleIntensityRanged, CropForegroundd,
    RandCropByPosNegLabeld, RandFlipd, RandRotate90d,
    NormalizeIntensityd, ToTensord
)
from monai.data import Dataset, DataLoader, CacheDataset

# Define model
model = SwinUNETR(
    img_size=(96,96,96), in_channels=1, out_channels=14,
    feature_size=48, use_checkpoint=True
).to(device)

# Or standard 3D UNet
model = UNet(
    spatial_dims=3, in_channels=1, out_channels=2,
    channels=(16,32,64,128,256), strides=(2,2,2,2),
    num_res_units=2
).to(device)

# Transforms
train_transforms = Compose([
    LoadImaged(keys=["image","label"]),
    EnsureChannelFirstd(keys=["image","label"]),
    Orientationd(keys=["image","label"], axcodes="RAS"),
    Spacingd(keys=["image","label"], pixdim=(1.0,1.0,1.0),
              mode=("bilinear","nearest")),
    ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250,
                          b_min=0.0, b_max=1.0, clip=True),
    RandCropByPosNegLabeld(keys=["image","label"], label_key="label",
                            spatial_size=(96,96,96), pos=1, neg=1, num_samples=4),
    RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=0),
    RandRotate90d(keys=["image","label"], prob=0.1, max_k=3),
])

loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
metric  = DiceMetric(include_background=False, reduction="mean")
```

---

## 20. Video Understanding

### 20.1 Video Reading & Writing

```python
import cv2

# --- Read video ---
cap = cv2.VideoCapture("video.mp4")
cap = cv2.VideoCapture(0)  # webcam

fps    = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Jump to frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
# Jump to time (ms)
cap.set(cv2.CAP_PROP_POS_MSEC, 5000)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    # process frame ...
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord('q'): break
cap.release()
cv2.destroyAllWindows()

# --- Write video ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'XVID', 'MJPG', 'H264'
out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))
for frame in frames:
    out.write(frame)
out.release()

# --- With PyAV (better codec support) ---
import av
with av.open("video.mp4") as container:
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="bgr24")
        # process img ...
```

### 20.2 Video Classification

```python
# --- VideoMAE / TimeSformer ---
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import torch, numpy as np
from pytorchvideo.data.encoded_video import EncodedVideo

processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

# Sample 16 frames uniformly
def sample_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total-1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

frames = sample_frames("video.mp4", 16)
inputs = processor(frames, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
predicted_label = model.config.id2label[outputs.logits.argmax(-1).item()]
print(predicted_label)

# --- SlowFast ---
import pytorchvideo.models as pv_models
model = pv_models.slowfast.create_slowfast(model_num_class=400)
```

### 20.3 Action Recognition & Temporal Analysis

```python
# Sliding window approach
def classify_actions_sliding_window(video_path, model, window_size=16, stride=8):
    frames = load_all_frames(video_path)
    predictions = []
    for start in range(0, len(frames)-window_size, stride):
        clip = frames[start:start+window_size]
        pred = classify_clip(model, clip)
        predictions.append({
            "start_frame": start,
            "end_frame": start+window_size,
            "action": pred["label"],
            "confidence": pred["score"]
        })
    return predictions

# Activity recognition with LSTM
class ActionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                             batch_first=True, dropout=0.3)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):          # x: (B, T, feature_dim)
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1])  # last step
```

---

## 21. Data Augmentation

### 21.1 Albumentations (Best Library)

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# --- Basic augmentation pipeline ---
train_transform = A.Compose([
    A.RandomResizedCrop(224, 224, scale=(0.5, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
    A.OneOf([
        A.GaussNoise(var_limit=(10,50)),
        A.GaussianBlur(blur_limit=(3,7)),
        A.MotionBlur(blur_limit=7),
        A.MedianBlur(blur_limit=5),
    ], p=0.3),
    A.OneOf([
        A.CLAHE(clip_limit=2),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30),
        A.RandomGamma(gamma_limit=(80,120)),
    ], p=0.5),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))
# formats: 'pascal_voc' [x1,y1,x2,y2], 'coco' [x,y,w,h], 'yolo' [xc,yc,w,h]

# Apply
image = cv2.cvtColor(cv2.imread("img.jpg"), cv2.COLOR_BGR2RGB)
bboxes = [[100, 50, 300, 200], [400, 100, 500, 300]]  # [x1,y1,x2,y2]
category_ids = [0, 1]

transformed = train_transform(image=image, bboxes=bboxes, category_ids=category_ids)
t_image      = transformed["image"]
t_bboxes     = transformed["bboxes"]

# For segmentation (with masks)
seg_transform = A.Compose([
    A.RandomCrop(512, 512),
    A.HorizontalFlip(p=0.5),
    A.Normalize(),
    ToTensorV2()
])
transformed = seg_transform(image=image, mask=mask)

# --- Advanced augmentations ---
heavy_aug = A.Compose([
    A.RandomResizedCrop(224, 224),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32,
                     min_holes=1, fill_value=0, p=0.3),  # Cutout/CoarseDropout
    A.GridDistortion(p=0.2),          # elastic deformation
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
    A.OpticalDistortion(p=0.2),
    A.Perspective(p=0.2),
    A.Affine(shear=(-10,10), p=0.3),
    A.ISONoise(p=0.2),
    A.ImageCompression(quality_lower=75, quality_upper=100, p=0.2),
    A.Solarize(p=0.1),
    A.Equalize(p=0.2),
    A.Sharpen(p=0.2),
    A.Emboss(p=0.1),
    A.Normalize(), ToTensorV2()
])
```

### 21.2 MixUp, CutMix, AugMix

```python
import torch
import numpy as np

# --- MixUp ---
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1-lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1-lam) * criterion(pred, y_b)

# --- CutMix ---
def rand_bbox(size, lam):
    W, H = size[3], size[2]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = np.clip(cx - cut_w//2, 0, W)
    y1 = np.clip(cy - cut_h//2, 0, H)
    x2 = np.clip(cx + cut_w//2, 0, W)
    y2 = np.clip(cy + cut_h//2, 0, H)
    return x1, y1, x2, y2

def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size(0)).to(x.device)
    y_a, y_b = y, y[rand_index]
    x1,y1_,x2,y2_ = rand_bbox(x.size(), lam)
    x[:, :, y1_:y2_, x1:x2] = x[rand_index, :, y1_:y2_, x1:x2]
    lam = 1 - ((x2-x1)*(y2_-y1_) / (x.size(-1)*x.size(-2)))
    return x, y_a, y_b, lam

# --- Label Smoothing ---
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

# --- TTA (Test Time Augmentation) ---
def tta_predict(model, image, n_augments=5):
    model.eval()
    preds = []
    tta_transform = A.Compose([
        A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(p=0.3),
        A.Normalize(), ToTensorV2()
    ])
    for _ in range(n_augments):
        aug_img = tta_transform(image=image)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            pred = torch.softmax(model(aug_img), dim=1)
        preds.append(pred)
    return torch.stack(preds).mean(0)
```

---

## 22. Transfer Learning & Fine-Tuning

### 22.1 Fine-Tuning Strategies

```python
import torch
import torch.nn as nn
import timm

# Strategy 1: Head only (feature extraction)
model = timm.create_model("efficientnet_b3", pretrained=True, num_classes=0)
for param in model.parameters():
    param.requires_grad = False
head = nn.Linear(model.num_features, num_classes)
# Only head trains

# Strategy 2: Gradual unfreezing
def unfreeze_last_n_blocks(model, n):
    blocks = list(model.children())[-n:]
    for block in blocks:
        for param in block.parameters():
            param.requires_grad = True

# Strategy 3: Discriminative learning rates
def get_layerwise_lr(model, base_lr, lr_multiplier=0.9):
    params = []
    layers = list(model.named_parameters())
    for i, (name, param) in enumerate(layers):
        lr = base_lr * (lr_multiplier ** (len(layers) - i))
        params.append({"params": param, "lr": lr})
    return params

optimizer = torch.optim.AdamW(get_layerwise_lr(model, 1e-3), weight_decay=0.01)

# Strategy 4: Warmup + Cosine decay
from torch.optim.lr_scheduler import OneCycleLR
scheduler = OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader),
                        epochs=num_epochs, pct_start=0.1)  # 10% warmup

# Full fine-tuning with weight decay
model = timm.create_model("convnext_small", pretrained=True, num_classes=num_classes)
# Layer decay (common for ViT fine-tuning)
def build_layer_decay_optimizer(model, base_lr=5e-4, layer_decay=0.75, weight_decay=0.05):
    num_layers = model.get_num_layers() if hasattr(model, 'get_num_layers') else 12
    param_groups = []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        lr_scale = layer_decay ** (num_layers - get_layer_idx(name, num_layers))
        param_groups.append({"params": param, "lr": base_lr*lr_scale,
                               "weight_decay": 0 if "bias" in name or "norm" in name else weight_decay})
    return torch.optim.AdamW(param_groups)
```

### 22.2 LoRA for Vision Models

```python
# LoRA (Low-Rank Adaptation) for efficient fine-tuning
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, original_layer, r=8, alpha=16):
        super().__init__()
        self.original = original_layer
        self.original.weight.requires_grad = False
        dim_in  = original_layer.in_features
        dim_out = original_layer.out_features
        self.lora_A = nn.Parameter(torch.randn(dim_in, r) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(r, dim_out))
        self.scale  = alpha / r

    def forward(self, x):
        base_out = self.original(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scale
        return base_out + lora_out

# Using PEFT library (easier)
from peft import LoraConfig, get_peft_model, TaskType

peft_config = LoraConfig(
    task_type=TaskType.IMAGE_CLASSIFICATION,
    r=8, lora_alpha=16, lora_dropout=0.1,
    target_modules=["query", "value"]  # attention projection layers
)
peft_model = get_peft_model(base_model, peft_config)
peft_model.print_trainable_parameters()
# Only ~1-2% parameters trainable!
```

---

## 23. Model Evaluation & Metrics

### 23.1 Classification Metrics

```python
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, average_precision_score,
                              roc_curve, precision_recall_curve)
import matplotlib.pyplot as plt
import torch

# Collect predictions
all_preds, all_labels, all_probs = [], [], []
model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

# Classification report
print(classification_report(all_labels, all_preds, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# AUC-ROC (multiclass)
auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')

# Top-5 accuracy
def top_k_accuracy(probs, labels, k=5):
    topk = np.argsort(probs, axis=1)[:, -k:]
    correct = sum(1 for i, t in enumerate(topk) if labels[i] in t)
    return correct / len(labels)

print(f"Top-5 Acc: {top_k_accuracy(all_probs, all_labels):.4f}")
```

### 23.2 Detection Metrics (mAP)

```python
# mAP calculation from scratch
def compute_ap(recall, precision):
    """Compute AP using 11-point interpolation."""
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        p = precision[recall >= t]
        ap += np.max(p) if len(p) > 0 else 0.0
    return ap / 11.0

def compute_map(pred_boxes, pred_scores, pred_labels,
                gt_boxes, gt_labels, iou_threshold=0.5, num_classes=80):
    aps = []
    for c in range(num_classes):
        pred_mask = pred_labels == c
        gt_mask   = gt_labels   == c
        if gt_mask.sum() == 0: continue
        p_boxes = pred_boxes[pred_mask]
        p_scores= pred_scores[pred_mask]
        g_boxes = gt_boxes[gt_mask]
        # Sort by score descending
        order = np.argsort(-p_scores)
        p_boxes, p_scores = p_boxes[order], p_scores[order]
        tp = np.zeros(len(p_boxes))
        fp = np.zeros(len(p_boxes))
        matched = np.zeros(len(g_boxes), dtype=bool)
        for i, pb in enumerate(p_boxes):
            ious = np.array([compute_iou(pb, gb) for gb in g_boxes])
            if len(ious) > 0 and ious.max() >= iou_threshold:
                j = ious.argmax()
                if not matched[j]:
                    tp[i] = 1; matched[j] = True
                else:
                    fp[i] = 1
            else:
                fp[i] = 1
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall    = tp_cum / len(g_boxes)
        precision = tp_cum / (tp_cum + fp_cum + 1e-6)
        aps.append(compute_ap(recall, precision))
    return np.mean(aps)

# COCO API (preferred for standard evaluation)
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

coco_gt  = COCO("annotations.json")
coco_dt  = coco_gt.loadRes("predictions.json")
coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
# Reports: mAP@[.5:.95], mAP@.5, mAP@.75, mAPs, mAPm, mAPl
```

### 23.3 Segmentation Metrics

```python
import numpy as np
import torch

def dice_score(pred, target, eps=1e-6):
    """pred, target: binary tensors (H,W) or (B,H,W)"""
    pred   = pred.float().flatten()
    target = target.float().flatten()
    intersection = (pred * target).sum()
    return (2*intersection + eps) / (pred.sum() + target.sum() + eps)

def iou_score(pred, target, eps=1e-6):
    pred   = pred.float().flatten()
    target = target.float().flatten()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + eps) / (union + eps)

def pixel_accuracy(pred, target):
    correct = (pred == target).sum().float()
    total   = target.numel()
    return (correct / total).item()

def mean_iou(pred, target, num_classes):
    ious = []
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        inter = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()
        if union == 0: continue
        ious.append(inter / union)
    return torch.stack(ious).mean().item()
```

---

## 24. Model Optimization & Deployment

### 24.1 ONNX Export & Inference

```python
import torch
import onnx, onnxruntime as ort
import numpy as np

# Export to ONNX
model.eval()
dummy_input = torch.randn(1, 3, 224, 224).to(device)
torch.onnx.export(
    model, dummy_input, "model.onnx",
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

# Verify
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)

# Inference
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
session = ort.InferenceSession("model.onnx", providers=providers)

input_name = session.get_inputs()[0].name
dummy_np = dummy_input.cpu().numpy()
output = session.run(None, {input_name: dummy_np})[0]

# Benchmark
import time
times = []
for _ in range(100):
    t0 = time.perf_counter()
    session.run(None, {input_name: dummy_np})
    times.append(time.perf_counter() - t0)
print(f"Avg latency: {np.mean(times)*1000:.2f}ms")
```

### 24.2 TensorRT

```python
# Method 1: via Ultralytics (easiest for YOLO)
model = YOLO("yolov8n.pt")
model.export(format="engine", half=True, imgsz=640)  # creates .engine

# Method 2: via torch2trt
from torch2trt import torch2trt
model_trt = torch2trt(model, [dummy_input], fp16_mode=True,
                        max_workspace_size=1<<30)
output_trt = model_trt(dummy_input)

# Method 3: TensorRT Python API
import tensorrt as trt
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path, fp16=True):
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30
        if fp16: config.set_flag(trt.BuilderFlag.FP16)
        with open(onnx_path, "rb") as f:
            parser.parse(f.read())
        return builder.build_engine(network, config)
```

### 24.3 Model Quantization & Pruning

```python
import torch
import torch.quantization as quant

# --- Post-Training Static Quantization ---
model.eval()
model.qconfig = quant.get_default_qconfig("fbgemm")  # or "qnnpack" for ARM
quant.prepare(model, inplace=True)

# Calibrate with representative data
with torch.no_grad():
    for images, _ in calibration_loader:
        model(images)

quant.convert(model, inplace=True)
# Model is now quantized (INT8)!

# --- Dynamic Quantization ---
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear, nn.LSTM}, dtype=torch.qint8
)

# --- Quantization Aware Training (best accuracy) ---
model.train()
model.qconfig = quant.get_default_qat_qconfig("fbgemm")
quant.prepare_qat(model, inplace=True)
# ... train normally ...
model.eval()
quant.convert(model, inplace=True)

# --- Pruning ---
import torch.nn.utils.prune as prune

# Unstructured pruning (30% of weights → 0)
prune.random_unstructured(model.conv1, name="weight", amount=0.3)
prune.l1_unstructured(model.conv1, name="weight", amount=0.3)

# Structured pruning (remove entire filters)
prune.ln_structured(model.conv1, name="weight", amount=0.3, n=2, dim=0)

# Global pruning
parameters = [(m, "weight") for m in model.modules()
              if isinstance(m, nn.Conv2d)]
prune.global_unstructured(parameters, pruning_method=prune.L1Unstructured, amount=0.2)

# Remove pruning masks (make permanent)
for module, _ in parameters:
    prune.remove(module, "weight")

# --- Knowledge Distillation ---
class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.7):
        super().__init__()
        self.T     = temperature
        self.alpha = alpha

    def forward(self, student_logits, teacher_logits, labels):
        soft_loss = nn.KLDivLoss(reduction='batchmean')(
            nn.functional.log_softmax(student_logits/self.T, dim=1),
            nn.functional.softmax(teacher_logits/self.T, dim=1)
        ) * (self.T ** 2)
        hard_loss = nn.CrossEntropyLoss()(student_logits, labels)
        return self.alpha * soft_loss + (1-self.alpha) * hard_loss
```

### 24.4 FastAPI Deployment

```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch, torchvision.transforms as T
from PIL import Image
import io, numpy as np

app = FastAPI(title="CV API", description="Computer Vision Model API")

# Load model once at startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = load_your_model().to(device).eval()
CLASS_NAMES = ["cat", "dog", "bird"]

transform = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

@app.post("/predict/classify")
async def classify(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg","image/png","image/jpg"]:
        raise HTTPException(400, "Invalid file type")
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1)[0]
        topk   = probs.topk(3)
    return {"predictions": [
        {"class": CLASS_NAMES[i], "confidence": float(p)}
        for i, p in zip(topk.indices, topk.values)
    ]}

@app.post("/predict/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    results = yolo_model(img)
    boxes = []
    for r in results:
        for box in r.boxes:
            boxes.append({
                "x1": float(box.xyxy[0][0]), "y1": float(box.xyxy[0][1]),
                "x2": float(box.xyxy[0][2]), "y2": float(box.xyxy[0][3]),
                "confidence": float(box.conf), "class": CLASS_NAMES[int(box.cls)]
            })
    return {"detections": boxes}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 25. Libraries & Frameworks Quick Reference

| Library | Purpose | Install |
|---|---|---|
| `opencv-python` | Image processing, classical CV | `pip install opencv-python` |
| `Pillow` | Image I/O, basic ops | `pip install Pillow` |
| `PyTorch` | Deep learning framework | `pip install torch torchvision` |
| `TensorFlow/Keras` | Alternative DL framework | `pip install tensorflow` |
| `torchvision` | CV models, transforms, datasets | bundled with PyTorch |
| `timm` | 500+ pretrained CV models | `pip install timm` |
| `ultralytics` | YOLO family models | `pip install ultralytics` |
| `transformers` | HuggingFace VLMs, ViT, etc. | `pip install transformers` |
| `diffusers` | Stable Diffusion, SDXL | `pip install diffusers` |
| `albumentations` | Fast image augmentation | `pip install albumentations` |
| `mediapipe` | Pose, hands, face detection | `pip install mediapipe` |
| `face_recognition` | Face recognition (dlib) | `pip install face-recognition` |
| `deepface` | Face analysis suite | `pip install deepface` |
| `segment_anything` | SAM by Meta | `pip install segment-anything` |
| `open3d` | 3D point clouds, mesh | `pip install open3d` |
| `pytesseract` | Tesseract OCR wrapper | `pip install pytesseract` |
| `easyocr` | Easy multilingual OCR | `pip install easyocr` |
| `paddleocr` | Best accuracy OCR | `pip install paddlepaddle paddleocr` |
| `pydicom` | DICOM medical images | `pip install pydicom` |
| `monai` | Medical image AI | `pip install monai` |
| `scikit-image` | Image processing algorithms | `pip install scikit-image` |
| `imageio` | Image/video I/O | `pip install imageio` |
| `pycocotools` | COCO dataset evaluation | `pip install pycocotools` |
| `wandb` | Experiment tracking | `pip install wandb` |
| `tensorboard` | Training visualization | `pip install tensorboard` |
| `onnx` | Model export/interop | `pip install onnx onnxruntime` |
| `boxmot` | Multi-object tracking | `pip install boxmot` |
| `clip` | OpenAI CLIP | `pip install git+https://github.com/openai/CLIP.git` |

---

## 26. Datasets Encyclopedia

| Dataset | Task | Size | Classes | Notes |
|---|---|---|---|---|
| **ImageNet-1K** | Classification | 1.28M | 1000 | Standard benchmark |
| **CIFAR-10/100** | Classification | 60K | 10/100 | 32×32 images |
| **COCO** | Det/Seg/Pose | 330K | 80 | Most popular detection |
| **Pascal VOC** | Detection | 22K | 20 | Classic benchmark |
| **Open Images** | Detection | 9M | 600 | Google dataset |
| **ADE20K** | Segmentation | 25K | 150 | Scene parsing |
| **CityScapes** | Seg (driving) | 25K | 30 | Street scenes |
| **KITTI** | Autonomous driving | 15K | — | LiDAR + stereo |
| **nuScenes** | Autonomous driving | 400K | 23 | Multi-sensor |
| **LVIS** | Instance Seg | 164K | 1203 | Long-tail |
| **WiderFace** | Face detection | 32K | — | 393K faces |
| **LFW** | Face recognition | 13K | 5749 | In-the-wild faces |
| **Kinetics-700** | Video class | 650K clips | 700 | Action recognition |
| **UCF101** | Video class | 13K clips | 101 | Human actions |
| **HMDB51** | Video class | 7K clips | 51 | Human motion |
| **ScanNet** | 3D scene | 2.5M frames | — | RGB-D indoor |
| **ShapeNet** | 3D shapes | 51K 3D | 55 | 3D object dataset |
| **DOTA** | Aerial detection | 11K | 15 | Satellite images |
| **Chest X-Ray14** | Medical | 112K | 14 | Pathology |
| **DRIVE** | Retinal | 40 | — | Vessel segmentation |
| **STL-10** | Unlabeled+class | 13K | 10 | Self-supervised |
| **CelebA** | Face attr | 200K | 40 attrs | Celebrity faces |
| **TextOCR** | OCR | 900K words | — | Scene text |
| **MVTec AD** | Anomaly det | 5K | 15 | Industrial defects |

```python
# Load common datasets
from torchvision import datasets

cifar10 = datasets.CIFAR10("data", train=True, download=True, transform=transform)
imagenet = datasets.ImageNet("data/imagenet", split="train", transform=transform)
coco    = datasets.CocoDetection("data/coco/images", "data/coco/annotations.json")
voc     = datasets.VOCDetection("data", year="2012", image_set="train")

# HuggingFace datasets
from datasets import load_dataset
ds = load_dataset("cifar100")
ds = load_dataset("imagenet-1k", split="train", streaming=True)
ds = load_dataset("open-images-v6")
```

---

## 27. Loss Functions Reference

| Loss | Use Case | Formula | Code |
|---|---|---|---|
| **Cross Entropy** | Classification | `-Σ y log(p)` | `nn.CrossEntropyLoss()` |
| **Binary CE** | Binary classification | `-y log(p) - (1-y) log(1-p)` | `nn.BCEWithLogitsLoss()` |
| **Focal Loss** | Class imbalance | `-(1-p)^γ log(p)` | see below |
| **MSE** | Regression, depth | `Σ(y-ŷ)²/n` | `nn.MSELoss()` |
| **MAE/L1** | Robust regression | `Σ|y-ŷ|/n` | `nn.L1Loss()` |
| **Huber** | Robust regression | smooth L1+L2 | `nn.HuberLoss()` |
| **Dice** | Segmentation | `2|X∩Y|/(|X|+|Y|)` | see below |
| **IoU Loss** | Segmentation | `1 - IoU` | see below |
| **GIoU/CIoU** | Detection boxes | extended IoU | torchvision |
| **Triplet** | Metric learning | `max(d_ap - d_an + m, 0)` | `nn.TripletMarginLoss()` |
| **Contrastive** | Siamese nets | pair-based | see below |
| **InfoNCE** | Self-supervised | contrastive | SimCLR, CLIP |
| **Perceptual** | Style transfer | VGG feature diff | see below |
| **SSIM** | Image quality | structural similarity | pip install piq |

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Focal Loss ---
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt  = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean() if self.reduction == "mean" else focal

# --- Dice Loss ---
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred   = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        inter  = (pred * target).sum()
        dice   = (2*inter + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice

# Combined Dice + BCE (great for medical segmentation)
class DiceBCELoss(nn.Module):
    def forward(self, pred, target):
        dice = DiceLoss()(pred, target)
        bce  = nn.BCEWithLogitsLoss()(pred, target)
        return 0.5*dice + 0.5*bce

# --- Perceptual Loss ---
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = torchvision.models.vgg16(weights="DEFAULT").features
        self.slice1 = nn.Sequential(*list(vgg.children())[:4]).eval()
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9]).eval()
        for p in self.parameters(): p.requires_grad = False

    def forward(self, pred, target):
        f1_p = self.slice1(pred);   f1_t = self.slice1(target)
        f2_p = self.slice2(f1_p);  f2_t = self.slice2(f1_t)
        return F.mse_loss(f1_p, f1_t) + F.mse_loss(f2_p, f2_t)

# --- InfoNCE / NT-Xent (SimCLR) ---
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.T = temperature

    def forward(self, z1, z2):
        B = z1.size(0)
        z  = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
        sim = torch.mm(z, z.T) / self.T
        labels = torch.cat([torch.arange(B)+B, torch.arange(B)]).to(z.device)
        mask   = torch.eye(2*B, dtype=bool).to(z.device)
        sim[mask] = -float('inf')
        return F.cross_entropy(sim, labels)
```

---

## 28. Architecture Cheat Sheet

```
CLASSIFICATION
──────────────
LeNet-5      → AlexNet → VGG → GoogLeNet/Inception
→ ResNet → DenseNet → EfficientNet → ConvNeXt
→ ViT → DeiT → Swin → MaxViT

OBJECT DETECTION
────────────────
Two-stage:
  R-CNN → Fast R-CNN → Faster R-CNN → Cascade RCNN → Sparse RCNN

One-stage:
  YOLO → YOLOv2 → v3 → v4 → v5 → YOLOv7 → YOLOv8 → YOLOv9 → v11
  SSD → RetinaNet (Focal Loss) → EfficientDet
  
Anchor-free:
  FCOS → CenterNet → DETR → DAB-DETR → DINO → RT-DETR

SEGMENTATION
────────────
Semantic:   FCN → SegNet → U-Net → PSPNet → DeepLab series
            → OCRNet → SegFormer → Mask2Former
Instance:   Mask RCNN → YOLACT → SOLOv2 → CondInst
Panoptic:   Panoptic FPN → MaskFormer → Mask2Former → OneFormer
Universal:  SAM → SAM2

GENERATION
──────────
VAE → GAN (DCGAN) → cGAN → StyleGAN2/3 → BigGAN
→ DDPM (diffusion) → DDIM → LDM → Stable Diffusion → SDXL
→ DiT → Flux → Stable Diffusion 3

TRACKING
────────
Siamese networks → SiamRPN → SiamMask
SORT → DeepSORT → StrongSORT → BoTrack → ByteTrack → OC-SORT

3D VISION
─────────
Point clouds: PointNet → PointNet++ → VoxNet → SECOND
Reconstruction: NeRF → Instant-NGP → Gaussian Splatting
Depth: MiDaS → DPT → ZoeDepth → Depth Anything V2

VL MODELS
─────────
CLIP → ALIGN → Florence → BLIP → BLIP-2 → InstructBLIP
→ LLaVA → LLaVA-1.5 → LLaVA-Next → Qwen-VL → InternVL → GPT-4V
```

---

## 29. CV Interview Q&A

### Q1: Explain the difference between Semantic and Instance Segmentation.
**A:** Semantic segmentation assigns a class label to every pixel (all cars = same color). Instance segmentation distinguishes individual objects of the same class (each car has a unique ID). Panoptic segmentation combines both.

### Q2: What is the role of Batch Normalization?
**A:** BN normalizes the activations of each layer to have zero mean and unit variance per mini-batch, then applies learnable scale (γ) and shift (β). Benefits: accelerates training (allows higher LR), acts as regularizer, reduces internal covariate shift. Limitation: problematic for small batch sizes (use Group Norm instead).

### Q3: Why is ReLU preferred over Sigmoid/Tanh in deep networks?
**A:** Sigmoid/Tanh suffer from vanishing gradients in deep networks (gradients → 0 for saturated inputs). ReLU has gradient = 1 for positive inputs, enabling faster training. Issues: dying ReLU (solved by Leaky ReLU, PReLU, ELU, GELU).

### Q4: Explain the concept of Receptive Field.
**A:** The receptive field is the region of the input image that affects a particular neuron's output. For a single conv layer with kernel k: RF=k. For stacked layers: RF_n = RF_{n-1} + (k-1) × stride_product. Dilated convolutions increase RF without losing spatial resolution.

### Q5: What is anchor-free detection and how does FCOS work?
**A:** Anchor-free detection avoids pre-defined anchor boxes. FCOS treats detection as a per-pixel regression task — for each pixel inside an object, it predicts the distances to the 4 sides (l,t,r,b) of the bounding box plus a centerness score. This eliminates anchor hyperparameter tuning.

### Q6: How does Non-Maximum Suppression (NMS) work?
**A:** (1) Sort detections by confidence score descending. (2) Keep the highest-confidence box. (3) Remove all other boxes with IoU > threshold with the kept box. (4) Repeat. Soft-NMS decays scores instead of removing, better for overlapping objects.

### Q7: What is Feature Pyramid Network (FPN)?
**A:** FPN adds a top-down pathway to a bottom-up CNN backbone. High-level semantic features from deep layers are upsampled and merged with high-resolution low-level features via lateral connections, creating a multi-scale feature map pyramid for detecting objects at multiple scales.

### Q8: Explain the attention mechanism in Vision Transformers.
**A:** ViT splits an image into fixed patches (e.g., 16×16), linearly embeds each, adds positional embeddings, and feeds to a standard Transformer. Multi-head self-attention computes Q, K, V from each patch; attention weights = softmax(QK^T/√d_k) × V. This allows global context from the first layer, unlike CNNs.

### Q9: What is the vanishing gradient problem and how do ResNets solve it?
**A:** In very deep networks, gradients become exponentially small during backprop through many layers. ResNets add skip connections: output = F(x) + x. The gradient can flow directly through the identity connection, ensuring gradients don't vanish even in 100+ layer networks.

### Q10: How do you handle class imbalance in detection?
**A:** (1) Focal Loss — down-weights easy negatives, focuses on hard examples. (2) Class-weighted loss — higher weight for rare classes. (3) OHEM — online hard example mining. (4) Oversampling rare classes. (5) Data augmentation for rare classes. (6) Balanced sampling strategies.

### Q11: What is the difference between L1, L2, and Huber loss?
```
L2 (MSE): (y-ŷ)²     — sensitive to outliers, smooth gradients
L1 (MAE): |y-ŷ|      — robust to outliers, non-smooth at 0
Huber:    L2 if |e|<δ, L1 if |e|≥δ — best of both
```

### Q12: Explain GAN training instability and solutions.
**A:** GAN training is adversarial and can suffer from mode collapse (generator produces limited variety) and non-convergence. Solutions: (1) Wasserstein GAN (WGAN) with gradient penalty — more stable. (2) Spectral normalization. (3) Progressive growing (ProGAN). (4) Training tricks: label smoothing, instance noise, balanced G/D updates.

---

## 30. End-to-End Project Templates

### Project 1: Real-time Object Detection System

```python
import cv2
from ultralytics import YOLO
import torch
import time

class RealTimeDetector:
    def __init__(self, model_path="yolov8n.pt", conf=0.5, source=0):
        self.model  = YOLO(model_path)
        self.conf   = conf
        self.source = source
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def run(self):
        cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        fps_history = []
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret: break

            results = self.model(frame, conf=self.conf,
                                  verbose=False, device=self.device)
            annotated = results[0].plot()

            fps = 1 / (time.time() - t0)
            fps_history.append(fps)
            avg_fps = sum(fps_history[-30:]) / len(fps_history[-30:])
            cv2.putText(annotated, f"FPS: {avg_fps:.1f}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.imshow("Detection", annotated)
            if cv2.waitKey(1) == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()

detector = RealTimeDetector(model_path="yolov8m.pt", conf=0.4)
detector.run()
```

### Project 2: Face Attendance System

```python
import face_recognition, cv2, numpy as np, pickle, datetime, csv

class AttendanceSystem:
    def __init__(self, db_path="faces.pkl"):
        self.db_path = db_path
        try:
            with open(db_path, "rb") as f:
                self.db = pickle.load(f)
        except: self.db = {}
        self.marked_today = set()

    def register(self, name, image_path):
        img = face_recognition.load_image_file(image_path)
        encs = face_recognition.face_encodings(img)
        if encs: self.db[name] = encs[0]
        with open(self.db_path, "wb") as f: pickle.dump(self.db, f)
        print(f"Registered: {name}")

    def mark_attendance(self, name):
        if name not in self.marked_today:
            self.marked_today.add(name)
            with open("attendance.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow([name, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            print(f"Attendance marked: {name}")

    def run(self):
        known_encs  = list(self.db.values())
        known_names = list(self.db.keys())
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret: break
            small = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
            rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locs  = face_recognition.face_locations(rgb)
            encs  = face_recognition.face_encodings(rgb, locs)
            for enc, (top,right,bottom,left) in zip(encs, locs):
                dists = face_recognition.face_distance(known_encs, enc)
                name  = known_names[dists.argmin()] if len(dists)>0 and dists.min()<0.5 else "Unknown"
                if name != "Unknown": self.mark_attendance(name)
                top*=4;right*=4;bottom*=4;left*=4
                color = (0,255,0) if name != "Unknown" else (0,0,255)
                cv2.rectangle(frame,(left,top),(right,bottom),color,2)
                cv2.putText(frame,name,(left,top-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
            cv2.imshow("Attendance System", frame)
            if cv2.waitKey(1) == ord('q'): break
        cap.release()
        cv2.destroyAllWindows()
```

### Project 3: Document Scanner

```python
import cv2, numpy as np
from PIL import Image
import pytesseract

def order_points(pts):
    rect = np.zeros((4,2), dtype=np.float32)
    s    = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
    diff     = np.diff(pts, axis=1)
    rect[1]  = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
    return rect

def scan_document(image_path, extract_text=True):
    img  = cv2.imread(image_path)
    orig = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edged   = cv2.Canny(blurred, 75, 200)

    # Find document contour
    contours,_ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours    = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    doc_cnt = None
    for cnt in contours:
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
        if len(approx) == 4:
            doc_cnt = approx; break

    if doc_cnt is None:
        print("No document found"); return img

    # Perspective transform
    pts = order_points(doc_cnt.reshape(4,2))
    (tl,tr,br,bl) = pts
    w = max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl))
    h = max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl))
    dst = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    M   = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(orig, M, (int(w),int(h)))

    # Enhance
    gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray_w)

    cv2.imwrite("scanned.jpg", enhanced)

    if extract_text:
        text = pytesseract.image_to_string(enhanced, config="--psm 6")
        with open("extracted.txt","w") as f: f.write(text)
        return enhanced, text
    return enhanced

scanned, text = scan_document("document.jpg")
```

### Project 4: Custom Object Detection Training (End-to-End)

```bash
# Dataset structure for YOLO:
# dataset/
#   images/
#     train/   img1.jpg, img2.jpg, ...
#     val/     img1.jpg, ...
#   labels/
#     train/   img1.txt (one line per box: class cx cy w h)
#     val/     img1.txt
#   data.yaml

# data.yaml content:
# path: /path/to/dataset
# train: images/train
# val: images/val
# nc: 3
# names: ['cat', 'dog', 'person']
```

```python
from ultralytics import YOLO
import yaml

# Create data.yaml
data_config = {
    "path": "/path/to/dataset",
    "train": "images/train",
    "val":   "images/val",
    "nc":    3,
    "names": ["cat", "dog", "person"]
}
with open("data.yaml", "w") as f:
    yaml.dump(data_config, f)

# Train
model = YOLO("yolov8s.pt")  # start from pretrained
results = model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    device="0",
    workers=8,
    project="runs/custom",
    name="experiment1",
    cache=True,
    amp=True,         # mixed precision
    patience=50,      # early stopping
    save=True,
    plots=True,
    augment=True,
    degrees=10,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,
)

# Evaluate
model_best = YOLO("runs/custom/experiment1/weights/best.pt")
metrics = model_best.val(data="data.yaml")
print(f"mAP50: {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")

# Export
model_best.export(format="onnx", half=True)
model_best.export(format="engine", half=True)  # TensorRT
```

---

## 🔧 Quick Tips & Best Practices

```
GENERAL
• Always normalize images before feeding to neural networks
• Use GPU for training, can use CPU for inference of small models
• Visualize your data before training — catch label errors early
• Start with a simple baseline, then iterate
• Use wandb/tensorboard to track experiments

TRAINING
• Learning rate is the most important hyperparameter
• Cosine annealing with warm restarts works well
• Gradient clipping prevents exploding gradients
• Mixed precision (fp16) gives ~2x speedup with same accuracy
• Batch norm: use batch ≥ 32; smaller batches → use GroupNorm

DATA
• More data > better models (usually)
• Check class distribution before training
• Augmentation is free data — always use it
• Pretrain on large dataset, fine-tune on target domain
• 80/10/10 or 70/15/15 train/val/test split

DETECTION
• Start with YOLOv8n for speed, YOLOv8x for accuracy
• IoU threshold 0.5 for NMS is standard
• mAP@0.5:0.95 is the standard COCO metric
• Anchor size should match your object size distribution

DEPLOYMENT
• Profile first: find actual bottleneck
• ONNX → TensorRT is the standard production pipeline
• TensorRT INT8 gives 3-4x speedup vs FP32
• Batch inference when possible
• Use async/streaming for video applications

DEBUGGING
• Check loss curves: exploding/vanishing = lr issue
• Check gradient norms during training
• Visualize augmented batches
• Check class distribution in predictions vs. ground truth
• Use smaller subset first to verify pipeline
```

---

## 📐 Math Quick Reference

```
Convolution output size:
  H_out = ⌊(H_in + 2P - D(K-1) - 1) / S + 1⌋
  W_out = ⌊(W_in + 2P - D(K-1) - 1) / S + 1⌋
  P=padding, D=dilation, K=kernel, S=stride

Same padding (output = input):
  P = (K-1) / 2  (for S=1, D=1)

Parameter count:
  Conv:   K × K × C_in × C_out + C_out (bias)
  Linear: in × out + out (bias)
  BN:     2 × C (gamma + beta)

Receptive field:
  RF_1 = K
  RF_n = RF_{n-1} + (K-1) × S_product

FLOP count (conv):
  2 × K² × C_in × C_out × H_out × W_out

Attention complexity:
  Full:   O(N²d)
  Window: O(W²Nd/W) = O(WNd)  [Swin]

IoU:      |A ∩ B| / |A ∪ B|
GIoU:     IoU - |C \ (A∪B)| / |C|  (C = smallest enclosing box)
CIoU:     IoU - ρ²(b,bgt)/c² - αv  (adds center distance + aspect ratio)

Softmax:  σ(z_i) = exp(z_i) / Σ exp(z_j)
Cross-entropy: L = -Σ y_i log(p_i)
Focal:    FL(p_t) = -(1-p_t)^γ log(p_t)
Dice:     1 - 2|X∩Y| / (|X|+|Y|)
```

---

*📌 Last updated: 2024–2025 | Covers classical CV through latest DL models*  
*💡 Pull request / contribute: Add missing topics or correct errors*  
*⭐ Star this handbook if it helps you!*
