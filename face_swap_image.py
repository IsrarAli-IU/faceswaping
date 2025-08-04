# face_swap_image.py
"""
Accurate face swapping between two images using OpenCV and face_alignment (warp & seamless clone).

Steps:
 1. Open this file in VS Code.
 2. Install dependencies:
      pip install opencv-python numpy torch face-alignment
 3. Place two images in the same folder: 'face1.jpg' and 'face2.jpg'.
 4. Run:
      python face_swap_image.py
 5. The result is saved as 'result.jpg'.

"""
import cv2
import numpy as np
import face_alignment
import torch

# Initialize face_alignment (uses GPU if available)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D,
    flip_input=False,
    device=DEVICE
)

# Detect 68 facial landmarks
def get_landmarks(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    preds = fa.get_landmarks(rgb)
    if preds is None or len(preds) == 0:
        raise RuntimeError("No face detected")
    return np.array(preds[0], dtype=np.int32)

# Apply an affine transform to a triangular region
def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    return cv2.warpAffine(src, warp_mat, (size[0], size[1]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

# Compute Delaunay triangles for a set of points
def delaunay_triangulation(rect, points):
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert(tuple(p))
    triangles = subdiv.getTriangleList()
    delaunay = []
    for t in triangles:
        pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        idx = []
        for p in pts:
            for i, pt in enumerate(points):
                if abs(p[0] - pt[0]) < 1.0 and abs(p[1] - pt[1]) < 1.0:
                    idx.append(i)
        if len(idx) == 3:
            delaunay.append(tuple(idx))
    return delaunay

# Perform the face swap
def swap_face(src, dst):
    src_points = get_landmarks(src)
    dst_points = get_landmarks(dst)

    # Get convex hull indices for destination face
    hull_indices = cv2.convexHull(dst_points, returnPoints=False).flatten()
    src_hull = src_points[hull_indices]
    dst_hull = dst_points[hull_indices]

    # Calculate Delaunay triangles on the destination hull
    h, w = dst.shape[:2]
    rect = (0, 0, w, h)
    dt = delaunay_triangulation(rect, dst_hull.tolist())

    # Warp each triangle from source to destination
    warped_src = np.zeros_like(dst)
    for tri in dt:
        src_tri = [tuple(src_hull[i]) for i in tri]
        dst_tri = [tuple(dst_hull[i]) for i in tri]

        # Bounding rectangles
        r1 = cv2.boundingRect(np.float32([src_tri]))
        r2 = cv2.boundingRect(np.float32([dst_tri]))

        # Offset points
        src_rect_pts = [(p[0] - r1[0], p[1] - r1[1]) for p in src_tri]
        dst_rect_pts = [(p[0] - r2[0], p[1] - r2[1]) for p in dst_tri]

        # Crop & warp
        src_crop = src[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
        warped_patch = apply_affine_transform(src_crop, src_rect_pts, dst_rect_pts, (r2[2], r2[3]))

        # Mask & merge
        mask = np.zeros((r2[3], r2[2], 3), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(dst_rect_pts), (1, 1, 1), 16, 0)
        warped_src[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = \
            warped_src[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * (1 - mask) + warped_patch * mask

    # Create mask for seamless cloning (destination hull)
    hull_mask = np.zeros(dst.shape, dtype=dst.dtype)
    cv2.fillConvexPoly(hull_mask, dst_hull, (255, 255, 255))
    x, y, w, h = cv2.boundingRect(dst_hull)
    center = (x + w//2, y + h//2)

    # Seamless clone the warped source onto destination
    output = cv2.seamlessClone(np.uint8(warped_src), dst, hull_mask, center, cv2.NORMAL_CLONE)
    return output

if __name__ == '__main__':
    src = cv2.imread('face1.jpg')
    dst = cv2.imread('face2.jpg')
    if src is None or dst is None:
        print("Error: 'face1.jpg' or 'face2.jpg' not found.")
        exit(1)

    result = swap_face(src, dst)
    cv2.imwrite('result.jpg', result)
    print("Face swapped image saved as 'result.jpg'.")
