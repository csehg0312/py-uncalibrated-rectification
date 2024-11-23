import cv2
import numpy as np
import matplotlib.pyplot as plt
def uncalibrated_rectify_orb(im1="wood_middle.jpg", im2="wood_middle.jpg"):
    # Read images
    I1 = cv2.imread(im1)
    I2 = cv2.imread(im2)

    # Convert to grayscale
    I1gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    I2gray = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

    # Feature detection (using ORB instead of SURF)
    orb = cv2.ORB_create(nfeatures=500)
    keypoints1, descriptors1 = orb.detectAndCompute(I1gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(I2gray, None)

    # Feature matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched keypoints
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    # Find fundamental matrix
    F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC)

    # Select inlier points
    inliers_src = src_pts[mask.ravel() == 1]
    inliers_dst = dst_pts[mask.ravel() == 1]

    # Stereo rectification
    _, H1, H2 = cv2.stereoRectifyUncalibrated(
        inliers_src, inliers_dst, F, imgSize=I1.shape[:2][::-1]
    )

    # Warp images
    I1_rect = cv2.warpPerspective(I1, H1, (I1.shape[1], I1.shape[0]))
    I2_rect = cv2.warpPerspective(I2, H2, (I2.shape[1], I2.shape[0]))

    # Convert rectified images to grayscale
    I1_rect_gray = cv2.cvtColor(I1_rect, cv2.COLOR_BGR2GRAY)
    I2_rect_gray = cv2.cvtColor(I2_rect, cv2.COLOR_BGR2GRAY)

    # Compute disparity
    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=9)
    disparity = stereo.compute(I1_rect_gray, I2_rect_gray)

    # Visualization
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.title('Original Left Image')
    plt.imshow(cv2.cvtColor(I1, cv2.COLOR_BGR2RGB))

    plt.subplot(132)
    plt.title('Original Right Image')
    plt.imshow(cv2.cvtColor(I2, cv2.COLOR_BGR2RGB))

    plt.subplot(133)
    plt.title('Disparity Map')
    plt.imshow(disparity, cmap='jet')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

    # Optional: Show rectified images
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.title('Rectified Left Image')
    plt.imshow(cv2.cvtColor(I1_rect, cv2.COLOR_BGR2RGB))

    plt.subplot(122)
    plt.title('Rectified Right Image')
    plt.imshow(cv2.cvtColor(I2_rect, cv2.COLOR_BGR2RGB))

    plt.tight_layout()
    plt.show()