import cv2
import numpy as np
import matplotlib.pyplot as plt

def uncalibrated_rectify_sift(im1="wood_middle.jpg", im2="wood_middle.jpg"):
    # Ensure you have a version of OpenCV that supports SIFT
    # For newer versions, you might need to use:
    # cv2.xfeatures2d.SIFT_create()

    # Read images
    I1 = cv2.imread(im1)
    I2 = cv2.imread(im2)

    # Convert to grayscale
    I1gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    I2gray = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

    # SIFT feature detection
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(I1gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(I2gray, None)

    # Feature matching using FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Extract matched keypoints
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

    # Find fundamental matrix
    F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, ransacReprojThreshold=3)

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
    plt.figure(figsize=(15, 10))

    # Original images with matches
    plt.subplot(221)
    plt.title('Matched Features')
    matched_img = cv2.drawMatches(I1, keypoints1, I2, keypoints2, 
                                good_matches[:50], None, 
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))

    # Original images
    plt.subplot(222)
    plt.title('Original Images')
    original_comparison = np.hstack((I1, I2))
    plt.imshow(cv2.cvtColor(original_comparison, cv2.COLOR_BGR2RGB))

    # Rectified images
    plt.subplot(223)
    plt.title('Rectified Left Image')
    plt.imshow(cv2.cvtColor(I1_rect, cv2.COLOR_BGR2RGB))

    plt.subplot(224)
    plt.title('Rectified Right Image')
    plt.imshow(cv2.cvtColor(I2_rect, cv2.COLOR_BGR2RGB))

    plt.tight_layout()
    plt.show()

    # Disparity map
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.title('Disparity Map')
    plt.imshow(disparity, cmap='jet')
    plt.colorbar()

    plt.subplot(122)
    plt.title('Normalized Disparity')
    plt.imshow(cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U), cmap='jet')
    plt.colorbar()

    plt.tight_layout()
    plt.show()