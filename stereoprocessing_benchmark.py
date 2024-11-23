import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import psutil
import os
import threading
import tracemalloc
import sys

class StereoProcessingBenchmark:
    def __init__(self, image_left_path, image_right_path):
        # Initialize performance tracking
        self.start_time = None
        self.end_time = None
        self.cpu_usage = []
        self.memory_usage = []
        
        # Load images
        self.I1 = cv2.imread(image_left_path)
        self.I2 = cv2.imread(image_right_path)
        
        if self.I1 is None or self.I2 is None:
            raise ValueError(f"Could not read images from {image_left_path} or {image_right_path}")
        
        # Grayscale conversion
        self.I1gray = cv2.cvtColor(self.I1, cv2.COLOR_BGR2GRAY)
        self.I2gray = cv2.cvtColor(self.I2, cv2.COLOR_BGR2GRAY)
        
    def start_monitoring(self):
        """Start performance monitoring"""
        # Start tracemalloc at the beginning of the method
        tracemalloc.start()
        self.start_time = time.time()
        
        # Start resource monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.end_time = time.time()
        tracemalloc.stop()
        
    def _monitor_resources(self):
        """Continuously monitor CPU and memory usage"""
        process = psutil.Process(os.getpid())
        
        while self.end_time is None:
            try:
                # CPU Usage
                self.cpu_usage.append(process.cpu_percent())
                
                # Memory Usage
                memory_info = process.memory_info()
                self.memory_usage.append(memory_info.rss / 1024 / 1024)  # MB
                
                time.sleep(0.1)
            except Exception as e:
                print(f"Monitoring error: {e}")
                break
    
    def run_sift_stereo_processing(self):
        """Perform SIFT-based stereo processing with benchmarking"""
        # Start monitoring
        self.start_monitoring()
        
        try:
            # SIFT feature detection
            sift = cv2.SIFT_create()
            keypoints1, descriptors1 = sift.detectAndCompute(self.I1gray, None)
            keypoints2, descriptors2 = sift.detectAndCompute(self.I2gray, None)

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
                inliers_src, inliers_dst, F, imgSize=self.I1.shape[:2][::-1]
            )

            # Warp images
            I1_rect = cv2.warpPerspective(self.I1, H1, (self.I1.shape[1], self.I1.shape[0]))
            I2_rect = cv2.warpPerspective(self.I2, H2, (self.I2.shape[1], self.I2.shape[0]))

            # Convert rectified images to grayscale
            I1_rect_gray = cv2.cvtColor(I1_rect, cv2.COLOR_BGR2GRAY)
            I2_rect_gray = cv2.cvtColor(I2_rect, cv2.COLOR_BGR2GRAY)

            # Compute disparity
            stereo = cv2.StereoBM_create(numDisparities=64, blockSize=9)
            disparity = stereo.compute(I1_rect_gray, I2_rect_gray)
            
            return I1_rect, I2_rect, disparity, keypoints1, keypoints2, good_matches
        
        finally:
            # Stop monitoring
            self.stop_monitoring()
    
    def generate_performance_report(self):
        """Generate a comprehensive performance report"""
        # Calculate performance metrics
        processing_time = self.end_time - self.start_time
        
        # Prepare report
        report = {
            "Processing Time (seconds)": processing_time,
            "Average CPU Usage (%)": np.mean(self.cpu_usage) if self.cpu_usage else 0,
            "Peak CPU Usage (%)": np.max(self.cpu_usage) if self.cpu_usage else 0,
            "Average Memory Usage (MB)": np.mean(self.memory_usage) if self.memory_usage else 0,
            "Peak Memory Usage (MB)": np.max(self.memory_usage) if self.memory_usage else 0,
        }
        
        return report
    
    def visualize_performance(self):
        """Visualize performance metrics"""
        plt.figure(figsize=(10, 5))
        
        # CPU Usage
        plt.subplot(121)
        plt.title("CPU Usage")
        plt.plot(self.cpu_usage)
        plt.xlabel("Measurement Point")
        plt.ylabel("CPU Usage (%)")
        
        # Memory Usage
        plt.subplot(122)
        plt.title("Memory Usage")
        plt.plot(self.memory_usage)
        plt.xlabel("Measurement Point")
        plt.ylabel("Memory Usage (MB)")

        plt.tight_layout()
        plt.show()

# Main execution
def main():
    # Initialize benchmark
    benchmark = StereoProcessingBenchmark("tea-right.jpg", "tea-left.jpg")

    # Run stereo processing
    I1_rect, I2_rect, disparity, keypoints1, keypoints2, good_matches = benchmark.run_sift_stereo_processing()

    # Generate performance report
    performance_report = benchmark.generate_performance_report()

    # Print performance report
    print("Performance Report:")
    for key, value in performance_report.items():
        print(f"{key}: {value}")

    # Visualize performance
    benchmark.visualize_performance()

    # Visualization of results
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.title('Rectified Left Image')
    plt.imshow(cv2.cvtColor(I1_rect, cv2.COLOR_BGR2RGB))

    plt.subplot(132)
    plt.title('Rectified Right Image')
    plt.imshow(cv2.cvtColor(I2_rect, cv2.COLOR_BGR2RGB))

    plt.subplot(133)
    plt.title('Disparity Map')
    plt.imshow(disparity, cmap='jet')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()