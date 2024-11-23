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
    
    def sad_matching(self, descriptor1, descriptor2, threshold=0.7):
        """
        Perform Sum of Absolute Differences (SAD) matching with advanced optimization
        
        Args:
        - descriptor1: Descriptors from the first image
        - descriptor2: Descriptors from the second image
        - threshold: Matching threshold
        
        Returns:
        - Good matches based on SAD
        """
        # Ensure descriptors are not empty
        if descriptor1.shape[0] == 0 or descriptor2.shape[0] == 0:
            print("No descriptors to match.")
            return []

        # Use NumPy broadcasting for efficient distance computation
        # Reshape descriptors to enable broadcasting
        desc1_expanded = descriptor1[:, np.newaxis, :]
        desc2_expanded = descriptor2[np.newaxis, :, :]
        
        # Compute SAD distances using broadcasting
        distances = np.sum(np.abs(desc1_expanded - desc2_expanded), axis=2)
        
        # Find the two nearest neighbors for each descriptor
        matches = []
        for i in range(distances.shape[0]):
            # Sort distances for the current descriptor
            sorted_indices = np.argsort(distances[i])
            
            # Ensure we have at least two matches
            if len(sorted_indices) < 2:
                continue
            
            # Get the two best matches
            best_match_idx = sorted_indices[0]
            second_best_idx = sorted_indices[1]
            
            # Ratio test
            if (distances[i, best_match_idx] < 
                threshold * distances[i, second_best_idx]):
                matches.append(cv2.DMatch(i, best_match_idx, 
                                        distances[i, best_match_idx]))
        
        return matches

    def run_sift_stereo_processing(self):
        """Perform SIFT-based stereo processing with benchmarking"""
        # Start monitoring
        self.start_monitoring()
        
        try:
            # Use more efficient SIFT detection
            sift = cv2.SIFT_create(
                nfeatures=500,  # Limit number of features to reduce computation
                contrastThreshold=0.04,  # Adjust contrast threshold
                edgeThreshold=10  # Adjust edge threshold
            )
            
            # Detect and compute features
            keypoints1, descriptors1 = sift.detectAndCompute(self.I1gray, None)
            keypoints2, descriptors2 = sift.detectAndCompute(self.I2gray, None)
                
            if descriptors1 is None or descriptors2 is None:
                raise ValueError("Could not compute descriptors. Check input images.")
            
            if len(descriptors1) == 0 or len(descriptors2) == 0:
                raise ValueError("No keypoints detected in one or both images.")

            # Use custom SAD matching
            matches = self.sad_matching(descriptors1, descriptors2)
            
            # Check if enough matches are found
            if len(matches) < 4:
                raise ValueError(f"Insufficient matches found: {len(matches)}")

            # Extract matched keypoints
            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

            # Find fundamental matrix with error handling
            F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, ransacReprojThreshold=3)
            
            if F is None or mask is None:
                raise ValueError("Could not compute fundamental matrix")

            # Select inlier points
            inliers_src = src_pts[mask.ravel() == 1]
            inliers_dst = dst_pts[mask.ravel() == 1]

            # Check if enough inliers exist
            if len(inliers_src) < 4:
                raise ValueError(f"Insufficient inliers: {len(inliers_src)}")

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

            # Compute disparity using StereoBM
            stereo = cv2.StereoBM_create(numDisparities=64, blockSize=9)
            disparity = stereo.compute(I1_rect_gray, I2_rect_gray)
            
            return I1_rect, I2_rect, disparity, keypoints1, keypoints2, matches
        
        except Exception as e:
            print(f"Error in stereo processing: {e}")
            # Return None or raise a custom exception
            raise
        
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

    def visualize_matches(self, keypoints1, keypoints2, matches):
        """
        Visualize matched keypoints
        
        Args:
        - keypoints1: Keypoints from the first image
        - keypoints2: Keypoints from the second image
        - matches: List of matched keypoints
        """
        # Create a blank image to draw matches
        matched_image = np.zeros((max(self.I1.shape[0], self.I2.shape[0]), self.I1.shape[1] + self.I2.shape[1], 3), dtype=np.uint8)
        
        # Place the first image on the left
        matched_image[:self.I1.shape[0], :self.I1.shape[1]] = self.I1
        
        # Place the second image on the right
        matched_image[:self.I2.shape[0], self.I1.shape[1]:] = self.I2
        
        # Draw matches
        for match in matches:
            pt1 = (int(keypoints1[match.queryIdx].pt[0]), int(keypoints1[match.queryIdx].pt[1]))
            pt2 = (int(keypoints2[match.trainIdx].pt[0]) + self.I1.shape[1], int(keypoints2[match.trainIdx].pt[1]))
            cv2.line(matched_image, pt1, pt2, (0, 255, 0), 1)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
        plt.title("Matched Keypoints")
        plt.axis('off')
        plt.show()

# Main execution
def main():
    try:
        # Initialize benchmark
        benchmark = StereoProcessingBenchmark("toffifee-eye-left.jpg", "toffifee-eye-right.jpg")

        # Run stereo processing
        try:
            I1_rect, I2_rect, disparity, keypoints1, keypoints2, matches = benchmark.run_sift_stereo_processing()
        except Exception as processing_error:
            print(f"Stereo processing failed: {processing_error}")
            return

        # Generate performance report
        performance_report = benchmark.generate_performance_report()

        # Print performance report
        print("Performance Report:")
        for key, value in performance_report.items():
            print(f"{key}: {value}")

        # Visualize performance
        benchmark.visualize_performance()

        # Visualize matched keypoints
        benchmark.visualize_matches(keypoints1, keypoints2, matches)

        # Visualization of results
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.title('Rectified Left Image')
        plt.imshow(cv2.cvtColor(I1_rect, cv2.COLOR_BGR2GRAY), cmap='gray')

        plt.subplot(132)
        plt.title('Rectified Right Image')
        plt.imshow(cv2.cvtColor(I2_rect, cv2.COLOR_BGR2GRAY), cmap='gray')

        plt.subplot(133)
        plt.title('Disparity Map')
        plt.imshow(disparity, cmap='jet')
        plt.colorbar()

        plt.tight_layout()
        plt.show()

    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()