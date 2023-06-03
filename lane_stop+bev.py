#!/usr/bin/env python3 
import rclpy
from rclpy.node import Node
import cv2
import numpy as np

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
bridge = CvBridge()
cv_image = np.empty(shape=[0])

class LaneFollower(Node):

    def __init__(self):
        super().__init__('lane_follower')
        self.image_sub = self.create_subscription(Image, "/image_raw", self.callback, 10)
        self.laser_sub = self.create_subscription(LaserScan, "/scan", self.laser_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/vuasrl_car/cmd_vel', 10)
        self.img_bgr = None
        self.min_distance = float("inf")

    def laser_callback(self, msg):
        angle_increment = msg.angle_increment
        num_readings = len(msg.ranges)

         # Define the angle range to focus on the front of the robot
        front_angle_range = 30  # degrees, change this value as needed

        front_index_range = int(front_angle_range / 2 / (angle_increment * 180 / np.pi))

        front_ranges = msg.ranges[(num_readings // 2) - front_index_range: (num_readings // 2) + front_index_range]

         # Filter out 'inf' values, which may occur if the sensor can't detect an obstacle
        front_ranges_filtered = [r for r in front_ranges if r != float('inf')]

        if front_ranges_filtered:
             self.min_distance = min(front_ranges_filtered)
        else:
             self.min_distance = float('inf')

        print("Minimum distance: ", self.min_distance)
        
    def callback(self, data):
        
        global cv_image
        cv_image = bridge.imgmsg_to_cv2(data, 'bgr8')
        frame = cv_image
        img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
#----------------------------bev--------------------------------------#
        p1 = [250, 50] # 좌하
        p2 = [390, 50] # 우하
        p3 = [400, 430] # 우상
        p4 = [240, 430] # 좌상

        # corners_point_arr는 변환 이전 이미지 좌표 4개 
        corner_points_arr = np.float32([p1, p2, p3, p4])
        height, width = img_hsv.shape[:2]

        image_p1 = [0, 0]
        image_p2 = [width, 0]
        image_p3 = [width, height]
        image_p4 = [0, height]

        image_params = np.float32([image_p1, image_p2, image_p3, image_p4])

        mat = cv2.getPerspectiveTransform(corner_points_arr, image_params)
        # mat = 변환행렬(3*3 행렬) 반
        image_transformed = cv2.warpPerspective(img_hsv, mat, (width, height))
#------------------------------------------------------------------#    

        lower_wlane = np.array([0,0,185])
        upper_wlane = np.array([30,60,255])

        lower_ylane = np.array([10,100,100])
        upper_ylane = np.array([40,255,255])

        img_wlane = cv2.inRange(img_hsv, lower_wlane, upper_wlane)
        img_ylane = cv2.inRange(img_hsv, lower_ylane, upper_ylane)

        # combine the two binary images
        img_lane = cv2.bitwise_or(img_wlane, img_ylane)

        # apply morphological operations to reduce noise
        kernel = np.ones((5,5), np.uint8)
        img_lane = cv2.morphologyEx(img_lane, cv2.MORPH_OPEN, kernel)
        img_lane = cv2.morphologyEx(img_lane, cv2.MORPH_CLOSE, kernel)

        # find contours
        contours, hierarchy = cv2.findContours(img_lane, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # find left and right edges
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            # find edges using Canny edge detection
            img_gray = cv2.cvtColor(image_transformed, cv2.COLOR_BGR2GRAY)
            img_edges = cv2.Canny(img_gray, 100, 200)
            
            left_edge = np.argmax(img_edges[cy,:cx])
            right_edge = cx + np.argmax(img_edges[cy,cx:])

            # calculate the midpoint between the left and right edges
            midpoint = (left_edge + right_edge) // 2
            
            # draw left and right edges
            cv2.line(image_transformed, (left_edge, cy), (cx, cy), (0, 255, 0), 2)
            cv2.line(image_transformed, (cx, cy), (right_edge, cy), (0, 255, 0), 2)
            
            # draw the midpoint as a vertical line
            cv2.line(image_transformed, (midpoint, 0), (midpoint, image_transformed.shape[0]), (0, 0, 255), 2)

            # move the robot based on the position of the center of the lane and check for obstacles
            error = cx - image_transformed.shape[1]/2
            twist = Twist()
            ##
            twist.linear.x = 5.0

            while self.min_distance <= 1.5: # Change this value according to the desired stopping distance
                twist.linear.z = 10.0
                twist.linear.x = 0.0
                self.get_logger().info('STOP')
                break
                
            
            twist.angular.z = -float(error) / 100
            self.cmd_vel_pub.publish(twist)    
         
        # display the image
        cv2.imshow("Lane Detection", image_transformed)
        #cv2.imshow("Lane Detection", frame)
        cv2.waitKey(1) 
        
def main(args=None):
    rclpy.init(args=args)
    lane_follower = LaneFollower()
    try:
        rclpy.spin(lane_follower)
    except KeyboardInterrupt:
        lane_follower.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        lane_follower.destroy_node()
        rclpy.shutdown()
        
if __name__ == '__main__':
  main()