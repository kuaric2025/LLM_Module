#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

from tf2_msgs.msg import TFMessage       

from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PointStamped, PoseStamped, TransformStamped
from std_msgs.msg import Float32MultiArray
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point

from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
# from tf_transformations import quaternion_from_euler
import numpy as np

import os
from io import BytesIO
# from PIL import Image, ImageDraw, ImageFont
from PIL import Image as CVImage
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

from backend_segmentation import generate_segmentation, draw_polygons

import random
import numpy as np
import copy

from sensor_msgs.msg import JointState
from ik_solver_pkg.srv import SetTargetPose
from geometry_msgs.msg import Pose

from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor

import json
import time

from utils import get_pca_vertical_axis_from_mask, angle_to_quaternion_z, wrap_to_pi, get_pca_orientation_from_mask
import sys
sys.path.append('/home/kucarst3-dlws/miniconda3/envs/isaac_rl/lib/python3.10/site-packages')

# ─────────── credentials (set env var or hard-code) ───────────
# set your OPENAI_APY_KEY

# ─────────────── LangChain LLM (prompt polishing) ─────────────
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

#=================load json file===========================
json_str = '/home/kucarst3-dlws/rs_ws/src/ik_solver_pkg/ik_solver_pkg/ur10_subtasks.json'

# Load JSON
with open(json_str, 'r', encoding='utf-8') as f:
    data = json.load(f)
    
# Extract keywords
steps = []
actions = []
objects = []
objects_name = set()
locations_from = []
locations_to = []

for item in data:
    steps.append(item["step"])
    actions.append(item["action"])
    objects.append(item["object"])
    objects_name.add(item["object"])
    locations_from.append(item["from"])
    locations_to.append(item["to"])

# florence2
# user_prompt = "segment the driller" #
# user_prompt = input("Describe what to segment (e.g. 'segment the red car'): ").strip()
task_prompt = '<REFERRING_EXPRESSION_SEGMENTATION>'
colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red',
            'lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue']
#=================================================================================
class ObjectToBaseTransformer(Node):
    def __init__(self):
        super().__init__('object_pixel_to_base_transformer')

        # TF and frames
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.camera_frame = 'camera_link'
        self.base_frame = 'base_link'
        self.depth = 1

        self.K = None# Camera intrinsics
        self.disp = None
        self.cv_image = None
        self.depth_image = None
        self.bridge = CvBridge()
        
        focal_length = 18.14756
        W = 1280
        H = 720
        h_aperture = 20.955
        v_aperture = 15.2908
        h_offset = 0.0
        v_offset = 0.0
        fx = focal_length*W/h_aperture
        fy = focal_length*H/v_aperture
        cx = W*0.5 + h_offset/h_aperture*W
        cy = H*0.5 + v_offset/v_aperture*H
        
        self.K = [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ]
        
        self.latest_pose = None
        self.processing = False
        
        # Subscribe to topics
        self.create_subscription(CameraInfo, '/camera_info', self.camera_info_callback, 10)
        # self.create_subscription(Float32MultiArray, '/rgb', self.pixel_callback, 10)
        self.create_subscription(Image, '/depth', self.depth_callback, 10)
        # self.polygons = 
        self.create_subscription(Image, '/rgb', self.rgb_callback, 10)
        
        
        self.create_subscription(TFMessage, '/tf', self.tf_callback, 10)
        self.seg_pub = self.create_publisher(Image, '/seg_image/compressed', 10)
        
        self.get_logger().info("Waiting for camera intrinsics and object pixel data...") 
        
        # self.goal_pub = self.create_publisher(PoseStamped, '/ur10_target_pose', 10)
        
        # Publish static TF for camera_link
        self.broadcaster = StaticTransformBroadcaster(self)
        self.publish_camera_static_tf()
        self.publish_bridge_tf()
        
        
        self.joint_pub= self.create_publisher(JointState, '/isaac_joint_states', 10)
        self.gripper_pub = self.create_publisher(JointState, '/isaac_gripper_state', 10)
 
        # Timer: publish every 1 second (or change to 0.1 for faster) 
        # self.timer = self.create_timer(1.0, self.rgb_callback) 
        self.timer = self.create_timer(0.5, self.timer_callback)

        self.client = self.create_client(SetTargetPose, '/set_target_pose')
        while not self.client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('⏳ Waiting for IK service...')
        self.get_logger().info('✅ Connected to IK service!')
        
        self.step_count = 0
        self.steps = steps
        self.actions = actions
        self.objects = objects
        self.locations_from = locations_from
        self.locations_to = locations_to
        
        self.object_masks = {}
        self.object_centers = {}
        self.object_orientations = {}
        # self.object_depths = {}
        self.objects_name = list(objects_name)
        self.objects_count = 0
        
        self.place_locations = {}
        self.object_height = {
            'yellow block': 0.094,
            'blue block': 0.0846,
            'green block': 0.0752,
            'red block': 0.0564,
        }
        self.tcp_offset_z = 0.19
        
        self.drop_height = {
            'yellow block': 0.094,
            'blue block': 0.1786,
            'green block': 0.2538,
            'red block': 0.3102,
        }
        self.sort_height = arr = np.zeros(len(self.object_height))
        
        self.localization = True
        
    def publish_camera_static_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'camera_link'

        t.transform.translation.x = 0.5
        t.transform.translation.y = 0.0
        t.transform.translation.z = 3.0

        t.transform.rotation.x = 1.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 0.0

        self.broadcaster.sendTransform(t)
        self.get_logger().info("Published static TF: world → camera_link")

    def publish_bridge_tf(self):
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = 'world'
        tf.child_frame_id = 'World'

        tf.transform.translation.x = 0.0
        tf.transform.translation.y = 0.0
        tf.transform.translation.z = 0.0
        tf.transform.rotation.x = 0.0
        tf.transform.rotation.y = 0.0
        tf.transform.rotation.z = 0.0
        tf.transform.rotation.w = 1.0

        self.broadcaster.sendTransform(tf)
        self.get_logger().info("🔗 Published static TF: world → World")
        
    def tf_callback(self, msg: TFMessage):
        for transform in msg.transforms:
            t = transform.transform.translation
            r = transform.transform.rotation
            # self.get_logger().info(
            #     f"Frame ID: {transform.header.frame_id} → {transform.child_frame_id}\n"
            #     f"  Translation: x={t.x:.3f}, y={t.y:.3f}, z={t.z:.3f}\n"
            #     f"  Rotation (quat): x={r.x:.3f}, y={r.y:.3f}, z={r.z:.3f}, w={r.w:.3f}"
            # )
            
    def depth_callback(self, msg:Image):
        # The encoding for Isaac Sim depth is typically '32FC1' (float32)
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            # self.get_logger().info(f"Depth image received. Shape: {self.depth_image.shape}, dtype: {self.depth_image.dtype}")
            # Print value at center pixel
            h, w = self.depth_image.shape
            center_value = self.depth_image[h//2, w//2]
            # self.get_logger().info(f"Center pixel depth: {center_value:.4f} meters")
        except Exception as e:
            self.get_logger().error(f"Error converting depth image: {e}")
            
    def camera_info_callback(self, msg: CameraInfo):
        self.K = np.array(msg.k).reshape(3, 3)
        self.get_logger().info("Camera intrinsics received.")
        # [1.1085125e+03 0.0000000e+00 6.4000000e+02]
        # [0.0000000e+00 1.1085125e+03 3.6000000e+02]
        # [0.0000000e+00 0.0000000e+00 1.0000000e+00]
        # self.get_logger().info(f"self.K: {self.K} \n")
        self.destroy_subscription(self.camera_info_callback)  # only needed once
       
    def rgb_callback(self, msg: Image):
        # self.get_logger().info("rgb_callback \n")
        try:
            if self.localization:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8') #'bgr8') #
                # cv2.imwrite('input_image.jpg', cv_image)
                # import pdb; pdb.set_trace()
                
                # # cv2.imshow("RGB Stream", cv_image)
                # # cv2.waitKey(1)
                for i in range(len(self.objects_name)):
                    user_prompt = self.objects_name[self.objects_count]
                    # self.get_logger().info(f"user_prompt: {user_prompt} \n")
                    result = generate_segmentation(cv_image, user_prompt, task_prompt)
                    polygons = np.array(result['<REFERRING_EXPRESSION_SEGMENTATION>']['polygons']).reshape(-1,2) # (**, 2)
        
                    # # polygons[:,0].mean()
                    output_image = copy.deepcopy(CVImage.fromarray(cv_image))
                    
                    draw_polygons(output_image, result['<REFERRING_EXPRESSION_SEGMENTATION>'], fill_mask=True)
                    # output_image.show(title="Segmented Image")
                    cv2.imwrite(f'output_image_{user_prompt}.jpg', np.array(output_image))
                    # seg_img = self.bridge.cv2_to_imgmsg(np.array(output_image), encoding='rgb8')
                    # self.seg_pub.publish(seg_img)
                    # img_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                    # cv2.imwrite(f'output_image_{user_prompt}_RGB.jpg', np.array(img_rgb))
                    
                    # import pdb; pdb.set_trace()
                    
                    ## get the orientation
                    mask = np.zeros(self.depth_image.shape, dtype=np.uint8)
                    cv2.fillPoly(mask, [polygons.astype(np.int32)], 1)
                    
                    # # import pdb; pdb.set_trace()
                    center_img, angle_img = get_pca_orientation_from_mask(mask)
                    if center_img is not None and angle_img is not None:
                        self.get_logger().info(f"Object orientation (in image): {np.degrees(angle_img):.2f} degrees")
                    else:
                        self.get_logger().warn("PCA failed on mask; insufficient points")
                    
                    if abs(angle_img) > np.radians(120):
                        # Try flipping (e.g. grasp at opposite orientation)
                        angle_img = wrap_to_pi(angle_img - np.pi)

                    self.object_centers[self.objects_name[self.objects_count]]=center_img
                    self.object_orientations[self.objects_name[self.objects_count]]=angle_img
                    self.object_masks[self.objects_name[self.objects_count]]=polygons
                    
                    # 3. Get median Z value                    
                    depth_values = self.depth_image[mask == 1]

                    if len(depth_values) == 0:
                        self.get_logger().warn("No depth pixels inside the mask. Using fallback Z=1.0")
                        Z = 1.0
                    else:
                        Z = float(np.median(depth_values))
                    self.object_height[self.objects_name[self.objects_count]]=Z
                
                    self.objects_count +=1

                self.localization = False

            if self.K is None: 
                self.get_logger().warn("Camera intrinsics not available yet.")
                return
            
            if self.step_count >=8:
                self.get_logger().warn(f"⚠️ Step count {self.step_count} exceeds total steps {len(self.objects)}. Pausing.")
                return  
            

            # Estimate object center in pixels
            uu = self.object_centers[self.objects[self.step_count]][0] #center_img[0] # polygons[:,0].mean()
            v = self.object_centers[self.objects[self.step_count]][1] #center_img[1] # polygons[:,1].mean()
            Z = self.object_height[self.objects[self.step_count]] ##1.0 #self.depth
            fx, fy = self.K[0][0], self.K[1][1]
            cx, cy = self.K[0][2], self.K[1][2]
            X = (uu - cx) * Z / fx
            Y = (v - cy) * Z / fy
            
            # import pdb; pdb.set_trace()

            point_cam = PointStamped()
            point_cam.header.frame_id = self.camera_frame
            point_cam.header.stamp = self.get_clock().now().to_msg()
            point_cam.point.x = X
            point_cam.point.y = Y
            point_cam.point.z = Z

            t_World_tcp = self.tf_buffer.lookup_transform('base_link', 'World', rclpy.time.Time(), timeout=Duration(seconds=2.0))
            t_world_camera = self.tf_buffer.lookup_transform('world', 'camera_link', rclpy.time.Time(), timeout=Duration(seconds=2.0))
            t_world_World = self.tf_buffer.lookup_transform('World', 'world', rclpy.time.Time(), timeout=Duration(seconds=2.0))

            # Combine transforms manually
            point_world = do_transform_point(point_cam, t_world_camera)
            point_World = do_transform_point(point_world, t_world_World)
            point_tcp = do_transform_point(point_World, t_World_tcp)
            
            # Map 2D image point to 3D target pose (this is mocked here)
            # Replace with actual perception/TF logic
            pose = Pose()
            pose.position.x = point_tcp.point.x #target_pos[0]
            pose.position.y = point_tcp.point.y #target_pos[1]
            pose.position.z = point_tcp.point.z #target_pos[2]
            
            angle_img = self.object_orientations[self.objects[self.step_count]]
            if angle_img is not None:
                # Here: adjust angle_img as needed depending on your camera setup!
                quat = angle_to_quaternion_z(angle_img)
                pose.orientation.x = quat.z
                pose.orientation.y = quat.w
                pose.orientation.z = quat.x
                pose.orientation.w = quat.y
                
            else:
                # fallback
                pose.orientation.x = 1.0
                pose.orientation.y = 0.0
                pose.orientation.z = 0.0
                pose.orientation.w = 0.0
            
            self.latest_pose = pose

        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
    
    
    def timer_callback(self):
        if self.step_count > len(self.steps):
            self.get_logger().info("Task Finish.")
            return  # Exit early to pause further actions
        
        if not self.latest_pose:
            return  # Pose not ready
        
        if self.processing:
            return  # Avoid concurrent execution
    
        action = self.actions[self.step_count]
        object_name = self.objects[self.step_count]
        self.get_logger().info(f"action: {action}, object_name: {object_name}")
        
        if self.latest_pose and not self.processing:
            self.processing = True

        if action == 'pick':
            self.get_logger().info(f"🟢 PICK: Moving to {object_name}")
            self.send_ik_request(self.latest_pose, action='pick')

        elif action == 'place':
            self.get_logger().info(f"🔵 PLACE: Moving to target place for {object_name}")
            # Optional: look up destination polygon from object/locations_to
            self.send_ik_request(self.latest_pose, action='place')

        else:
            self.get_logger().warn(f"⚠️ Unknown action '{action}'")

        self.latest_pose = None
        self.processing = False
                  
    def send_ik_request(self, pose, action): ## TO DO: Planar + vertical movement
        if not self.client.service_is_ready():
            self.get_logger().warn("IK service not ready.")
            return
        
        # Set some heights (adjust as needed)
        safe_z = 0.5      # above the object
        grasp_z = pose.position.z  # actual grasping z (from perception)
        approach_offset = 0.06     # distance above object to approach

        if action == 'pick':
            # ---- Step 1: Move in plane to above object ----
            above_pose = Pose()
            above_pose.position.x = pose.position.x
            above_pose.position.y = pose.position.y
            above_pose.position.z = safe_z  # safe height above object
            above_pose.orientation = pose.orientation

            self._call_ik_and_wait(above_pose, description="Above object (pre-grasp)")
            
            # ---- Step 2: Move down to object (grasp height) ----
            down_pose = Pose()
            down_pose.position.x = pose.position.x
            down_pose.position.y = pose.position.y
            down_pose.position.z = grasp_z + self.tcp_offset_z
            down_pose.orientation = pose.orientation

            self._call_ik_and_wait(down_pose, description="At grasp height")
            
            # ---- Step 3: Close gripper ----
            self.close_gripper()
            time.sleep(1.2)
            
            # ---- Step 4: Move up to safe Z ----
            up_pose = Pose()
            up_pose.position.x = pose.position.x
            up_pose.position.y = pose.position.y
            up_pose.position.z = safe_z
            up_pose.orientation = pose.orientation

            self.sort_height[int(self.step_count/2)] = pose.position.z
            self.get_logger().info(f"sort_height_H: {pose.position.z}")

            self._call_ik_and_wait(up_pose, description="Lift object")
        elif action=='place':
            pose.position.x = 0.7
            pose.position.y = 0.2
            
            # ---- Step 1: Move in plane to above object ----
            above_pose = Pose()
            above_pose.position.x = pose.position.x
            above_pose.position.y = pose.position.y
            above_pose.position.z = safe_z
            above_pose.orientation = pose.orientation
            
            self._call_ik_and_wait(above_pose, description="Above place location")
            
            # ---- Step 2: Move down to top of other object (drop height) ----
            down_pose = Pose()
            down_pose.position.x = pose.position.x
            down_pose.position.y = pose.position.y
            # down_pose.position.z = self.drop_height[self.objects[self.step_count]] + self.tcp_offset_z + approach_offset
            down_pose.position.z = sum(self.sort_height) + self.tcp_offset_z + approach_offset
            # import pdb; pdb.set_trace()
            self.get_logger().info(f"down_pose_H: {down_pose.position.z}")
            down_pose.orientation = pose.orientation

            self._call_ik_and_wait(down_pose, description="At place height")
            
            # ---- Step 3: Close gripper ----
            self.open_gripper()
            time.sleep(1.2)

            # ---- Step 4: Move up to safe Z ---
            self._call_ik_and_wait(above_pose, description="Retreat after place")
        
        
        # 🔁 Give time for robot to move (adjust seconds as needed)
        time.sleep(5.0) 
        
        self.processing = True
        self.get_logger().info("called async")
        
        self.step_count += 1
            
    def _call_ik_and_wait(self, pose, description=""):
        # Helper to call IK and wait for result
        request = SetTargetPose.Request()
        request.target_pose.position.x = pose.position.x 
        request.target_pose.position.y = pose.position.y 
        request.target_pose.position.z = pose.position.z 
        request.target_pose.orientation.x = pose.orientation.x
        request.target_pose.orientation.y = pose.orientation.y
        request.target_pose.orientation.z = pose.orientation.z
        request.target_pose.orientation.w = pose.orientation.w
        request.execute = True
        
        yaw_rad = 2 * np.arctan2(pose.orientation.x, pose.orientation.y)
        yaw_deg = np.degrees(yaw_rad)
        
        self.get_logger().info(f"IK to {description}: x={pose.position.x:.3f}, y={pose.position.y:.3f}, z={pose.position.z:.3f}, ori={yaw_deg}")
        # future = self.client.call_async(request)

        
        self.get_logger().info(f"Pose to IK: x={pose.position.x:.3f}, y={pose.position.y:.3f}, z={pose.position.z:.3f}")
        self.pending_future = self.client.call_async(request)
        self.get_logger().info(f"self.pending_future: {self.pending_future}")
        
        def handle_response(fut):
            result = fut.result()
            if result and result.success:
                self.get_logger().info(f"✅ IK Success: {result.message}")
            else:
                self.get_logger().error("❌ IK call failed or no result.")

        self.pending_future.add_done_callback(handle_response)
        
        # 🔁 Give time for robot to move (adjust seconds as needed)
        time.sleep(5.0) 
        
            

    def close_gripper(self):
        self.get_logger().info("Close gripper.")
        msg = JointState()
        position = 0.8
        msg.name = ['finger_joint']
        msg.position = [position]
        self.gripper_pub.publish(msg)
        self.get_logger().info(f'Sent finger_joint position: {position}')
        
    def open_gripper(self):
        self.get_logger().info("Open gripper.")
        msg = JointState()
        position = 0.0
        msg.name = ['finger_joint']
        msg.position = [position]
        self.gripper_pub.publish(msg)
        self.get_logger().info(f'Sent finger_joint position: {position}')

def main(args=None):
    # rclpy.init(args=args)
    # node = ObjectToBaseTransformer()
    # rclpy.spin(node)
    # node.destroy_node()
    # rclpy.shutdown()
    
    rclpy.init(args=args)
    node = ObjectToBaseTransformer()
    
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


# def main(args=None):
#     rclpy.init(args=args)
#     # node = ImageSubscriber()
#     node = TFSubscriber()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()
#     cv2.destroyAllWindows()


#############
# camera --> world:
#    0.5, 0.0, 3.0
#    0.0, 0.0, 0.0//1.0 0.0 0.0 0.0
#    0.5, 2.0, 1.0 
