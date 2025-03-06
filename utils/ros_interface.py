import threading

import cv2
import numpy as np
import rospy
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

from utils.keyboard_command import CommandThread

class RosInterface:
    def __init__(self):
        self.bridge = CvBridge() # 桥接器，用于在 OpenCV 图像格式和 ROS 图像消息之间进行转换的工具

        # 初始化图像数据存储变量
        self.cam_front_tensor_img = None #存储来自不同摄像头的图像数据
        self.cam_left_tensor_img = None
        self.cam_right_tensor_img = None
        self.cam_rear_tensor_img = None
        self.pose_info = None # 存储车辆当前的位姿信息
        self.target_point_info = None # 存储目标点信息
        self.rviz_target = None # 存储从 RViz 接收的目标位置

        # 初始化线程锁,为每个共享数据变量创建一个线程锁
        self.cam_front_tensor_img_lock = threading.Lock()
        self.cam_left_tensor_img_lock = threading.Lock()
        self.cam_right_tensor_img_lock = threading.Lock()
        self.cam_rear_tensor_img_lock = threading.Lock()
        self.pose_info_lock = threading.Lock()
        self.target_point_lock = threading.Lock()
        self.rviz_target_lock = threading.Lock()

    def image_thread_function(self, camera_label):
        camera_tag = camera_label.lower().split("_")[1] # 将摄像头标签转换为小写，并提取下划线后的相机类型
        topic_name = "/driver/pinhole_vitual/{}".format(camera_tag)
        callback_str = "self.{}_callback_function".format(camera_tag) # 依据相机便签设置回调函数名
        rospy.Subscriber(topic_name, Image, eval(callback_str), queue_size=1) # 订阅该话题，设置回调函数
        rospy.spin() # spin() 循环，等待并处理图像消息

    def pose_thread_function(self):
        # 订阅 /ego_pose 话题
        rospy.Subscriber("/ego_pose", PoseStamped, self.pose_callback, queue_size=1)

    def target_point_thread_function(self):
        # 订阅两个话题
        rospy.Subscriber("/e2e_parking/set_target_point", Bool, self.target_point_callback, queue_size=1) # 用于设置目标点
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.rviz_target_callback, queue_size=1)

    def image_general_callback(self, msg, bri, img_lock, tag="no"):
        cv_img = bri.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        img_lock.acquire()
        general_img = self.cv_img2tensor_img(cv_img)
        img_lock.release()

        return general_img

    def pose_callback(self, msg: PoseStamped):
        self.pose_info_lock.acquire()
        self.pose_info = msg.pose
        self.pose_info_lock.release()

    def target_point_callback(self, msg):
        self.target_point_info = Pose()
        self.target_point_lock.acquire()
        self.target_point_info = self.pose_info
        self.target_point_lock.release()

    def rviz_target_callback(self, msg):
        self.rviz_target_lock.acquire()
        self.rviz_target = msg.pose
        # print(msg.pose)

        self.rviz_target_lock.release()

    def cv_img2tensor_img(self, img_cv):
        img_cv = np.float32(np.clip(img_cv / 255, 0, 1))
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(np.transpose(img_cv, ((2, 0, 1)))).cuda()
        return img_tensor

    def front_callback_function(self, msg):
        self.cam_front_tensor_img = self.image_general_callback(msg, self.bridge, self.cam_front_tensor_img_lock, tag="rgb_front")

    def left_callback_function(self, msg):
        self.cam_left_tensor_img = self.image_general_callback(msg, self.bridge, self.cam_left_tensor_img_lock, tag="rgb_left")

    def right_callback_function(self, msg):
        self.cam_right_tensor_img = self.image_general_callback(msg, self.bridge, self.cam_right_tensor_img_lock, tag="rgb_right")

    def back_callback_function(self, msg):
        self.cam_rear_tensor_img = self.image_general_callback(msg, self.bridge, self.cam_rear_tensor_img_lock, tag="rgb_rear")

    def receive_info(self):
        threads = [] # 创建一个空列表 threads 用于存储所有创建的线程

        # 创建摄像头图像接收线程
        for camera_label in ["CAM_FRONT", "CAM_LEFT", "CAM_RIGHT", "CAM_BACK"]: 
            # 为每个摄像头创建一个专用线程，目标函数是 image_thread_function，参数是摄像头标签
            image_thread = threading.Thread(target=self.image_thread_function, args=(camera_label, ))
            image_thread.start() # 立即启动该线程
            threads.append(image_thread)
        
        # 创建位姿信息接收线程
        pose_thread = threading.Thread(target=self.pose_thread_function)
        pose_thread.start()
        threads.append(pose_thread)

        # 创建目标点接收线程
        target_point_thread = threading.Thread(target=self.target_point_thread_function)
        target_point_thread.start()
        threads.append(target_point_thread)

        # 创建命令处理线程
        command_thread = CommandThread([])
        command_thread.start()
        threads.append(command_thread)
        return threads
    
    def get_images(self, image_tag):
        if image_tag == "rgb_front":
            return self.cam_front_tensor_img
        elif image_tag == "rgb_rear":
            return self.cam_rear_tensor_img
        elif image_tag == "rgb_left":
            return self.cam_left_tensor_img
        elif image_tag == "rgb_right":
            return self.cam_right_tensor_img
        
    def get_pose(self) -> Pose:
        return self.pose_info

    def get_rviz_target(self) -> PoseStamped:
        return self.rviz_target
