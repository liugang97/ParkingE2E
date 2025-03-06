import argparse
import rospy
import utils.fix_libtiff
from model_interface.model_interface import get_parking_model
from utils.config import get_inference_config_obj
from utils.ros_interface import RosInterface

# 主程序入口
if __name__ == "__main__":
    # 创建命令行参数解析器
    arg_parser = argparse.ArgumentParser()
    
    # 添加参数：推理配置文件路径，默认为"./config/inference_real.yaml"
    arg_parser.add_argument('--inference_config_path', default="./config/inference_real.yaml", type=str)
    
    # 解析命令行参数
    args = arg_parser.parse_args()

    # 初始化ROS节点，节点名称为"e2e_traj_pred"
    rospy.init_node("e2e_traj_pred")

    # 创建ROS接口对象
    ros_interface_obj = RosInterface()
    
    # 启动接收信息的线程
    threads = ros_interface_obj.receive_info()

    # 获取推理配置对象
    inference_cfg = get_inference_config_obj(args.inference_config_path)

    # 获取泊车推理模型模块类
    ParkingInferenceModelModule = get_parking_model(data_mode=inference_cfg.train_meta_config.data_mode, run_mode="inference")
    
    # 创建泊车推理对象
    parking_inference_obj = ParkingInferenceModelModule(inference_cfg, ros_interface_obj=ros_interface_obj)
    
    # 执行预测，模式由配置文件中的predict_mode指定
    parking_inference_obj.predict(mode=inference_cfg.predict_mode)

    # 等待所有线程完成
    for thread in threads:
        thread.join()