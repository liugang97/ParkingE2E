import torch
from torch import nn

from model_interface.model.bev_encoder import BevEncoder, BevQuery
from model_interface.model.gru_trajectory_decoder import GRUTrajectoryDecoder
from model_interface.model.lss_bev_model import LssBevModel
from model_interface.model.trajectory_decoder import TrajectoryDecoder
from utils.config import Configuration


class ParkingModelReal(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()

        self.cfg = cfg

        # Camera Encoder
        self.lss_bev_model = LssBevModel(self.cfg) # 负责将摄像头图像转换为鸟瞰图（BEV，Bird's Eye View）特征。这一步通常涉及到将多视角的图像信息整合成一个统一的视角，以便后续处理。
        self.image_res_encoder = BevEncoder(in_channel=self.cfg.bev_encoder_in_channel)# 进一步处理鸟瞰图特征，提取更高层次的特征信息。in_channel 参数指定了输入通道数，通常与输入图像的特征维度相关。

        # Target Encoder
        self.target_res_encoder = BevEncoder(in_channel=1) # 另一个 BevEncoder 实例，专门用于处理目标特征。

        # BEV Query
        self.bev_query = BevQuery(self.cfg) # 这个模块负责从鸟瞰图特征中提取有用的信息。它可能使用注意力机制（如 Transformer）来关注特定的空间区域或特征。通过查询机制，模型可以动态地选择和聚合特征，以便更好地理解环境和目标。

        # Trajectory Decoder
        self.trajectory_decoder = self.get_trajectory_decoder()# 负责将编码和查询得到的特征信息解码为具体的车辆轨迹。这一步通常涉及到将高维特征转换为低维的控制指令或路径点。

    def forward(self, data):
        # Encoder
        bev_feature, pred_depth, bev_target = self.encoder(data, mode="train")

        # Decoder
        pred_traj_point = self.trajectory_decoder(bev_feature, data['gt_traj_point_token'].cuda())

        return pred_traj_point, pred_depth, bev_target

    def predict_transformer(self, data, predict_token_num):
        # Encoder
        bev_feature, pred_depth, bev_target = self.encoder(data, mode="predict")

        # Auto Regressive Decoder
        autoregressive_point = data['gt_traj_point_token'].cuda() # During inference, we regard BOS as gt_traj_point_token.
        for _ in range(predict_token_num):
            pred_traj_point = self.trajectory_decoder.predict(bev_feature, autoregressive_point)
            autoregressive_point = torch.cat([autoregressive_point, pred_traj_point], dim=1)

        return autoregressive_point, pred_depth, bev_target

    def predict_gru(self, data):
        # Encoder
        bev_feature, _, _ = self.encoder(data, mode="predict")

        # Decoder
        autoregressive_point = self.trajectory_decoder(bev_feature).squeeze()
        return autoregressive_point

    def encoder(self, data, mode):
        # Camera Encoder
        images = data['image'].to(self.cfg.device, non_blocking=True)
        intrinsics = data['intrinsics'].to(self.cfg.device, non_blocking=True)
        extrinsics = data['extrinsics'].to(self.cfg.device, non_blocking=True)
        bev_camera, pred_depth = self.lss_bev_model(images, intrinsics, extrinsics)
        bev_camera_encoder = self.image_res_encoder(bev_camera, flatten=False)
    
        # Target Encoder
        target_point = data['fuzzy_target_point'] if self.cfg.use_fuzzy_target else data['target_point']
        target_point = target_point.to(self.cfg.device, non_blocking=True)
        bev_target = self.get_target_bev(target_point, mode=mode)
        bev_target_encoder = self.target_res_encoder(bev_target, flatten=False)
        
        # Feature Fusion
        bev_feature = self.get_feature_fusion(bev_target_encoder, bev_camera_encoder)

        bev_feature = torch.flatten(bev_feature, 2)

        return bev_feature, pred_depth, bev_target

    def get_target_bev(self, target_point, mode):
        h, w = int((self.cfg.bev_y_bound[1] - self.cfg.bev_y_bound[0]) / self.cfg.bev_y_bound[2]), int((self.cfg.bev_x_bound[1] - self.cfg.bev_x_bound[0]) / self.cfg.bev_x_bound[2])
        b = self.cfg.batch_size if mode == "train" else 1

        # Get target point
        bev_target = torch.zeros((b, 1, h, w), dtype=torch.float).to(self.cfg.device, non_blocking=True)
        x_pixel = (h / 2 + target_point[:, 0] / self.cfg.bev_x_bound[2]).unsqueeze(0).T.int()
        y_pixel = (w / 2 + target_point[:, 1] / self.cfg.bev_y_bound[2]).unsqueeze(0).T.int()
        target_point = torch.cat([x_pixel, y_pixel], dim=1)

        # Add noise
        if self.cfg.add_noise_to_target and mode == "train":
            noise_threshold = int(self.cfg.target_noise_threshold / self.cfg.bev_x_bound[2])
            noise = (torch.rand_like(target_point, dtype=torch.float) * noise_threshold * 2 - noise_threshold).int()
            target_point += noise

        # Get target point tensor in the BEV view
        for batch in range(b):
            bev_target_batch = bev_target[batch][0]
            target_point_batch = target_point[batch]
            range_minmax = int(self.cfg.target_range / self.cfg.bev_x_bound[2])
            bev_target_batch[target_point_batch[0] - range_minmax: target_point_batch[0] + range_minmax + 1,
                             target_point_batch[1] - range_minmax: target_point_batch[1] + range_minmax + 1] = 1.0
        return bev_target
    

    def get_feature_fusion(self, bev_target_encoder, bev_camera_encoder):
        if self.cfg.fusion_method == "query":
            bev_feature = self.bev_query(bev_target_encoder, bev_camera_encoder)
        elif self.cfg.fusion_method == "plus":
            bev_feature = bev_target_encoder + bev_camera_encoder
        elif self.cfg.fusion_method == "concat":
            concat_feature = torch.concatenate([bev_target_encoder, bev_camera_encoder], dim=1)
            conv = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False).cuda()
            bev_feature = conv(concat_feature)
        else:
            raise ValueError(f"Don't support fusion_method '{self.cfg.fusion_method}'!")
        
        return bev_feature
    
    def get_trajectory_decoder(self):
        if self.cfg.decoder_method == "transformer":
            trajectory_decoder = TrajectoryDecoder(self.cfg)
        elif self.cfg.decoder_method == "gru":
            trajectory_decoder = GRUTrajectoryDecoder(self.cfg)
        else:
            raise ValueError(f"Don't support decoder_method '{self.cfg.decoder_method}'!")
        
        return trajectory_decoder