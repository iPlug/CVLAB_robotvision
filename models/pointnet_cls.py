import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer

class get_model(nn.Module):
    def __init__(self, k=40, normal_channel=True):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x, trans_feat
    
    def extract_features(self, x, feature_level=None, pre_maxpool=False):
        """
        Extract features from the network
        
        Args:
            x (torch.Tensor): Input point cloud [B, C, N]
            get_pre_maxpool (bool): If True, return features before maxpooling
        
        Returns:
            dict: Dictionary containing requested features
        """

        if pre_maxpool:
            global_feat, trans, trans_feat, pre_maxpool_features = self.feat(x, return_pre_maxpool=True)
            return {
                'global_features': global_feat,
                'pre_maxpool_features': pre_maxpool_features,
                'transform': trans,
                'transform_feat': trans_feat
            }
        else:
            global_feat, trans, trans_feat = self.feat(x)
            return {
                'global_features': global_feat,
                'transform': trans,
                'transform_feat': trans_feat
            }

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss
