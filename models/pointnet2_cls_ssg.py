import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction


class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)


        return x, l3_points

    def extract_features(self, xyz, pre_maxpool=False, feature_level='global'):
        """
        Extract features from different levels of the network
        
        Args:
            xyz (torch.Tensor): Input point cloud [B, C, N]
            pre_maxpool (bool): If True, return features before maxpooling
            feature_level (str): Level of features to extract
                            'global': global features (1024-dim) [default]
                            'local': local features from each SA layer
                            'all': both global and local features
        
        Returns:
            dict: Dictionary containing requested features
        """
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        # Get features from each SA layer
        if pre_maxpool:
            l1_xyz, l1_points, l1_pre_max = self.sa1(xyz, norm, return_pre_maxpool=True)
            l2_xyz, l2_points, l2_pre_max = self.sa2(l1_xyz, l1_points, return_pre_maxpool=True)
            l3_xyz, l3_points, l3_pre_max = self.sa3(l2_xyz, l2_points, return_pre_maxpool=True)
        else:
            l1_xyz, l1_points = self.sa1(xyz, norm)
            l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
            l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # Global features
        global_features = l3_points.view(B, 1024)
        
        results = {}
        
        if feature_level == 'global':
            results['global_features'] = global_features
            if pre_maxpool:
                results['pre_maxpool_features'] = l3_pre_max
        elif feature_level == 'local':
            results = {
                'sa1_features': l1_points,
                'sa2_features': l2_points,
                'sa3_features': l3_points,
                'sa1_xyz': l1_xyz,
                'sa2_xyz': l2_xyz,
                'sa3_xyz': l3_xyz
            }
            if pre_maxpool:
                results['sa1_pre_maxpool'] = l1_pre_max
                results['sa2_pre_maxpool'] = l2_pre_max
                results['sa3_pre_maxpool'] = l3_pre_max
        elif feature_level == 'all':
            results = {
                'global_features': global_features,
                'sa1_features': l1_points,
                'sa2_features': l2_points,
                'sa3_features': l3_points,
                'sa1_xyz': l1_xyz,
                'sa2_xyz': l2_xyz,
                'sa3_xyz': l3_xyz
            }
            if pre_maxpool:
                results['sa1_pre_maxpool'] = l1_pre_max
                results['sa2_pre_maxpool'] = l2_pre_max
                results['sa3_pre_maxpool'] = l3_pre_max
        else:
            raise ValueError(f"Unknown feature_level: {feature_level}")
            
        return results

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
