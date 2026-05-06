import cv2
import torch
import torch.nn as nn
import torchvision
import numpy as np
import os
import torchvision.transforms as transforms

# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE UFLDv2 ARCHITECTURE (No external dependencies)
# ─────────────────────────────────────────────────────────────────────────────
class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        model = torchvision.models.resnet18(pretrained=False)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x2, x3, x4

class UFLDv2Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_grid_row = 200
        self.num_cls_row = 72
        self.num_grid_col = 100
        self.num_cls_col = 81
        self.num_lane_on_row = 4
        self.num_lane_on_col = 4
        
        self.dim1 = self.num_grid_row * self.num_cls_row * self.num_lane_on_row
        self.dim2 = self.num_grid_col * self.num_cls_col * self.num_lane_on_col
        self.dim3 = 2 * self.num_cls_row * self.num_lane_on_row
        self.dim4 = 2 * self.num_cls_col * self.num_lane_on_col
        self.total_dim = self.dim1 + self.dim2 + self.dim3 + self.dim4
        
        mlp_mid_dim = 2048
        self.input_dim = (320 // 32) * (1600 // 32) * 8  # 4000
        
        self.model = ResNetBackbone()
        self.cls = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, mlp_mid_dim),
            nn.ReLU(),
            nn.Linear(mlp_mid_dim, self.total_dim),
        )
        self.pool = nn.Conv2d(512, 8, 1)

    def forward(self, x):
        x2, x3, fea = self.model(x)
        fea = self.pool(fea)
        fea = fea.view(-1, self.input_dim)
        out = self.cls(fea)

        return {
            'loc_row': out[:,:self.dim1].view(-1, self.num_grid_row, self.num_cls_row, self.num_lane_on_row), 
            'loc_col': out[:,self.dim1:self.dim1+self.dim2].view(-1, self.num_grid_col, self.num_cls_col, self.num_lane_on_col),
            'exist_row': out[:,self.dim1+self.dim2:self.dim1+self.dim2+self.dim3].view(-1, 2, self.num_cls_row, self.num_lane_on_row), 
            'exist_col': out[:,-self.dim4:].view(-1, 2, self.num_cls_col, self.num_lane_on_col)
        }

# ─────────────────────────────────────────────────────────────────────────────
# Configuration & State 
# ─────────────────────────────────────────────────────────────────────────────
UFLD_WEIGHTS = os.path.join("models", "culane_res18.pth")

TRAIN_WIDTH = 1600
TRAIN_HEIGHT = 320
CROP_RATIO = 0.6
ROW_ANCHORS = np.linspace(0.42, 1, 72)
COL_ANCHORS = np.linspace(0, 1, 81)

_lane_net = None
_device   = None
_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def _load_ufld():
    global _lane_net, _device
    if _lane_net is not None:
        return True

    if not os.path.exists(UFLD_WEIGHTS):
        print(f"⚠️ UFLD weights missing at {UFLD_WEIGHTS}. Lane detection disabled.")
        return False

    try:
        if torch.cuda.is_available():
            _device = torch.device('cuda:0')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            _device = torch.device('mps')
        else:
            _device = torch.device('cpu')

        print(f"[INFO] Loading UFLDv2 Lane Detection on {_device}...")
        
        _lane_net = UFLDv2Net().to(_device)

        state_dict = torch.load(UFLD_WEIGHTS, map_location='cpu')['model']
        compatible_state_dict = {k[7:] if 'module.' in k else k: v for k, v in state_dict.items()}
        _lane_net.load_state_dict(compatible_state_dict, strict=False)
        _lane_net.eval()
        
        print("✅ Ultra-Fast-Lane-Detection-V2 (ResNet18) loaded successfully.")
        return True
    except Exception as e:
        print(f"❌ Failed to load UFLDv2: {e}")
        _lane_net = None
        return False

def pred2coords(pred, original_image_width, original_image_height, local_width=1):
    batch_size, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
    batch_size, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape

    max_indices_row = pred['loc_row'].argmax(1).cpu()
    valid_row = pred['exist_row'].argmax(1).cpu()

    max_indices_col = pred['loc_col'].argmax(1).cpu()
    valid_col = pred['exist_col'].argmax(1).cpu()

    pred['loc_row'] = pred['loc_row'].cpu()
    pred['loc_col'] = pred['loc_col'].cpu()

    coords = []
    row_lane_idx = [1, 2]
    col_lane_idx = [0, 3]

    for i in row_lane_idx:
        tmp = []
        if valid_row[0, :, i].sum() > num_cls_row / 2:
            for k in range(valid_row.shape[1]):
                if valid_row[0, k, i]:
                    all_ind = torch.tensor(list(range(max(0, max_indices_row[0, k, i] - local_width), min(num_grid_row - 1, max_indices_row[0, k, i] + local_width) + 1)))
                    out_tmp = (pred['loc_row'][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_row - 1) * original_image_width
                    tmp.append((int(out_tmp), int(ROW_ANCHORS[k] * original_image_height)))
        if tmp: coords.append(tmp)

    for i in col_lane_idx:
        tmp = []
        if valid_col[0, :, i].sum() > num_cls_col / 4:
            for k in range(valid_col.shape[1]):
                if valid_col[0, k, i]:
                    all_ind = torch.tensor(list(range(max(0, max_indices_col[0, k, i] - local_width), min(num_grid_col - 1, max_indices_col[0, k, i] + local_width) + 1)))
                    out_tmp = (pred['loc_col'][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_col - 1) * original_image_height
                    tmp.append((int(COL_ANCHORS[k] * original_image_width), int(out_tmp)))
        if tmp: coords.append(tmp)

    return coords

def detect_lanes_data(frame: np.ndarray):
    if not _load_ufld() or frame is None:
        return {}
    img_h, img_w = frame.shape[:2]
    resize_h = int(TRAIN_HEIGHT / CROP_RATIO)
    img = cv2.resize(frame, (TRAIN_WIDTH, resize_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = _transform(img)
    tensor = tensor[:, -TRAIN_HEIGHT:, :].unsqueeze(0).to(_device)
    with torch.no_grad():
        pred = _lane_net(tensor)
    lanes = pred2coords(
        pred,
        original_image_width=img_w,
        original_image_height=img_h
    )
    lane_dict = {}
    for i, lane in enumerate(lanes):
        # Skip invalid lanes
        if lane is None or len(lane) < 2:
            continue
        try:
            # Convert to OpenCV-compatible format
            lane_arr = np.array(lane, dtype=np.int32)
            # Required shape for OpenCV polylines
            lane_arr = lane_arr.reshape((-1, 1, 2))
            lane_dict[i] = lane_arr
        except Exception as e:
            print(f"[lane] Failed processing lane {i}: {e}")
    return lane_dict


def detect_lanes_image(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    lane_dict = detect_lanes_data(img)
    for lane_id, lane_points in lane_dict.items():
        # Additional safety check
        if lane_points is None or len(lane_points) < 2:
            continue
        try:
            cv2.polylines(
                img,
                [lane_points],
                isClosed=False,
                color=(0, 255, 0),
                thickness=5
            )
        except Exception as e:
            print(f"[draw] Lane {lane_id} error: {e}")
    return img