import sys
import os

import torch
from PIL import Image
from torchvision import transforms

# 添加项目根目录到 PYTHONPATH，确保可以导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入 UNet 和 Pix2Pix 模型
from unet.nets.unet import Unet
from pix2pix.models.pix2pix_model import Pix2PixModel

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 初始化 UNet 模型
unet_model = Unet(num_classes=1)
unet_model_path = r'E:\Software\pycharm\project3\unet_pix2pix\unet\logs\best_epoch_weights.pth'  # 修改为实际 UNet 权重文件路径
unet_model.load_state_dict(torch.load(unet_model_path, map_location=device))
unet_model.to(device)
unet_model.eval()

# 2. 初始化 Pix2Pix 模型
pix2pix_model = Pix2PixModel()
pix2pix_model_path = r'E:\Software\pycharm\project3\unet_pix2pix\pix2pix\latest_net_G.pth'  # 修改为实际 Pix2Pix 权重文件路径
pix2pix_model.netG.load_state_dict(torch.load(pix2pix_model_path, map_location=device))
pix2pix_model.netG.to(device)
pix2pix_model.netG.eval()

# 3. 定义预处理函数
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# 4. 读取输入图像
input_image_path = r'E:\Software\pycharm\project3\unet_pix2pix\0001.jpg'  # 请确保路径正确
input_image = Image.open(input_image_path).convert('RGB')
input_tensor = transform(input_image).unsqueeze(0).to(device)

# 5. 使用 UNet 进行语义分割
with torch.no_grad():
    segmentation_output = unet_model(input_tensor)
    segmentation_output = (segmentation_output > 0.5).float()  # 二值化处理

# 6. 使用 Pix2Pix 进行风格转换
with torch.no_grad():
    pix2pix_output = pix2pix_model.netG(segmentation_output)

# 7. 转换为可视化的图像格式并保存
output_image = pix2pix_output.squeeze().cpu().detach()
output_image = transforms.ToPILImage()(output_image)
output_image.save('output_image.jpg')

print("图像生成完成，已保存为 output_image.jpg")
