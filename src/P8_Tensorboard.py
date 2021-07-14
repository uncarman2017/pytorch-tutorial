from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
image_path = "dataset/train/ants_image/6240338_93729615ec.jpg"
img_PIL = Image.open(image_path)
# img_PIL.show()
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)  # (H,W,C)
# HWC为高度(H)+宽度(W)+通道(C)格式
writer.add_image("train", img_array, 2, dataformats='HWC')
# y = 2x
for i in range(100):
    writer.add_scalar("y=x", i, i)

writer.close()