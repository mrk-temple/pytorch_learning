from PIL import Image
from torchvision import transforms 
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('logs')
img = Image.open("images/pytorch.jpg")
print(img)

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image('ToTensor', img_tensor)
writer.close()

# Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([9, 8, 5], [2, 1, 1])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image('Normalize', img_norm)
writer.close()

# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
#img_resize PIL -> totensor -> img_resize  tensor
img_resize = trans_totensor(img_resize)
writer.add_image('Resize', img_resize, 0)
writer.close()
print(img_resize)

# Compose - resize - 2
trans_resize_2 = transforms.Resize(512)
# PIL -> PIL -> tensor
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image('Resize', img_resize_2, 1)
writer.close()

# RandomCrop
trans_random_crop = transforms.RandomCrop(512, 1024)
trans_compose_2 = transforms.Compose([trans_random_crop, trans_totensor])
for i in range(10):
    img_random_crop = trans_compose_2(img)
    writer.add_image('RandomCropHW', img_random_crop, i)
writer.close()



# 关注输入和输出
# 看官方文档
#关注需要什么参数 
# 不知道返回值的时候
# print or print(type()) or debug
