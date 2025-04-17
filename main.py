import os
import numpy as np
from PIL import Image

# مسیر پوشه تصاویرت رو اینجا بذار
image_folder = 'dataset/Train'

# سایز جدید تصاویر
target_size = (32, 32)

# لیست برای ذخیره آرایه‌ها
image_arrays = []

# تکرار روی همه فایل‌ها
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(image_folder, filename)
        img = Image.open(img_path).convert('RGB')
        img = img.resize(target_size)           
        img_array = np.array(img) / 255.0
        img_array = img_array.astype('float32')
        image_arrays.append(img_array)

X = np.stack(image_arrays)

print("Dataset shape:", X.shape)
