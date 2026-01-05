import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import re  
from PIL import Image

# **設定資料夾**
input_folder = "dataset_folder"  # 原始圖片資料夾
output_folder = "processed_images"  # 切割後的圖片儲存資料夾
model_path = "digit_recognition_model.h5"  # 模型存檔位置
os.makedirs(output_folder, exist_ok=True)  # 確保輸出資料夾存在

# **建立 CNN 模型**
print("🆕 建立新 CNN 模型...")
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

processed_count = 0  # 記錄已處理的圖片數量
image_index = 0  # 記錄目前處理到的第幾張圖片

# **函式：切割圖片**
def split_image(image_path, output_folder):
    start, end, num_slices = 7, 83, 5  
    image = Image.open(image_path)
    width, height = image.size

    crop_width = end - start
    slice_width = crop_width // num_slices

    filename = os.path.basename(image_path)
    name_parts = filename.split("_")

    image_time = f"{name_parts[0]}_{name_parts[1]}_"
    image_name = os.path.splitext(name_parts[2])[0]  

    output_files = []  

    for i in range(num_slices):
        left = start + i * slice_width
        right = left + slice_width if i < num_slices - 1 else end
        
        cropped = image.crop((left, 0, right, height))
        suffix = image_name[i] if i < len(image_name) else str(i)
        save_path = os.path.join(output_folder, f"{i}_{image_time}{suffix}.jpg")
        cropped.save(save_path)
        output_files.append(save_path)

    return output_files  

# **逐張處理並訓練**
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_index += 1  # 記錄當前處理的第幾張圖片
        print(f"📷 正在處理第 {image_index} 張圖片",end="")
        
        image_path = os.path.join(input_folder, filename)
        cropped_files = split_image(image_path, output_folder)  

        image_data = []
        labels = []
        
        for cropped_file in cropped_files:
            match = re.findall(r'(\d+)', cropped_file)  
            if match:
                label = int(match[-1])  
                img = cv2.imread(cropped_file, cv2.IMREAD_GRAYSCALE)  
                img = cv2.resize(img, (28, 28))  
                img = img / 255.0  
                image_data.append(img)
                labels.append(label)
        
        if image_data:
            image_data = np.array(image_data).reshape(-1, 28, 28, 1)  
            labels = np.array(labels)

            model.fit(image_data, labels, epochs=2, batch_size=5, verbose=0)  # 隱藏詳細輸出
            processed_count += len(image_data)  

            print(f"✅ 已訓練 {processed_count} 張切割後的圖片")

# **存檔（不包含 optimizer）**
model.save(model_path, include_optimizer=False)  
print(f"💾 訓練完成，模型已儲存至 {model_path}（不包含 optimizer）")
print(f"📊 總共處理並訓練了 {processed_count} 張圖片！")
