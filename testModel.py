import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib import rcParams  # 設定全域字型
from PIL import Image

# 設定全域字型
rcParams['font.family'] = 'Microsoft JhengHei'

# 設定變數
model_path = "digit_recognition_model.h5"  # 已訓練的模型
test_image_path = "20250219_082720_97919.jpg"  # 測試圖片
output_folder = "split_images"  # 存放切割後圖片的資料夾

# 確保模型存在
if not os.path.exists(model_path):
    raise FileNotFoundError("❌ 找不到模型，請先訓練並存檔 digit_recognition_model.h5")

# 載入模型
print("🔄 載入模型中...")
model = load_model(model_path)
print("✅ 模型載入完成！")

# 建立輸出資料夾
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 圖片預處理函式
def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")  # 轉換成灰階
    image = image.resize((28, 28))  # 調整為 28x28
    image = np.array(image) / 255.0  # 正規化
    image = image.reshape(1, 28, 28, 1)  # 調整形狀符合 CNN 輸入
    return image

# 圖片切割函式
def split_image(image_path, output_folder):
    start, end, num_slices = 7, 83, 5  # 設定切割範圍與數量
    image = Image.open(image_path)
    width, height = image.size

    crop_width = end - start
    slice_width = crop_width // num_slices

    filename = os.path.basename(image_path)
    name_parts = filename.split("_")

    image_time = f"{name_parts[0]}_{name_parts[1]}_"
    image_name = os.path.splitext(name_parts[2])[0]  # 取檔名數字部分

    output_files = []  

    for i in range(num_slices):
        left = start + i * slice_width
        right = left + slice_width if i < num_slices - 1 else end
        
        cropped = image.crop((left, 0, right, height))
        suffix = image_name[i] if i < len(image_name) else str(i)  # 命名
        save_path = os.path.join(output_folder, f"{i}_{image_time}{suffix}.jpg")
        cropped.save(save_path)
        output_files.append(save_path)

    return output_files    

# 切割圖片
if not os.path.exists(test_image_path):
    raise FileNotFoundError(f"❌ 找不到測試圖片：{test_image_path}")

print("🔄 正在切割圖片...")
split_images = split_image(test_image_path, output_folder)
print(f"✅ 圖片已切割為 {len(split_images)} 片！")

# 依序進行預測
predicted_numbers = []
for i, img_path in enumerate(split_images):
    image = preprocess_image(img_path)
    prediction = model.predict(image)
    predicted_label = np.argmax(prediction)  # 取得最可能的數字
    confidence = np.max(prediction)  # 取得最大信心值

    # 顯示結果
    plt.subplot(1, len(split_images), i + 1)
    plt.imshow(image.reshape(28, 28), cmap="gray")
    plt.title(f"{predicted_label} ({confidence:.2f})")
    plt.axis("off")

    predicted_numbers.append(str(predicted_label))

# 顯示所有切割圖預測結果
plt.show()

# 組合所有預測數字
final_number = "".join(predicted_numbers)
print(f"📊 最終預測結果：{final_number}")
