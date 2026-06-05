import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
# Cấu hình ban đầu
TRAIN_DIR = "/mnt/c/Users/ADMIN/Desktop/DATN_CNN/dataset_training/train" #Load ảnh tập train
VAL_DIR   = "/mnt/c/Users/ADMIN/Desktop/DATN_CNN/dataset_training/val"#load ảnh tập validation
IMG_H=165 # chiều cao ảnh cần đồng nhất cho input, tính bằng pixels
IMG_W=220 #chiều rộng cần đồng nhất cho tập input
BATCH=32 # Số ảnh mỗi lần cập nhật trong số Weight => ảnh hướng trực tiếp đến tốc độ và chất lượng train
EPOCHS=50 # Số lần duyêt toàn bộ dataset (1 epoch- duyệt toàn bộ ảnh trong tập train)
NUM_CLASSES=15  # số class output cần phân loại - dùng ở layer dense ở block cuối cùng


# Load Data
train_ds=tf.keras.utils.image_dataset_from_directory( #tự động đọc ảnh từ folder trong đường dẫn, gán nhẵn theo tên của folder con
    TRAIN_DIR, # dựa
    image_size=(IMG_H, IMG_W), #reziise về kích cỡ đã set
    batch_size=BATCH,
    label_mode ="categorical", #Nhãn dạng one hot vector
    seed=42,
)

val_ds=tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=(IMG_H, IMG_W),
    batch_size=BATCH,
    label_mode="categorical",
    seed=42,
)
class_names=train_ds.class_names #thuộc tính của dataset, tự động lấy tên từ folder con
#Normalize [0,255] -> [0,1]
normalize=layers.Rescaling(1.0/255) #Layer nhân toàn bộ giá trị pixel với hệ số(pixel 0->255 thành 0->1)
train_ds  = train_ds.map(lambda x, y: (normalize(x), y)).cache().prefetch(tf.data.AUTOTUNE)# map-dùng để áp dụng lên từng phần tử của dataset, - lambda x,y - hàm ẩn danh nhận 2 tham số x(ảnh), y(nhãn), normalize ảnh x, giữ nguyên nhãn y
val_ds    = val_ds.map(  lambda x, y: (normalize(x), y)).cache().prefetch(tf.data.AUTOTUNE) # cache(), sau lần đọc đầu tiên, lưu toàn bộ dataset vào RAM, các epoch sa đọc từ RAM thay vì ổ cứng -> tốc độ nhanh hơn
#prefetch - tải sẵn batch tiếp theo trong khi GPU đang train batch hiện tại. 
#tf.data.AUOTUNE - để TensorFlow tự quyết định số batch cần tải sẵn-> tối ưu

# Build CNN model
# cấu trúc model: Input → [Conv32 → Pool] → [Conv64 → Pool] → [Conv128 → Pool]→ Flatten → Dense(256) → Softmax(N)
# Thông tin: 
# Kernel 3x3, stride = 1, padding valid, no đilation 
# Activation : ReLU - Tất cả các block layers
#Pooling layer : Max Pooling 2x2

model=models.Sequential(name="UAV_CNN") # Sequential - kiểu kiến trúc model tuần tự, layer xếp chồng, output layer này là input layer kia.
model.add(tf.keras.Input(shape=(IMG_H, IMG_W, 3)))
#add- thêm layer vào model
#tf.keras.Input - Khai báo shape đầu vào -
#shape (165x220x3) - tensor 3 chiều -tensor=3

#BLOCK 1- 32 kernels
model.add(layers.Conv2D(32, (3,3),strides=(1,1), padding="valid", activation="relu", dilation_rate=1 ))
model.add(layers.MaxPool2D((2,2)))
#layers.Conv2D- Lớp tích chập 2D, 32 kernel-convolution filer, 3x3 - kích thước kernel, không có padding, hàm kích hoạt là relu, strides=1 -kernel dịch chuyển 1 pixel theo chiều ngang, 1 pixel theo chiều dọc
# MaxPooling - Lớp pooling lấy giá trị lớn nhất trong vùng 2x2

#Block 2- 64 kernels
model.add(layers.Conv2D(64, (3,3), strides=(1,1), padding="valid", activation="relu", dilation_rate=1))
model.add(layers.MaxPool2D((2,2)))

#Block3-128 Kernels
model.add(layers.Conv2D(128, (3,3), strides=(1,1), padding="valid", activation="relu", dilation_rate=1))
model.add(layers.MaxPool2D((2,2)))

# Head
model.add(layers.Flatten())#Flatten- chuyển tensor nhiều chiều -> vector 1 chiều-> chỉ reshape
model.add(layers.Dense(256, activation="relu")) # Fully connected layer, mọi nơ ron đề kết nối với mọi nơ ron của layer trước , 256 số noron của layer
model.add(layers.Dropout(0.5)) #Layer dropout- tắt ngẫu nhiên 50% số noron mỗi batch trong lúc train
model.add(layers.Dense(NUM_CLASSES, activation="softmax"))# layer dense cho ra 17 noron output, mỗi noron=1 class,
#softmax chuyển 15 số thành sác xuất, tổng sx 15 class=1

model.summary()


# ComPile
# Loss Function : categorical cross-entropy
# Optimizer: Adam và batch = 32
model.compile( # compile-cấu hình quá trình train
    optimizer=tf.keras.optimizers.Adam(),#optimizez lựa chọn thuật toán cấp nhật weight
    loss="categorical_crossentropy", # hàm loss là cate_cross - dùng cho nhiều class với nhãn one-hot
    metrics=["accuracy"], #metric-chỉ số eo dõi lúc train=accuracy - không ảnh hưởng trọng số
)




#Train model
cb=[
    callbacks.ModelCheckpoint("model_Adam_32.keras", monitor="val_accuracy",save_best_only=True, verbose=1),#Lưu tên model là best_model.keras
    #theo dõi monitor="val_accuracy"- theo dõi chỉ số accuracy trên tập val để quyết định lưu
    #save_best_only=True: chỉ lưu lại khi val_accuracy cao hơi lần trước
    #verbose=1- in thông báo khi lưu: VD: Epoch 5:val_accuracy improved.....
    callbacks.EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True, verbose=1),
    #patience=15- chờ 15 epoch liên tục mà chỉ số theo dõi val_accuracy không cải thiện mới dừng
    #restore_best_weights=True- Sau khi dừng, tự động load lại weight tốt nhất đã lưu
]
history=model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cb)
#fit- bắt đầu train model
# tham số: train_ds, dataset_val- dùng để đánh giá sau mỗi epoch, EPOCHS tối đa,cb-list chứa ModelCheckpoint và EarlyStopping
# History- object lưu toàn bộ log train:accuracy, loss từng epoch

#EVALUATE
y_prob = model.predict(val_ds, verbose=1)     # predict: chỉ chạy forward pass, không cập nhật weight                       
y_pred = np.argmax(y_prob, axis=1)                                   
y_true = np.concatenate([np.argmax(y, axis=1) for _, y in val_ds])

uncertainty = 1 - np.max(y_prob, axis=1)                            
print("\n── Per-class Report ──────────────────────────")
for i, name in enumerate(class_names):
    tp = np.sum((y_true==i) & (y_pred==i))
    fp = np.sum((y_true!=i) & (y_pred==i))
    fn = np.sum((y_true==i) & (y_pred!=i))
    p  = tp / (tp + fp + 1e-9)
    r  = tp / (tp + fn + 1e-9)
    f1 = 2*p*r / (p + r + 1e-9)
    print(f"  {name:<20}  precision={p:.2f}  recall={r:.2f}  f1={f1:.2f}")

print(f"\n  Overall accuracy : {np.mean(y_true==y_pred)*100:.2f}%")
print(f"  Mean uncertainty : {uncertainty.mean():.4f}")

# Confusion matrix 
cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
for t, p in zip(y_true, y_pred):
    cm[t][p] += 1

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history["accuracy"],     label="Train")
axes[0].plot(history.history["val_accuracy"], label="Val")
axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].grid(True)
axes[1].plot(history.history["loss"],         label="Train")
axes[1].plot(history.history["val_loss"],     label="Val")
axes[1].set_title("Loss"); axes[1].legend(); axes[1].grid(True)
plt.tight_layout(); plt.savefig("history_Adam_32.png", dpi=150); plt.show()

fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(cm, cmap="Blues")
plt.colorbar(im, ax=ax)
ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(class_names, rotation=90)
ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(class_names)
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        ax.text(j, i, cm[i,j], ha="center", va="center",
                color="white" if cm[i,j] > cm.max()/2 else "black")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
plt.tight_layout(); plt.savefig("confusion_matrix_Adam_32.png", dpi=150); plt.show()

print("\nDone! Saved → best_model.keras / history_optimizer.png / confusion_matrix_optimizer.png")