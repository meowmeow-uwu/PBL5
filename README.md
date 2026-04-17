# Tomato Quality Classification (PyTorch)

Hệ thống nhận diện và phân loại mức độ trưởng thành, chất lượng của cà chua bằng công nghệ **Mạng Neural Tích Chập (Custom CNN)** phát triển từ đầu bằng PyTorch, kèm theo các phương pháp **Đánh giá đặc trưng (Machine Learning Classifier)** như SVM, Random Forest và KNN để đánh giá phụ.

Dự án này đã được chia làm 3 module chính độc lập giúp dễ dàng tinh chỉnh trong các bước khác nhau, với khả năng tự động khôi phục quá trình học (resume training checkpoints).

## Cấu trúc Dự án

```
.
├── train_module.py      # Huấn luyện mô hình từ Dataset cơ bản
├── transfer_module.py   # Kế thừa kết quả (Transfer Learning) cho Dataset_Cachua
├── predict_module.py    # Nhận diện tự động chuẩn đoán bệnh/chất lượng cho từng tấm ảnh
├── model.py             # Định nghĩa cấu trúc Neural Network CustomCNN và Training Loop 
├── preprocessing.py     # Cắt nền ảnh (Background Cancellation) sử dụng OpenCV
├── augmentation.py      # Nhanh bản và lật mở dữ liệu ảnh dùng Torchvision Transforms
├── classifiers.py       # Tập hợp các thuật toán Machine Learning
├── visualization.py     # Xuất các biểu đồ trực quan (như confusion matrix)
└── config.py            # Cấu hình các tham số hyper-parameters
```

## Tính năng

- **Kiến trúc Tự Định Nghĩa (Custom CNN)** gồm *5 Khối Feature* (Conv2D -> BatchNorm -> ReLU -> MaxPool) nối tiếp nhau, rút trích vector đặc trưng thành 512D.
- **Tiền Xử Lý (Background Cancellation)** thông minh: Dùng thuật toán Otsu Threshold kết hợp với Morphological Operations qua nhánh Red/Green nhằm loại bỏ hoàn toàn các phông nền gây nhiễu, tập trung vào quả Cà Chua.
- **Trở lại sau Sự cố (Epoch Resuming)**: `train_module` và `transfer_module` sẽ tự động ghi nhớ trạng thái optimizer, params, giá trị loss từng epoch dưới định dạng `_last.pth`. Nếu chương trình bị sự cố ngắt giữa chừng, bạn chỉ cần gõ lại lệnh, module tự động học tiếp từ điểm ngắt đó thay vì lặp lại từ epoch số 0.
- **Real-time Charting**: Biểu đồ Accuracy/Loss lập tức xuất file `.png` (ở thư mục `results/`) mỗi khi kết thúc 1 epoch. Không cần đợi toàn bộ lịch trình học kết thúc.
- **Xác thực kết quả nâng cao**: Extract 512D Features qua không gian MLearrning: (KNN kèm PCA), RandomForest và SVM Kernels.

## Hướng dẫn Sử dụng (Workflow)

Bạn cần tải xuống toàn bộ dependencies bằng cách sử dụng `uv`:
```bash
uv sync   # Sẽ cài đặt toàn bộ Torch + Torchvision + Scikit-Learn + OpenCV
```

### 1. Training Mặc định (Base Training)

Dùng để huấn luyện CustomCNN trên tập dữ liệu ban đầu. Tham số đường dẫn và Epoch nằm trong `config.py` ở thẻ `DATASET_DIR`.

```bash
python train_module.py
```
- Module sẽ lưu mô hình tốt nhất vào: `results/train_save_model/base_cnn_best.pth`
- Biểu đồ và thông báo sẽ được lưu lại đồng thời trong đường dẫn này.

### 2. Transfer Learning trên Dataset_Cachua

Sau khi bạn đã hoàn thiện Base Training, hệ thống cho phép kế thừa các kinh nghiệm trích xuất vector thông qua Transfer Learning nhằm áp dụng cho biến thể dữ liệu mới hoặc môi trường hình ảnh mới nằm tại `DATASET_CACHUA_DIR`.

```bash
python transfer_module.py
```
- Tự động lấy file `base_cnn` làm móng.
- Thực hiện training cho riêng bài toán mới và cho ra `results/transfer_save_model/transfer_cnn_best.pth`


### 3. Phân Đoán / Suy đổi Trực tiếp (Inference)

Khi bạn muốn đưa một hình ảnh thực tế (ảnh 1 quả trứng cà chua) chụp ngoài màn hình vào để CNN nhận định xem đây là **Reject**, **Ripe**, hay **Unripe**:

```bash
python predict_module.py --image "Đường/dẫn/tới/anhthon.jpg"
```
**Option 2:** (Tuỳ chỉnh trọng số nếu bắt gặp file custom weights)
```bash
python predict_module.py --image "test.jpg" --model "results/train_save_model/base_cnn_last.pth"
```

Quá trình sẽ diễn ra hoàn toàn tương đương với lúc Training do module sẽ gọi thẳng hàm `background_cancellation` cắt nền trước khi đưa vào PyTorch.
Khung xác suất (Probability Logits) cho 3 classes sẽ được hiển thị ngay tại Terminal.  

## Cấu hình (Configurations)
Hiệu chỉnh toàn bộ parameters thông qua `config.py`:
- `IMG_SIZE = 299`: Kích thước Resize
- `FINE_TUNE_EPOCHS = 30`: Tổng lượng Epoch
- `BATCH_SIZE = 32`: Lượng data đẩy vào VRAM GPU cùng 1 thời điểm.
- Thuật toán KNN điều chỉnh số lượng thông số Neighbor, cấu hình Threshold Kernels tương ứng.
