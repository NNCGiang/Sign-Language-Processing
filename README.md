# Vietnamese Sign Language Recognition & Translation System

Hệ thống nhận diện **Ngôn ngữ Ký hiệu Việt Nam (VSL)** từ video/webcam → **văn bản tiếng Việt** → **giọng nói tiếng Việt**.

```
Video/Webcam → OpenCV → YOLO26-pose (người + khung xương) → MediaPipe Hands/Face trên ROI (hoặc YOLO26-hand)
            → vector 291 chiều/frame → chuỗi thời gian → BiGRU/BiLSTM + attention → ký hiệu + độ tin cậy
            → giải mã liên tục → luật dịch tiếng Việt → TTS (edge-tts / gTTS / VieNeu offline)
```

Phân tích kiến trúc, tính khả thi YOLO26 và yêu cầu dataset: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

> **Điều quan trọng:** YOLO26 không "hiểu" ký hiệu. Nó chỉ phát hiện người và khớp cơ thể trên từng frame
> (YOLO26-pose COCO có 17 điểm, **không có ngón tay**). Ý nghĩa ký hiệu được học bởi mô hình chuỗi GRU/LSTM
> từ chuyển động của thân, bàn tay (21 điểm/tay) và khuôn mặt.

---

## 1. Cài đặt (Windows 10/11)

Yêu cầu: **Python 3.10–3.12** (64-bit), webcam, (tùy chọn) GPU NVIDIA.

```powershell
cd VietnameseSignLanguage
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip

# PyTorch — chọn MỘT dòng:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu      # chỉ CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124    # GPU NVIDIA (CUDA 12.4)

pip install -r requirements.txt
```

Kiểm tra GPU: `python -c "import torch; print(torch.cuda.is_available())"`.
Linux/macOS: thay `.venv\Scripts\activate` bằng `source .venv/bin/activate`.

Lần chạy đầu tiên sẽ tự tải:
- `models/yolo26/yolo26n-pose.pt` (Ultralytics, GitHub)
- `models/mediapipe/hand_landmarker.task`, `face_landmarker.task` (Google)

Nếu mạng chặn, tải thủ công theo URL trong `configs/config.yaml` và đặt đúng thư mục trên.

## 2. Kiểm tra nhanh toàn bộ pipeline (chưa cần dữ liệu thật)

```powershell
python tools/smoke_test.py --quick
```

Lệnh này chạy unit test bộ dịch + bộ giải mã câu, tạo **dữ liệu mẫu tổng hợp** trong `dataset_sample/`,
rồi chạy check → extract (YOLO26) → train → evaluate → predict → TTS. Kết quả ở `runs/smoke/`.

⚠ Dữ liệu mẫu là ảnh người được xoay/lắc/co dãn — **không phải ngôn ngữ ký hiệu**. Độ chính xác trên bộ này
chỉ cho biết pipeline chạy đúng, không nói gì về khả năng nhận diện ký hiệu thật.

## 3. Thu thập dữ liệu thật

Chưa có bộ dữ liệu VSL dạng video công khai, đủ lớn để dùng ngay (xem khảo sát trong docs/ARCHITECTURE.md).
Cách tự thu thập:

1. **Chọn danh sách từ** (bắt đầu 5–10 từ) và thống nhất cách ký với người Điếc/thông dịch viên.
   Tham khảo từ điển ký hiệu của Bộ GD&ĐT: https://qipedc.moet.gov.vn
2. **Quay** bằng công cụ có sẵn. Mỗi người, mỗi từ ≥ 30 clip, mỗi clip ~2,5 s,
   **bắt đầu và kết thúc ở tư thế nghỉ** (tay buông):
   ```powershell
   python tools/record_dataset.py --label xin_chao --signer nguoi01 --count 30
   python tools/record_dataset.py --label _nen     --signer nguoi01 --count 30   # nhãn nền: đứng yên, gãi đầu, chỉnh tóc…
   ```
   Đổi ánh sáng, nền, khoảng cách (1–2,5 m), góc lệch nhẹ giữa các lượt quay. Càng nhiều người càng tốt (≥ 3, khuyến nghị 10+).
3. **Chia theo người** (người trong tập test không xuất hiện trong train):
   ```powershell
   python tools/split_dataset.py dataset_raw dataset --by-signer
   ```
4. **Khai báo nghĩa tiếng Việt** cho nhãn mới trong `configs/labels.yaml`, và (nếu cần) mẫu câu trong `configs/grammar.yaml`.

Có thể dùng video sẵn có: đặt theo cấu trúc dưới đây, tên thư mục nhãn **không dấu, snake_case**:

```
dataset/
├── train/xin_chao/xin_chao__nguoi01__001.mp4 ...
├── val/xin_chao/...
└── test/xin_chao/...
```

Đặt tên file dạng `<nhãn>__<người>__<số>.mp4` để hệ thống phát hiện được người ký hiệu bị trùng giữa train và test.

## 4. Huấn luyện & đánh giá

```powershell
python train.py check      # kiểm tra định dạng, số lượng, cảnh báo trùng người train/test
python train.py extract    # YOLO26 + MediaPipe → runs/features/*.npy (có cache, chạy lại chỉ xử lý video mới)
python train.py train      # BiGRU → models/checkpoints/best_model.pt (early stopping theo val loss)
python -m src.evaluation.evaluate    # đánh giá trên TẬP TEST độc lập
# hoặc tất cả trong một lệnh:
python train.py all
```

Kết quả đánh giá trong `runs/evaluation/`:

| File | Nội dung |
|---|---|
| `metrics.json` | Accuracy, Precision, Recall, F1 (macro/weighted), Loss, độ phủ & độ chính xác khi áp ngưỡng, ms/frame, FPS |
| `classification_report.txt` | chỉ số theo từng nhãn |
| `confusion_matrix.png/.csv` | ma trận nhầm lẫn |
| `loss_curve.png`, `accuracy_curve.png` | training/validation loss & accuracy |
| `latency.png` | thời gian trích đặc trưng và suy luận |
| `misclassified.csv` | các mẫu sai để xem lại |

**WER/CER cho câu liên tục:** quay vài video nói cả câu, tạo file CSV:
```csv
video,reference
D:/cau/cau01.mp4,Xin chào! Tôi khỏe.
```
rồi chạy `python -m src.evaluation.evaluate --sentences cau.csv`.

Đổi sang LSTM: `model.rnn_type: lstm` trong `configs/config.yaml`.

### (Tùy chọn) Fine-tune YOLO26 cho bàn tay

```powershell
python train.py yolo-hands --epochs 100 --batch 16
```

Fine-tune `yolo26n-pose.pt` trên bộ **Hand Keypoints** của Ultralytics (21 điểm/tay, tự tải ~370 MB; nên dùng GPU).
Kết quả lưu ở `models/yolo26/yolo26n-hand-pose.pt`. Để dùng: đặt `hands.backend: yolo` trong config, rồi **chạy lại
extract + train** (đặc trưng thay đổi thì mô hình phải học lại; hệ thống sẽ báo lỗi nếu cấu hình không khớp checkpoint).
Có thể thay `--data` bằng YAML bộ dữ liệu bàn tay tự gán nhãn (định dạng YOLO-pose, `kpt_shape: [21, 3]`);
`python -m src.preprocessing.frame_extractor <video|thư_mục> frames_out --fps 5` giúp xuất frame để gán nhãn.

## 5. Sử dụng

**Giao diện** (tải video, webcam, phát âm, phát lại, xóa):
```powershell
python -m app.main
```

**Dòng lệnh — video có sẵn:**
```powershell
python predict.py video.mp4 --speak
python predict.py thu_muc_video/ --json runs/ket_qua.json
```

**Dòng lệnh — webcam:**
```powershell
python -m src.realtime.webcam_recognition --camera 0      # q: thoát, c: xóa, s: đọc lại
```

Ví dụ kết quả:
```
Ký hiệu nhận diện: [XIN CHÀO] [TÔI] [KHỎE]
Văn bản dịch     : Xin chào! Tôi khỏe.
```

### Nhận diện thời gian thực hoạt động thế nào
- Buffer 2 giây đặc trưng gần nhất, phân loại lại mỗi 3 frame.
- Chỉ chấp nhận ký hiệu khi **3 cửa sổ liên tiếp** cùng kết quả và độ tin cậy ≥ **0,70**.
- Không nhận lặp lại cùng ký hiệu nếu chưa có khoảng ngắt.
- **1,2 giây không thấy tay** → kết thúc câu → dịch → đọc cả câu (không đọc từng từ).
- Tất cả tham số nằm ở mục `inference:` trong `configs/config.yaml`. CPU chậm: đặt `yolo.detect_interval: 2`
  (chạy YOLO26 mỗi 2 frame) hoặc `face.enabled: false` (khi đổi `face.enabled` phải extract + train lại).

## 6. Text-to-Speech tiếng Việt

Thứ tự engine trong `tts.engine_order`; engine lỗi tự động được bỏ qua:
- `edge` — giọng neural `vi-VN-HoaiMyNeural` (nữ) / `vi-VN-NamMinhNeural` (nam), cần Internet.
- `gtts` — Google, cần Internet.
- `vieneu` — chạy **offline** trên CPU: `pip install vieneu`.

Âm thanh được cache trong `runs/audio/` (câu đã đọc sẽ không tổng hợp lại).

## 7. Cấu trúc mã nguồn

```
src/preprocessing/video_loader.py     đọc & kiểm tra video, chuẩn hóa FPS/kích thước, loại frame lỗi, quét/kiểm tra dataset
src/preprocessing/frame_extractor.py  trích frame (bộ nhớ hoặc ảnh JPG)
src/detection/yolo_detector.py        YOLO26-pose (người), YOLO26-hand, hàm fine-tune
src/detection/keypoint_extractor.py   ROI + MediaPipe Hands/Face → vector 291 chiều đã chuẩn hóa, vẽ overlay
src/recognition/feature_sequence.py   cache đặc trưng, nội suy thời gian, vận tốc, augmentation, Dataset
src/recognition/lstm_model.py         BiGRU/BiLSTM + attention pooling
src/recognition/sign_classifier.py    nạp checkpoint, kiểm tra tương thích, dự đoán + ngưỡng
src/translation/vietnamese_translator.py  gloss → câu tiếng Việt (từ điển + luật)
src/speech/vietnamese_tts.py          TTS nhiều engine, cache, phát/phát lại
src/realtime/webcam_recognition.py    bộ giải mã liên tục, RealtimeRecognizer, nhận diện video/webcam
src/evaluation/evaluate.py            chỉ số, biểu đồ, WER/CER
app/main.py                           giao diện PyQt5
train.py / predict.py                 điểm vào dòng lệnh
tools/                                quay dữ liệu, chia dữ liệu, dữ liệu mẫu, smoke test
```

## 8. Giới hạn

- Độ chính xác phụ thuộc hoàn toàn vào dữ liệu: số người, độ đa dạng, độ chuẩn của ký hiệu. Không có con số
  nào được hứa trước — hãy đọc `runs/evaluation/metrics.json` sau khi huấn luyện trên dữ liệu của bạn.
- Dịch là **luật + từ điển**, chỉ xử lý thứ tự từ VSL ở các mẫu đã khai báo trong `grammar.yaml`.
  Dịch tự do cần corpus song song gloss–câu.
- Ký hiệu dùng một tay thuận: nếu người thuận tay trái, hãy có họ trong dữ liệu huấn luyện.
- Webcam: hệ thống xử lý ảnh **không lật** (chỉ lật khi hiển thị) để tay trái/phải khớp với dữ liệu quay bằng `record_dataset.py`.

## 9. Xử lý sự cố

| Lỗi | Cách xử lý |
|---|---|
| `Không tải được ...hand_landmarker.task` | tải thủ công URL trong config, đặt vào `models/mediapipe/` |
| `Cấu hình đặc trưng hiện tại ... khác lúc huấn luyện` | đã đổi `hands.backend`/`face.enabled` → chạy lại `train.py extract --overwrite` và `train.py train` |
| Webcam không mở trên Windows | thử `--camera 1`; đóng ứng dụng khác đang dùng camera |
| Tiếng Việt lỗi font trong console | chạy `chcp 65001` trước khi chạy lệnh |
| Không có âm thanh | kiểm tra Internet (edge/gtts) hoặc `pip install vieneu`; xem log trong `runs/logs/` |
| `hand_ratio` thấp khi extract | tay ra khỏi khung/thiếu sáng — quay lại video đó |
