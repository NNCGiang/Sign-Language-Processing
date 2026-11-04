# Giai đoạn 1 — Phân tích kiến trúc, tính khả thi YOLO26 và yêu cầu dữ liệu

## 1. Kiểm tra tính khả thi của YOLO26

| Câu hỏi | Kết luận |
|---|---|
| YOLO26 có tồn tại, dùng được qua Ultralytics không? | Có. `pip install ultralytics`, gọi `YOLO("yolo26n-pose.pt")`. Trọng số tự tải từ GitHub release của Ultralytics. |
| YOLO26 làm được những task nào? | Detect, Segment, Classify, Pose/Keypoints, OBB (+ semantic seg, open-vocab trong YOLOE-26). |
| YOLO26-pose cho ra gì? | 17 điểm khớp COCO / người: mũi, mắt, tai, vai, khuỷu tay, cổ tay, hông, gối, cổ chân. **Không có điểm ngón tay, không có điểm khuôn mặt chi tiết.** |
| YOLO26 có tự hiểu một chuỗi ký hiệu không? | **Không.** YOLO xử lý từng ảnh độc lập. Ý nghĩa của ký hiệu nằm ở *chuyển động theo thời gian* + *hình dạng bàn tay* → bắt buộc có mô hình chuỗi (GRU/LSTM) phía sau. |
| Có fine-tune YOLO26 cho bàn tay được không? | Có. Ultralytics có sẵn bộ dữ liệu `hand-keypoints.yaml` (21 điểm/bàn tay). Fine-tune `yolo26n-pose.pt` trên bộ này → mô hình YOLO26 phát hiện bàn tay + 21 điểm ngón tay (`python train.py yolo-hands`). |

**Hệ quả thiết kế:** YOLO26 đóng vai trò *bộ phát hiện không gian*: tìm người ký hiệu, cắt vùng quan tâm (ROI), cho khung xương thân trên, và (tùy chọn) bàn tay sau khi fine-tune. Hình dạng ngón tay mặc định lấy bằng MediaPipe HandLandmarker chạy **trên ROI do YOLO26 cắt ra** (chính xác hơn khi người đứng xa camera). Biểu cảm khuôn mặt lấy bằng MediaPipe FaceLandmarker (tùy chọn).

Lưu ý môi trường: MediaPipe ≥ 0.10.30 đã bỏ API cũ `mp.solutions.holistic`; dự án dùng **MediaPipe Tasks API** (`HandLandmarker`, `FaceLandmarker`), file mô hình `.task` được tự tải về `models/mediapipe/` lần chạy đầu.

## 2. Kiến trúc pipeline

```
VIDEO (mp4/avi/mov) hoặc WEBCAM
   │  src/preprocessing/video_loader.py   – kiểm tra file, đọc OpenCV, chuẩn hóa FPS & kích thước, loại frame lỗi
   ▼
FRAME (BGR, ≤640px)
   │  src/detection/yolo_detector.py      – YOLO26-pose: bbox người + 17 khớp
   │  src/detection/keypoint_extractor.py – ROI người → MediaPipe Hands (hoặc YOLO26-hand) + Face
   ▼
VECTOR ĐẶC TRƯNG / FRAME (291 chiều)
   │   thân trên 13×(x,y,conf) | 2 tay × [21×(x,y,z) theo thân + 21×(x,y) hình dạng cục bộ] | cờ có tay | 20 điểm mặt
   │   mọi tọa độ chuẩn hóa: gốc = trung điểm 2 vai, đơn vị = độ rộng vai → bất biến vị trí/khoảng cách camera
   ▼
CHUỖI THỜI GIAN  src/recognition/feature_sequence.py – nội suy về T=48 bước, + vận tốc (Δ) → bất biến tốc độ ký hiệu
   ▼
BiGRU/BiLSTM + attention pooling  src/recognition/lstm_model.py
   ▼
PHÂN LOẠI + ĐỘ TIN CẬY  src/recognition/sign_classifier.py – softmax, top-k, ngưỡng confidence
   ▼
GIẢI MÃ LIÊN TỤC  src/realtime/webcam_recognition.py – cửa sổ trượt, xác nhận qua nhiều cửa sổ, chống lặp, phát hiện kết thúc câu
   ▼
DỊCH TIẾNG VIỆT  src/translation/vietnamese_translator.py – từ điển nhãn + luật cụm từ (grammar.yaml), dấu câu, viết hoa
   ▼
TTS TIẾNG VIỆT  src/speech/vietnamese_tts.py – edge-tts (giọng neural vi-VN) → gTTS → VieNeu (offline), có cache & phát lại
```

### Vì sao GRU/LSTM trên keypoints thay vì CNN 3D trên ảnh?
- Dữ liệu ký hiệu tiếng Việt công khai còn ít; keypoints giảm mạnh số chiều nên học được với vài chục video/nhãn.
- Bất biến với nền, quần áo, ánh sáng (đã do bộ phát hiện xử lý).
- Chạy thời gian thực trên CPU.
- Phân biệt được ký hiệu có hình dạng tay giống nhau nhưng khác chuyển động nhờ đặc trưng vận tốc + mô hình chuỗi hai chiều.

### Nhận diện liên tục (câu nhiều ký hiệu)
Mô hình được huấn luyện ở mức **từ/cụm từ (isolated)**. Câu liên tục được giải mã bằng cửa sổ trượt (mặc định 2 s, bước 3 frame):
1. Mỗi cửa sổ → phân loại → (nhãn, độ tin cậy).
2. Chỉ chấp nhận nhãn khi top-1 giống nhau ở `confirm_windows` cửa sổ liên tiếp **và** độ tin cậy ≥ ngưỡng.
3. Không phát lại cùng nhãn cho tới khi có cửa sổ khác nhãn/khoảng nghỉ (`cooldown_sec`).
4. Không thấy tay trong `idle_sec` → kết thúc câu → dịch → đọc **cả câu** (không đọc từng nhãn).
5. Nhãn nền (`_nen`) — video không ký hiệu/tay nghỉ — nên có trong dataset để giảm nhận nhầm.

## 3. Khảo sát dữ liệu ngôn ngữ ký hiệu Việt Nam (VSL)

Không có bộ dữ liệu VSL lớn, chuẩn hóa, công khai tương đương WLASL. Một số nguồn tham khảo:
- **VOYA_VSL** (Hugging Face): ~161 lớp, đã trích keypoints MediaPipe (60×1605), không phải video gốc → không dùng trực tiếp với pipeline YOLO26 này, nhưng tham khảo danh mục từ.
- Nghiên cứu MoViNet-A2 (ĐH Mở TP.HCM): 98 từ/cụm từ, 18 học sinh trường khuyết tật Lâm Đồng, 4.709 video — không công khai sẵn, cần liên hệ tác giả.
- ViSL One-shot (ITMO): sinh dữ liệu từ 4.364 video VSL ba miền.
- Từ điển ngôn ngữ ký hiệu Việt Nam (qipedc.moet.gov.vn) — video mẫu một người/từ, hữu ích làm *tham chiếu khi quay*, không đủ làm dữ liệu huấn luyện.

**Khuyến nghị:** tự thu thập bằng `tools/record_dataset.py` theo giao thức dưới đây, có sự tham gia của người Điếc/thông dịch viên để đảm bảo ký hiệu đúng (VSL có khác biệt vùng miền Bắc–Trung–Nam).

## 4. Yêu cầu dataset

| Hạng mục | Tối thiểu (demo) | Khuyến nghị |
|---|---|---|
| Số nhãn | 5–10 | 30–100 |
| Video / nhãn | 30 | 100+ |
| Số người ký hiệu | 3 | 10+ (khác giới, tuổi, tay thuận) |
| Độ dài clip | 1.5–3 s, bắt đầu & kết thúc ở tư thế nghỉ | như trên |
| Góc quay | chính diện | chính diện ± 30°, 2–3 khoảng cách |
| Ánh sáng/nền | 2 điều kiện | nhiều phòng, nền khác nhau |
| Nhãn nền `_nen` | 30 clip đứng yên / gãi đầu / cầm điện thoại | 100+ |

**Chia dữ liệu theo NGƯỜI**, không chia ngẫu nhiên theo video: nếu cùng một người xuất hiện ở cả train và test, độ chính xác sẽ bị thổi phồng. `tools/split_dataset.py --by-signer` thực hiện việc này dựa trên tên file `nhan__nguoi__so.mp4`.

**Quy ước tên thư mục nhãn:** không dấu, `snake_case` (`xin_chao`, `cam_on`). Văn bản tiếng Việt có dấu khai báo trong `configs/labels.yaml`.

## 5. Kế hoạch triển khai

| Giai đoạn | Nội dung | File |
|---|---|---|
| 1 | Phân tích, kiến trúc, yêu cầu dữ liệu | `docs/ARCHITECTURE.md` |
| 2 | Đọc video, chuẩn hóa, trích keypoints, cache `.npy` | `src/preprocessing/*`, `src/detection/*`, `train.py extract` |
| 3 | YOLO26 (pretrained + fine-tune tay), mô hình BiGRU/LSTM | `src/detection/yolo_detector.py`, `src/recognition/*`, `train.py yolo-hands` |
| 4 | Huấn luyện, early stopping, checkpoint, đánh giá trên test độc lập | `train.py train`, `src/evaluation/evaluate.py` |
| 5 | Dịch tiếng Việt, TTS | `src/translation/*`, `src/speech/*` |
| 6 | Webcam thời gian thực, giao diện PyQt5 | `src/realtime/*`, `app/main.py` |
| 7 | Kiểm thử, README | `tools/smoke_test.py`, `README.md` |

## 6. Giới hạn đã biết (trung thực)
- Dịch gloss → câu là **luật + từ điển**, không phải mô hình dịch máy; ngữ pháp VSL (thứ tự từ, phủ định cuối câu…) chỉ được xử lý ở các mẫu đã khai báo trong `grammar.yaml`. Dịch tự do đòi hỏi corpus song song gloss–câu, hiện chưa có công khai.
- Ranh giới ký hiệu trong câu liên tục được ước lượng bằng cửa sổ trượt; độ chính xác thấp hơn nhận diện từ đơn.
- Mọi chỉ số (Accuracy, FPS…) chỉ có ý nghĩa sau khi huấn luyện trên dữ liệu thật; dữ liệu mẫu tổng hợp trong `tools/make_sample_dataset.py` chỉ dùng để kiểm tra pipeline chạy đúng.
