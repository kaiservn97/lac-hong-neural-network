# Danh Sách Đề Tài Mini-Project

Dưới đây là danh sách đề tài nhỏ, có tính ứng dụng rõ ràng trong bối cảnh Việt Nam, đủ để sinh viên làm báo cáo hoặc mini-project từ nay đến cuối tháng 9 (khoảng 4-6 tuần).

## 1. Bất động sản và tài chính cá nhân

### 1.1 Phân loại mức giá nhà tại Hà Nội hoặc HCMC (Low/Medium/High)

- Ý tưởng: Dùng ANN đơn giản (MLP) dự đoán mức giá nhà từ diện tích, số phòng, số tầng, quận/huyện, sau đó phân loại thành 3 mức giá.
- Dataset:
  - [Vietnam Housing Dataset (Hanoi)](https://www.kaggle.com/datasets/ladcva/vietnam-housing-dataset-hanoi)
  - [Vietnam Housing Dataset (Hanoi) - Data](https://www.kaggle.com/datasets/ladcva/vietnam-housing-dataset-hanoi/data)
  - [House Pricing HCMC](https://www.kaggle.com/datasets/trnduythanhkhttt/housepricinghcm)
  - [Notebook tham khảo Kaggle](https://www.kaggle.com/ladcva/vietnam-housing-dataset-hanoi/code)
- Mức độ: Nhóm 2-3 sinh viên, làm EDA + 1-2 mô hình (baseline tuyến tính + MLP).

### 1.2 Dự đoán giá/m² bất động sản và so sánh quận/huyện

- Ý tưởng: Hồi quy dự đoán `price_per_m2` và vẽ heatmap giá theo quận/huyện.
- Dataset:
  - [Vietnamese Real Estate Listings May 2024](https://www.kaggle.com/datasets/qmanhbeo/vietnamese-real-estate-listings-may-2024)
- Mức độ: Có thể dùng subset khoảng 10-20k mẫu, MLP nhỏ hoặc XGBoost + so sánh với ANN.

### 1.3 Ứng dụng mô hình định giá nhà đơn giản cho người mua nhà lần đầu

- Ý tưởng: Xây 1 model ANN nhỏ + form nhập thông tin (Streamlit hoặc Gradio) để user nhập diện tích, số phòng, khu vực và nhận gợi ý khoảng giá.
- Dataset:
  - [oceanNG/vietnam-real-estates](https://huggingface.co/datasets/oceanNG/vietnam-real-estates/tree/main)
  - [coindex/vietnam-real-estates](https://huggingface.co/datasets/coindex/vietnam-real-estates)
  - [Vietnamese Real Estate Listings May 2024](https://www.kaggle.com/datasets/qmanhbeo/vietnamese-real-estate-listings-may-2024)
- Mức độ: Phần DL vẫn là MLP đơn giản, phần UI nhẹ.

## 2. Cảm xúc và sentiment tiếng Việt trên mạng xã hội

### 2.1 Phân loại sentiment 3 lớp (Positive/Negative/Neutral) cho comment sinh viên

- Ý tưởng: Dùng LSTM hoặc MLP trên embedding, hoặc fine-tune model pre-trained đơn giản để phân loại cảm xúc các post Facebook về đời sống sinh viên.
- Dataset:
  - [Vietnamese Sentiment Analysis on Social Media Based on BERT Architecture](https://hjs.huflit.edu.vn/index.php/hjs/article/download/335/206)
- Mức độ: Nhóm 2-3 sinh viên; có thể làm baseline SVM hoặc Logistic + LSTM/ANN.

### 2.2 Nhận diện cảm xúc chi tiết từ comment mạng xã hội tiếng Việt

- Ý tưởng: Multi-class emotion classification (vui, buồn, giận, lo lắng,...).
- Dataset:
  - [ViGoEmotions - ACL Anthology PDF](https://aclanthology.org/2026.eacl-long.129.pdf)
- Mức độ: Chọn 5-7 emotion phổ biến nhất, dùng MLP hoặc BiLSTM đơn giản.

### 2.3 Target-oriented emotion: Cảm xúc theo từng đối tượng trong câu

- Ý tưởng: Ví dụ comment "Mạng Viettel nhanh nhưng app ngân hàng lag" -> cảm xúc khác nhau cho từng target.
- Dataset:
  - [ViTOED dataset](https://arxiv.org/html/2608.12776v1)
- Mức độ: Có thể đơn giản hóa: chỉ focus polarity (tích cực/tiêu cực) cho target chính.

### 2.4 Phân tích sentiment bình luận sản phẩm hoặc dịch vụ Việt Nam

- Ý tưởng: Thu thập thêm review từ Tiki/Shopee/Facebook (hoặc lấy từ các dataset tổng hợp trong các paper về PhoBERT và sentiment), xây model phân loại sentiment, ứng dụng để thống kê mức hài lòng theo brand hoặc ngành.
- Dataset hoặc tham khảo:
  - [TNU Journal paper PDF](https://jst.tnu.edu.vn/jst/article/download/12889/pdf)
  - [Vietnamese Sentiment Analysis Survey PDF](https://arxiv.org/pdf/2210.02063.pdf)
- Mức độ: Vừa; có thể kết hợp BERT-base-Vi hoặc ANN trên feature truyền thống.

## 3. Giao thông và thị giác máy tính (CV)

### 3.1 Nhận dạng biển báo giao thông Việt Nam (Classification hoặc Detection)

- Ý tưởng: Huấn luyện CNN (ResNet nhỏ, MobileNet hoặc CNN tự xây) để phân loại biển báo từ ảnh.
- Dataset:
  - [Vietnam Traffic Signs (VNTS)](https://www.kaggle.com/datasets/maitam/vietnamese-traffic-signs/data)
- Mức độ: Nhóm 3-4 sinh viên, đủ để build 1 model classification + demo inference trên ảnh/video ngắn.

### 3.2 Nhận diện loại phương tiện giao thông trên đường

- Ý tưởng: Dùng CNN hoặc YOLO nhỏ để phân loại/đếm xe qua camera.
- Dataset:
  - [UIT-VinaDeveS22 mô tả dataset](https://hackmd.io/@gXJvSTQdQ7CFubNivs8MqQ/BJPfvEZ4c)
- Mức độ: Có thể bắt đầu từ object detection pre-trained (YOLOv5/v8) và fine-tune trên subset.

### 3.3 Hệ thống cảnh báo tốc độ hoặc biển cấm từ camera hành trình

- Ý tưởng: Kết hợp nhận dạng biển báo (VNTS) + logic đơn giản để cảnh báo khi thấy biển giới hạn tốc độ hoặc biển cấm.
- Dataset: VNTS + video/ảnh bổ sung.
- Mức độ: Phần DL gần giống đề 3.1, thêm logic và visualization.

## 4. Giáo dục và dữ liệu sinh viên

### 4.1 Phân tích cảm xúc sinh viên về môn học, giảng viên, cơ sở vật chất

- Ý tưởng: Thu thập comment từ fanpage/confession của trường, phân loại positive/negative/neutral, làm dashboard cho Khoa.
- Dataset hoặc tham khảo:
  - [CEUR Workshop paper](https://ceur-ws.org/Vol-3876/paper2.pdf)
  - [HJS HUFLIT paper](https://hjs.huflit.edu.vn/index.php/hjs/article/download/335/206)
- Mức độ: Nhóm 2-3 sinh viên; dùng BERT-based hoặc ANN + BOW/TF-IDF.

### 4.2 Hệ thống gợi ý cải thiện môn học dựa trên sentiment

- Ý tưởng: Từ dữ liệu cảm xúc sinh viên (Confession/Google Form), nhóm ra top 3 vấn đề (lịch thi, bài tập, giảng dạy) và đề xuất cải tiến.
- Dataset hoặc tham khảo:
  - [HJS HUFLIT paper](https://hjs.huflit.edu.vn/index.php/hjs/article/download/335/206)
- Mức độ: Mô hình DL có thể đơn giản, trọng tâm là phân tích và visualization.

## 5. Ứng dụng khác

### 5.1 Dự đoán giá thuê phòng trọ hoặc căn hộ tại HCMC/Hà Nội

- Ý tưởng: Dùng dataset real estate nhưng chỉ filter listing thuê, xây model regression giá thuê/tháng hoặc giá/m².
- Dataset:
  - [Real Estate in Vietnam](https://www.kaggle.com/datasets/andyvo1009/real-estate-in-vietnam/data)
  - [Vietnamese Real Estate Listings May 2024](https://www.kaggle.com/datasets/qmanhbeo/vietnamese-real-estate-listings-may-2024)
- Ứng dụng: Giúp sinh viên thấy giá thuê hợp lý theo khu vực và diện tích.

### 5.2 Phân loại mức độ lịch sự hoặc tiêu cực trong comment mạng xã hội tiếng Việt

- Ý tưởng: Dùng dataset như ViHSD, UIT-VSMEC hoặc dữ liệu VLSP về toxic/hate speech tiếng Việt để phân loại bình luận thành sạch/hate/offensive.
- Dataset hoặc tham khảo:
  - [Survey paper PDF](https://arxiv.org/pdf/2210.02063.pdf)
  - [CEUR Workshop paper](https://ceur-ws.org/Vol-3876/paper2.pdf)
- Ứng dụng: Bộ lọc comment cho page trường hoặc fanpage doanh nghiệp.

### 5.3 Dashboard phân tích thị trường bất động sản Việt Nam dùng DL + trực quan hóa

- Ý tưởng: Dùng một phần nhỏ từ dataset lớn về bất động sản Việt Nam để:
  - Train model ANN đơn giản dự đoán giá.
  - Vẽ bản đồ/heatmap giá trung bình theo tỉnh/quận, biểu đồ xu hướng theo tháng.
- Dataset hoặc tham khảo:
  - [coindex/vietnam-real-estates](https://huggingface.co/datasets/coindex/vietnam-real-estates)
  - [HF Daily Briefer nhắc tới dataset](https://www.hfdailybriefer.com/post/1882)
  - [oceanNG/vietnam-real-estates](https://huggingface.co/datasets/oceanNG/vietnam-real-estates/tree/main)
- Mức độ: Nên downsample (ví dụ 50k-100k record) để phù hợp thời gian.

## Gợi ý phân bổ và chọn đề

- Thời gian đến cuối tháng 9:
  - Với nhóm 2-3 sinh viên, nên chọn đề có dataset sẵn, bài toán rõ (regression hoặc classification), pipeline đầy đủ: EDA -> baseline truyền thống -> 1-2 mô hình ANN/MLP/CNN đơn giản -> so sánh và viết báo cáo.

- Nếu lớp đông, có thể chia theo hướng:
  - Nhóm bất động sản: 1.1, 1.2, 1.3, 5.1, 5.3.
  - Nhóm NLP: 2.1, 2.2, 2.3, 2.4, 4.1, 4.2, 5.2.
  - Nhóm CV: 3.1, 3.2, 3.3.

Nếu bạn cho mình biết dự kiến mỗi nhóm bao nhiêu sinh viên, có GPU hay không, và muốn ưu tiên NLP/CV/tabular, mình có thể shortlist lại 5-7 đề ready-to-use kèm khung báo cáo + skeleton notebook cho từng loại.
