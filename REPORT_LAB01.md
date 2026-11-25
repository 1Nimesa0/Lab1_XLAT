# BÁO CÁO THỰC HÀNH CHI TIẾT - LAB 1
## PHÂN TÍCH VÀ XỬ LÝ TÍN HIỆU ÂM THANH SỐ
### HỌC PHẦN: XỬ LÝ ÂM THANH VÀ TIẾNG NÓI (CSE457)
**Khoa Công nghệ Thông tin – Bộ môn Trí tuệ Nhân tạo – Trường Đại học Thủy lợi**

---

* **Sinh viên thực hiện:** Trần Hoài Nam
* **Mã sinh viên:** 2351260676
* **Học phần:** CSE457 – Xử lý Âm thanh và Tiếng nói
* **Nội dung:** Từ tín hiệu rời rạc đến FFT/STFT, lọc số, lượng tử hóa và mã hóa âm thanh
* **Mã nguồn chính:** `Lab1_main.ipynb`

---

## MỤC LỤC
1. [Mục tiêu và Tổng quan Pipeline thực hành](#1-mục-tiêu-và-tổng-quan-pipeline-thực-hành)
2. [Phần A: Đọc và kiểm tra dữ liệu âm thanh](#2-phần-a-đọc-và-kiểm-tra-dữ-liệu-âm-thanh)
3. [Phần B: Phân tích miền thời gian](#3-phần-b-phân-tích-miền-thời-gian)
4. [Phần C: Phân tích miền tần số bằng FFT](#4-phần-c-phân-tích-miền-tần-số-bằng-fft)
5. [Phần D: STFT và Spectrogram](#5-phần-d-stft-và-spectrogram)
6. [Phần E: Thí nghiệm cửa sổ (Rectangular vs Hamming)](#6-phần-e-thí-nghiệm-cửa-sổ-rectangular-vs-hamming)
7. [Phần F: Thiết kế và áp dụng Bộ lọc số FIR](#7-phần-f-thiết-kế-và-áp-dụng-bộ-lọc-số-fir)
8. [Phần G: Lượng tử hóa, Resampling và Mã hóa](#8-phần-g-lượng-tử-hóa-resampling-và-mã-hóa)
9. [Tổng hợp Ma trận kết quả tối thiểu và Thí nghiệm độc lập](#9-tổng-hợp-ma-trận-kết-quả-tối-thiểu-và-thí-nghiệm-độc-lập)
10. [Giải đáp chi tiết 7 Câu hỏi báo cáo (Mục 6)](#10-giải-đáp-chi-tiết-7-câu-hỏi-báo-cáo-mục-6)
11. [Kết luận và Tài liệu tham khảo](#11-kết-luận-và-tài-liệu-tham-khảo)

---

## 1. MỤC TIÊU VÀ TỔNG QUAN PIPELINE THỰC HÀNH

Bài thực hành Lab 1 xây dựng nền tảng xử lý tín hiệu âm thanh số (DSP) trên tệp tiếng nói thực tế (`input/input.mp3`) theo chuỗi năng lực cốt lõi của Chương 1 học phần CSE457:

```text
Audio (MP3) → Biểu diễn số chuẩn hóa [-1, 1] → Phân tích miền thời gian (Waveform/RMS/Energy)
            → Phân tích miền tần số (FFT) → Phân tích thời gian–tần số (STFT/Spectrogram)
            → Thí nghiệm cửa sổ → Lọc số FIR (LPF/HPF) → Lượng tử hóa, Resampling & Mã hóa
```

Toàn bộ kết quả, đồ thị (`figures/`) và tệp âm thanh đầu ra (`audio/`) đều được tái lập trực tiếp từ notebook `Lab1_main.ipynb`.

---

## 2. PHẦN A: ĐỌC VÀ KIỂM TRA DỮ LIỆU ÂM THANH

### 2.1. Bảng Metadata tệp âm thanh đầu vào
Dữ liệu đầu vào là tệp tiếng nói tiếng Anh được tạo bằng giọng nam `en-US-ChristopherNeural` (*"Hello, Good morning Sir. Welcome back. I hope you are having a wonderful day."*):

| Thuộc tính | Giá trị đo được | Ghi chú kỹ thuật |
| :--- | :--- | :--- |
| **Tên tệp** | `input/input.mp3` | Định dạng nén lossy MPEG Layer-3 |
| **Tần số lấy mẫu ($F_s$)** | **24,000 Hz** | Chu kỳ lấy mẫu $T = 1/F_s \approx 41.67\ \mu\text{s}$ |
| **Tần số Nyquist ($F_s / 2$)** | **12,000 Hz** | Tần số cao nhất biểu diễn không aliasing |
| **Số kênh ($C$)** | **1 (Mono)** | Tín hiệu đơn kênh, không cần trộn stereo |
| **Thời lượng ($Duration$)** | **7.440 s** | Tổng số mẫu $N = 178,560$ mẫu |
| **Kiểu dữ liệu xử lý** | `float32` chuẩn hóa $[-1, 1]$ | Chia cho $\max_n |x[n]|$ để chuẩn hóa |
| **Kích thước tệp MP3** | **44,640 bytes** ($\approx 43.59\text{ KB}$) | Bit rate trung bình thực tế $\approx 48.00\text{ kbps}$ |

### 2.2. Nhận xét kỹ thuật
* Tệp âm thanh đầu vào đã ở dạng **Mono 24 kHz**, có tần số Nyquist là $12\text{ kHz}$, hoàn toàn đáp ứng tiêu chuẩn bảo toàn đầy đủ dải tần tiếng nói con người (năng lượng formant và phụ âm xát chủ yếu nằm dưới $8\text{ kHz}$).
* Bước chuẩn hóa về miền $[-1, 1]$ giúp thống nhất thang biên độ cho các phép tính năng lượng, FFT, lọc số và lượng tử hóa về sau.

---

## 3. PHẦN B: PHÂN TÍCH MIỀN THỜI GIAN

### 3.1. Công thức và Số liệu toàn tệp
Các đại lượng miền thời gian được tính theo công thức:
$$\text{Peak} = \max_n |x[n]|, \qquad \text{RMS} = \sqrt{\frac{1}{N} \sum_{n=0}^{N-1} x^2[n]}, \qquad E = \sum_{n=0}^{N-1} x^2[n]$$

| Đại lượng | Toàn tệp (`0.0 – 7.44 s`) | Đoạn 1 (`0.5 – 1.0 s`) | Đoạn 2 (`5.5 – 6.5 s`) |
| :--- | :---: | :---: | :---: |
| **Peak (biên độ chuẩn hóa)** | **1.0000** | 0.8843 | 0.8756 |
| **RMS** | **0.1177** (`-18.59 dBFS`) | 0.1172 (`-18.62 dBFS`) | 0.1405 (`-17.05 dBFS`) |
| **Năng lượng ($E$)** | **2472.46** | 164.88 | 473.62 |
| **Kiểm tra Clipping** | Cảnh báo (`Peak >= 0.99`) | Không clipping | Không clipping |

### 3.2. Đồ thị Waveform
![Waveform toàn tệp](figures/waveform_full.png)
*Hình 1: Dạng sóng (waveform) chuẩn hóa của toàn bộ tệp `input/input.mp3` ($F_s = 24\text{ kHz}$, dài $7.440\text{ s}$).*

![Waveform đoạn 0.5-1.0s](figures/waveform_segment.png)
*Hình 2: Phóng to dạng sóng đoạn $0.5 - 1.0\text{ s}$ cho thấy cấu trúc dao động tuần hoàn của âm hữu thanh và khoảng lặng ngắn.*

### 3.3. Nhận xét kỹ thuật
* Toàn tệp có **Peak = 1.0000** do phép chia chuẩn hóa $x_{\text{norm}} = x / \max|x|$ đẩy mẫu lớn nhất chạm đúng biên $1.0$; tuy nhiên mức hiệu dụng **RMS = 0.1177 (-18.59 dBFS)** khá cân bằng, cho thấy tỷ số đỉnh trên hiệu dụng (crest factor $\approx 18.6\text{ dB}$) điển hình của tín hiệu tiếng nói có nhiều khoảng nghỉ giữa các từ/câu.
* Khi so sánh hai đoạn đặc trưng: đoạn `0.5 – 1.0 s` chứa khoảng nghỉ chuyển tiếp sau câu chào nên có **RMS = 0.1172** và năng lượng trung bình thấp hơn, trong khi đoạn `5.5 – 6.5 s` là vùng phát âm liên tục giàu nguyên âm hữu thanh nên có **RMS = 0.1405 (-17.05 dBFS)** cao hơn rõ rệt.

---

## 4. PHẦN C: PHÂN TÍCH MIỀN TẦN SỐ BẰNG FFT

### 4.1. Thiết lập và Công thức
Chọn đoạn ổn định dài $1.0\text{ s}$ từ $5.5\text{ s}$ đến $6.5\text{ s}$ ($24,000$ mẫu), nhân cửa sổ Hamming $w[n]$ và tính biến đổi FFT thực (`np.fft.rfft`):
$$X[k] = \sum_{n=0}^{N-1} x[n] w[n] e^{-j 2\pi k n / N_{\text{FFT}}}, \qquad \Delta f = \frac{F_s}{N_{\text{FFT}}}, \qquad \text{Mag}_{\text{dB}}[k] = 20 \log_{10}(|X[k]| + 10^{-12})$$

### 4.2. Kết quả đo đạc và So sánh NFFT

| Tham số FFT | $N_{\text{FFT}} = 65,536$ | $N_{\text{FFT}} = 2,048$ |
| :--- | :---: | :---: |
| **Frequency-bin spacing ($\Delta f = F_s / N_{\text{FFT}}$)** | **0.366 Hz** | **11.719 Hz** |
| **Độ dài đoạn dữ liệu gốc** | $24,000$ mẫu ($1.0\text{ s}$) | $24,000$ mẫu (cắt/dùng $2,048$ mẫu nếu không đệm) |
| **3 đỉnh phổ nổi bật đo được** | **240.60 Hz**, **127.44 Hz**, **191.89 Hz** | Độ mịn lưới tần số thô hơn ~32 lần |

![Phổ FFT đoạn 5.5-6.5s](figures/fft_spectrum.png)
*Hình 3: Phổ biên độ tương đối (relative dB) của đoạn $5.5 - 6.5\text{ s}$ (cửa sổ Hamming, $N_{\text{FFT}} = 65,536$, hiển thị dải $0 - 6,000\text{ Hz}$).*

### 4.3. Nhận xét kỹ thuật
* Ba đỉnh phổ nổi bật nhất tập trung ở vùng tần số thấp (**127.44 Hz, 191.89 Hz, 240.60 Hz**), phản ánh đúng tần số cơ bản $F_0$ (~120–127 Hz) và hài bậc hai $2F_0$ (~240 Hz) của giọng nam trầm.
* Năng lượng tiếng nói tập trung chủ yếu dưới $4\text{ kHz}$ và suy giảm dần về phía $6\text{ kHz}$.
* **Phân biệt frequency-bin spacing và true resolution:** Tăng $N_{\text{FFT}}$ từ $2,048$ lên $65,536$ bằng cách zero-padding giúp làm mịn bước lưới tần số $\Delta f$ từ $11.719\text{ Hz}$ xuống $0.366\text{ Hz}$ (nội suy phổ mượt hơn), nhưng **không làm tăng độ phân giải vật lý thực (true resolution)** vốn được quyết định bởi độ dài cửa sổ thời gian thực tế $L$ và loại cửa sổ.

---

## 5. PHẦN D: STFT VÀ SPECTROGRAM

### 5.1. Thiết lập thí nghiệm có kiểm soát
Giữ cố định $F_s = 24,000\text{ Hz}$, cửa sổ `hamming`, $N_{\text{FFT}} = 2,048$, tỷ lệ chồng lấn $\approx 60\%$ (`hop = frame / 2.5`), chỉ thay đổi độ dài khung `frame_length` qua 3 mức: **10 ms**, **25 ms** (cấu hình chuẩn, `hop = 10 ms`) và **50 ms**.

| Cấu hình | Frame length ($L$) | Hop size ($H$) | Độ phân giải thời gian | Độ phân giải tần số |
| :--- | :---: | :---: | :---: | :---: |
| **Ngắn (10 ms)** | 240 mẫu | 96 mẫu (4 ms) | Rất tốt (sắc nét theo trục ngang) | Kém (các vạch hài bị nhòe theo trục dọc) |
| **Chuẩn (25 ms)** | 600 mẫu | 240 mẫu (10 ms) | Cân bằng tốt | Tách rõ các dải formant và hài cơ bản |
| **Dài (50 ms)** | 1,200 mẫu | 480 mẫu (20 ms) | Kém (làm mờ các chuyển tiếp nhanh) | Rất tốt (hiện rõ từng đường hài song song) |

### 5.2. Đồ thị Spectrogram so sánh 3 độ dài khung

![Spectrogram 10ms](figures/spectrogram_10ms.png)
*Hình 4: Spectrogram với khung ngắn 10 ms — bắt rất rõ các biến cố tức thời (transient) và ranh giới âm tiết.*

![Spectrogram 25ms](figures/spectrogram_25ms.png)
*Hình 5: Spectrogram cấu hình chuẩn (Frame 25 ms, Hop 10 ms) — cân bằng tối ưu giữa thời gian và tần số.*

![Spectrogram 50ms](figures/spectrogram_50ms.png)
*Hình 6: Spectrogram với khung dài 50 ms — các đường hài (harmonics) nằm ngang hiện ra rất sắc nét nhưng ranh giới thời gian bị làm mờ.*

### 5.3. Nhận xét kỹ thuật
* Trên cả 3 hình đều thấy rõ **vùng năng lượng ổn định** (các nguyên âm kéo dài với các dải formant sáng rõ dưới $3.5\text{ kHz}$), **vùng transient** (các phụ âm bật/xát trải năng lượng lên vùng $4 - 6\text{ kHz}$ trong thời gian rất ngắn) và **khoảng lặng** giữa hai câu nói.
* Kết quả minh họa trực quan **nguyên lý bất định thời gian – tần số (Time–Frequency Resolution Trade-off)**: khung `10 ms` cho độ phân giải thời gian cao (wideband spectrogram), khung `50 ms` cho độ phân giải tần số cao (narrowband spectrogram), còn khung `25 ms` (hop `10 ms`) là điểm cân bằng chuẩn trong xử lý tiếng nói.

---

## 6. PHẦN E: THÍ NGHIỆM CỬA SỔ (RECTANGULAR VS HAMMING)

### 6.1. Thiết lập có kiểm soát
Giữ nguyên đoạn dữ liệu `5.5 – 6.5 s` ($24,000$ mẫu) và $N_{\text{FFT}} = 65,536$, chỉ thay đổi hàm cửa sổ:
* **Rectangular:** $w_{\text{rect}}[n] = 1$ (không nhân cửa sổ).
* **Hamming:** $w_{\text{ham}}[n] = 0.54 - 0.46 \cos\left(\frac{2\pi n}{L-1}\right)$.

![So sánh cửa sổ Rectangular vs Hamming](figures/window_comparison.png)
*Hình 7: So sánh phổ log-magnitude (relative dB, dải $0 - 2,000\text{ Hz}$) giữa cửa sổ Rectangular và Hamming trên cùng một đoạn tín hiệu.*

### 6.2. Nhận xét kỹ thuật
* **Side-lobe & Spectral Leakage:** Cửa sổ **Hamming** làm giảm mạnh mức sàn nhiễu (side-lobe thấp hơn rõ rệt, suy giảm đỉnh side-lobe thứ nhất khoảng $-43\text{ dB}$ so với $-13\text{ dB}$ của Rectangular), từ đó giảm thiểu **rò rỉ phổ (spectral leakage)** và giúp các đỉnh phổ thật nổi bật sạch sẽ hơn trên nền phổ.
* **Main-lobe:** Đổi lại, búp chính (main-lobe) của Hamming rộng gấp đôi Rectangular ($8\pi/L$ so với $4\pi/L$), làm các đỉnh tần số gần nhau có xu hướng hơi bè rộng hơn một chút.

---

## 7. PHẦN F: THIẾT KẾ VÀ ÁP DỤNG BỘ LỌC SỐ FIR

### 7.1. Thông số thiết kế bộ lọc
Sử dụng `scipy.signal.firwin` thiết kế hai bộ lọc FIR tuyến tính pha đối xứng:
* **Low-pass FIR:** `numtaps = 201` (bậc $M = 200$), `cutoff = 2,000 Hz`, $F_s = 24,000\text{ Hz}$, `window = 'hamming'`. Xuất tệp: `audio/filtered_lowpass_2k.wav`.
* **High-pass FIR:** `numtaps = 201`, `cutoff = 2,000 Hz`, $F_s = 24,000\text{ Hz}$, `window = 'hamming'`, `pass_zero = False`. Xuất tệp: `audio/filtered_highpass_2k.wav`.
* **Độ trễ nhóm (Group Delay):**
  $$\tau_g = \frac{L - 1}{2} = \frac{201 - 1}{2} = 100\text{ mẫu} = \frac{100}{24,000} \approx \mathbf{4.17\text{ ms}}$$

### 7.2. Đáp ứng tần số $H(f)$ và Phổ trước/sau lọc

![Đáp ứng tần số FIR Low-pass 2kHz](figures/filter_response_lp.png)
*Hình 8: Đáp ứng biên độ tần số $|H(f)|$ (dB) của bộ lọc FIR Low-pass 2 kHz, 201 taps, cửa sổ Hamming.*

![Phổ trước và sau lọc Low-pass 2kHz](figures/spectrum_before_after_filter.png)
*Hình 9: So sánh phổ biên độ tương đối (dB) trước và sau khi qua bộ lọc FIR Low-pass 2 kHz.*

### 7.3. Nhận xét kỹ thuật
* Trên Hình 8, dải thông ($0 - 2\text{ kHz}$) có đáp ứng phẳng xấp xỉ $0\text{ dB}$, dải chuyển tiếp hẹp quanh $2\text{ kHz}$ và dải chắn ($> 2.3\text{ kHz}$) suy hao mạnh xuống dưới $-50\text{ dB}$ đến $-70\text{ dB}$ đúng đặc trưng cửa sổ Hamming.
* Trên Hình 9, phổ sau lọc (`filtered_lowpass_2k.wav`) giữ nguyên hoàn toàn năng lượng dưới $2\text{ kHz}$ và bị cắt suy giảm mạnh trên $2\text{ kHz}$, khớp hoàn toàn với $|H(f)|$.
* **Cảm nhận nghe:** Bản **Low-pass 2 kHz** nghe ấm nhưng "tối/nghẹt" (muffled) do mất năng lượng cao của các phụ âm xát (/s/, /f/) và độ sắc nét; ngược lại, bản **High-pass 2 kHz** mất toàn bộ giọng nền và $F_0$, chỉ còn nghe thấy tiếng gió/xì mỏng ở tần số cao.

---

## 8. PHẦN G: LƯỢNG TỬ HÓA, RESAMPLING VÀ MÃ HÓA

### 8.1. Thí nghiệm Lượng tử hóa đều và SNR
Sử dụng hàm lượng tử đều đối xứng trên $[-1, 1]$ với $q_{\max} = 2^{B-1} - 1$ và đo SNR thực nghiệm:
$$\text{SNR} = 10 \log_{10} \left( \frac{\sum_n x^2[n]}{\sum_n (\hat{x}[n] - x[n])^2} \right) \text{ (dB)}$$

So sánh với công thức lý thuyết Rabiner–Schafer $\text{SNR}_Q = 6B + 4.77 - 20 \log_{10}(X_{\max}/\sigma_x)$, với $X_{\max} = 1.0$, $\sigma_x = \text{RMS} = 0.1177 \Rightarrow 20 \log_{10}(1/0.1177) = 18.59\text{ dB}$:

| Số bit/mẫu ($B$) | Số mức ($2^B$) | SNR lý thuyết ước lượng (dB) | SNR đo thực nghiệm (dB) | Chênh lệch mỗi +4 bit | Tệp xuất |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **4 bit** | 16 | $10.18\text{ dB}$ | **12.97 dB** | — | — |
| **8 bit** | 256 | $34.18\text{ dB}$ | **36.71 dB** | $+23.74\text{ dB}$ ($\approx 5.94\text{ dB/bit}$) | `audio/quantized_8bit.wav` |
| **16 bit** | 65,536 | $82.18\text{ dB}$ | **84.10 dB** | $+47.39\text{ dB}$ ($\approx 5.92\text{ dB/bit}$) | — |

*Nhận xét:* SNR thực nghiệm tăng gần tuyến tính ở mức $\approx 5.93\text{ dB/bit}$, rất sát quy luật $6.02\text{ dB/bit}$. Giá trị đo thực tế lệch nhẹ so với công thức lý thuyết do tín hiệu tiếng nói không phân bố đều trên $[-1, 1]$ mà tập trung nhiều ở biên độ nhỏ (có nhiều khoảng lặng) và cách làm tròn theo $q_{\max} = 2^{B-1}-1$. Nhiễu lượng tử (tiếng rè/rít nền) nghe rõ nhất ở các vùng tín hiệu biên độ nhỏ và khoảng lặng, đặc biệt ở 4 bit và 8 bit.

### 8.2. Thí nghiệm Resampling (16 kHz và 8 kHz)
Áp dụng bộ lọc chống aliasing tích hợp trong `librosa.resample` để chuyển từ $24\text{ kHz}$ về $16\text{ kHz}$ (`audio/resampled_16000Hz.wav`, $238,124\text{ B}$) và $8\text{ kHz}$ (`audio/resampled_8000Hz.wav`, $119,084\text{ B}$):
* **Không xuất hiện aliasing** vì bộ resampler chuẩn đã lọc thông thấp dưới tần số Nyquist mới ($8\text{ kHz}$ và $4\text{ kHz}$) trước khi giảm mẫu.
* Ở **16 kHz**, chất lượng tiếng nói vẫn rất trong và tự nhiên; xuống **8 kHz**, dải tần bị giới hạn dưới $4\text{ kHz}$ khiến âm thanh nghe giống chất lượng điện thoại truyền thống, giảm rõ độ sắc nét của phụ âm xát.

### 8.3. Tính toán PCM Bit rate, Kích thước tệp và Compression Ratio
Áp dụng công thức $R_{\text{PCM}} = F_s \times B \times C$ và $\text{Size} \approx R_{\text{PCM}} \times \text{Duration} / 8$:

| Định dạng so sánh | Tần số lấy mẫu | Kênh | Bit rate | Kích thước (7.440 s) | Compression Ratio (so với PCM tương ứng) | Saving (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PCM 16-bit Mono (tương ứng tệp gốc)** | 24,000 Hz | 1 | **384.0 kbps** | **357,120 B** (0.341 MB) | 1.00 : 1 | 0.00% |
| **MP3 gốc (`input/input.mp3`)** | 24,000 Hz | 1 | **48.0 kbps** | **44,640 B** (0.043 MB) | **8.00 : 1** (so với PCM mono) | **87.50%** |
| **PCM 16-bit Stereo (tham chiếu trong notebook)** | 24,000 Hz | 2 | **768.0 kbps** | **714,240 B** (0.681 MB) | 1.00 : 1 | 0.00% |
| **MP3 256 kbps (tham chiếu)** | 24,000 Hz | 2 | 256.0 kbps | 238,080 B (0.227 MB) | **3.00 : 1** (so với PCM stereo) | 66.67% |
| **MP3 128 kbps (tham chiếu)** | 24,000 Hz | 2 | 128.0 kbps | 119,040 B (0.114 MB) | **6.00 : 1** (so với PCM stereo) | 83.33% |

*Lưu ý:* Tệp `input.mp3` vốn đã nén lossy ở $48\text{ kbps}$, nên việc giải mã sang WAV không thể phục hồi thông tin tần số đã bị loại bỏ bởi bộ mã hóa perceptual trước đó.

---

## 9. TỔNG HỢP MA TRẬN KẾT QUẢ TỐI THIỂU VÀ THÍ NGHIỆM ĐỘC LẬP

| Khối / Thí nghiệm | Thiết lập đã chạy | Bằng chứng đầu ra | Tóm tắt kết quả chính |
| :--- | :--- | :--- | :--- |
| **A–B. Metadata & Time** | `input/input.mp3`, chuẩn hóa $[-1,1]$ | `waveform_full.png`, `waveform_segment.png` | $F_s=24\text{ kHz}$, Mono, $7.440\text{ s}$, Peak $=1.0$, RMS $=0.1177$ (`-18.59 dBFS`) |
| **C. FFT** | Đoạn `5.5–6.5s`, Hamming, NFFT 2048 vs 65536 | `fft_spectrum.png` | $\Delta f = 11.719\text{ Hz}$ vs $0.366\text{ Hz}$; 3 đỉnh: `240.60`, `127.44`, `191.89 Hz` |
| **D. STFT / Spectrogram** | Frame `10 / 25 / 50 ms`, overlap ~60% | `spectrogram_10ms/25ms/50ms.png` | Minh họa rõ trade-off thời gian (10 ms) vs tần số (50 ms) |
| **E. Window** | Rectangular vs Hamming, NFFT 65536 | `window_comparison.png` | Hamming giảm mạnh side-lobe/leakage nhưng main-lobe rộng hơn |
| **F. Filtering** | FIR LPF & HPF 2 kHz, 201 taps, Hamming | `filter_response_lp.png`, `spectrum_before_after_filter.png`, `filtered_*.wav` | Dải chắn $<-50\text{ dB}$, group delay $\approx 4.17\text{ ms}$, phổ khớp $H(f)$ |
| **G. Quant/Resample/Coding** | 4/8/16 bit; 16/8 kHz; PCM vs MP3 | `quantized_8bit.wav`, `resampled_*.wav` | SNR: `12.97 / 36.71 / 84.10 dB`; CR mono `8.00:1` (saving `87.50%`) |

---

## 10. GIẢI ĐÁP CHI TIẾT 7 CÂU HỎI BÁO CÁO (MỤC 6)

### Câu 1: Giải thích bằng công thức tại sao $F_s = 44.1\text{ kHz}$ chỉ biểu diễn độc lập đến $22.05\text{ kHz}$.
Theo định lý lấy mẫu Nyquist–Shannon, tín hiệu băng tần giới hạn $F_{\max}$ chỉ có thể khôi phục chính xác không bị chồng phổ (aliasing) khi:
$$F_s \ge 2 F_{\max} \iff F_{\max} \le \frac{F_s}{2}$$
Với $F_s = 44,100\text{ Hz}$, tần số Nyquist là $F_N = \frac{44,100}{2} = \mathbf{22,050\text{ Hz} = 22.05\text{ kHz}}$. Về mặt toán học, với tín hiệu thực $x[n]$, phổ DTFT/DFT có tính đối xứng liên hợp ($|X(F_s - f)| = |X(f)|$) và tuần hoàn theo chu kỳ $F_s$. Mọi thành phần tần số $f > 22.05\text{ kHz}$ khi lấy mẫu ở $44.1\text{ kHz}$ sẽ bị gập (alias) về tần số $|f - k F_s| \le 22.05\text{ kHz}$, không thể phân biệt độc lập với thành phần tần số thấp hơn.

### Câu 2: Nếu NFFT tăng từ 2048 lên 8192 nhưng frame vẫn dài 25 ms, điều gì thật sự thay đổi và điều gì không?
* **Điều thật sự thay đổi:** Bước khoảng cách giữa hai bin tần số liền kề (frequency-bin spacing) $\Delta f = F_s / N_{\text{FFT}}$ giảm đi **4 lần** (lưới điểm lấy mẫu trên phổ DTFT dày hơn 4 lần nhờ chèn thêm số 0 – zero-padding), giúp đồ thị phổ trông trơn mượt hơn và dễ đọc chính xác hơn vị trí đỉnh phổ cô lập.
* **Điều KHÔNG thay đổi:** **Độ phân giải tần số vật lý thực (true frequency resolution)** không tăng, vì zero-padding không bổ sung thêm bất kỳ thông tin tín hiệu mới nào. Khả năng phân tách hai thành phần sin gần nhau vẫn hoàn toàn bị giới hạn bởi bề rộng búp chính (main-lobe width $\propto 1/T_{\text{frame}}$) của cửa sổ có độ dài cố định $25\text{ ms}$.

### Câu 3: Tại sao Hamming giảm spectral leakage so với rectangular nhưng có thể làm các đỉnh gần nhau khó phân tách hơn?
* Cửa sổ **Rectangular** cắt đột ngột tín hiệu ở hai đầu frame, tạo ra sự gián đoạn mạnh trên miền thời gian nên phổ của nó có các **side-lobe rất cao** (đỉnh phụ thứ nhất chỉ thấp hơn búp chính $-13\text{ dB}$), làm năng lượng từ tần số mạnh tràn sang các bin lân cận (**spectral leakage**).
* Cửa sổ **Hamming** vuốt êm biên độ tín hiệu về gần $0$ ở hai đầu, giúp nén side-lobe xuống xấp xỉ **$-43\text{ dB}$**, giảm leakage rất mạnh. Tuy nhiên, theo nguyên lý biến đổi Fourier, việc thu hẹp năng lượng cửa sổ theo thời gian làm **búp chính (main-lobe) mở rộng gấp đôi** ($8\pi/L$ so với $4\pi/L$). Khi hai đỉnh tần số nằm quá gần nhau, hai main-lobe rộng của cửa sổ Hamming sẽ chồng lấp vào nhau, khiến chúng khó phân tách thành hai đỉnh riêng biệt hơn.

### Câu 4: Với FIR 201 taps đối xứng tại 44.1 kHz, độ trễ xấp xỉ bao nhiêu mili giây? Độ trễ đó có quan trọng trong xử lý thời gian thực không?
Với bộ lọc FIR tuyến tính pha đối xứng độ dài $L = 201$ taps tại $F_s = 44,100\text{ Hz}$:
$$\tau_g = \frac{L - 1}{2} = \frac{201 - 1}{2} = 100\text{ mẫu} \implies t_g = \frac{100}{44,100} \approx \mathbf{2.27\text{ ms}}$$
* **Ý nghĩa trong thời gian thực:** Với các ứng dụng hội thoại thông thường (VoIP, trợ lý ảo có ngưỡng chịu trễ tổng $\sim 50 - 150\text{ ms}$), độ trễ $2.27\text{ ms}$ là **rất nhỏ và hoàn toàn chấp nhận được**. Tuy nhiên, trong các hệ thống yêu cầu độ trễ siêu thấp như **kiểm âm phòng thu trực tiếp (live monitoring)**, **máy trợ thính (hearing aids)** hoặc **chống ồn chủ động (ANC)** (ngưỡng trễ cho phép thường $< 5 - 10\text{ ms}$ cho toàn chuỗi), việc xếp chồng nhiều tầng lọc FIR dài như vậy sẽ trở nên quan trọng và cần được cân nhắc kỹ (hoặc chuyển sang IIR/minimum-phase FIR).

### Câu 5: Từ công thức $\text{SNR}_Q$, giải thích ảnh hưởng của $B$ và $\sigma_x$. Tại sao giảm mức tín hiệu đầu vào có thể làm SNR lượng tử giảm?
Công thức Rabiner–Schafer cho lượng tử đều $B$-bit trên khoảng $[-X_{\max}, X_{\max}]$:
$$\text{SNR}_Q(\text{dB}) = 6B + 4.77 - 20 \log_{10}\left(\frac{X_{\max}}{\sigma_x}\right)$$
* **Ảnh hưởng của $B$:** Mỗi bit bổ sung làm tăng gấp đôi số mức lượng tử, giảm bước lượng tử $\Delta$ đi một nửa $\Rightarrow$ công suất nhiễu $\Delta^2/12$ giảm $4$ lần $\Rightarrow$ $\text{SNR}_Q$ tăng xấp xỉ **$+6\text{ dB/bit}$**.
* **Ảnh hưởng của $\sigma_x$:** Với một bộ ADC cố định có biên độ full-scale $X_{\max}$ và bước $\Delta$ không đổi, công suất nhiễu lượng tử $P_e \approx \Delta^2/12$ là hằng số. Khi giảm mức tín hiệu đầu vào (giảm RMS $\sigma_x$), tỷ số $X_{\max}/\sigma_x$ tăng lên, số hạng $-20\log_{10}(X_{\max}/\sigma_x)$ âm sâu hơn $\Rightarrow$ **$\text{SNR}_Q$ giảm trực tiếp** (tín hiệu nhỏ chỉ sử dụng được một phần nhỏ số mức lượng tử có sẵn trong khi sàn nhiễu lượng tử giữ nguyên).

### Câu 6: Một file WAV 16-bit stereo 44.1 kHz dài 60 s có kích thước PCM lý thuyết bao nhiêu MB? So sánh với MP3 128 kbps.
* **Kích thước PCM lý thuyết (WAV 16-bit, Stereo $C=2$, $F_s = 44,100\text{ Hz}$, $D = 60\text{ s}$):**
  $$R_{\text{PCM}} = 44,100 \times 16 \times 2 = 1,411,200\text{ bit/s} = 1,411.2\text{ kbps}$$
  $$\text{Size}_{\text{PCM}} = \frac{1,411,200 \times 60}{8} = 10,584,000\text{ bytes} \approx \mathbf{10.58\text{ MB}} \text{ (hệ thập phân) hay } \mathbf{10.09\text{ MiB}} \text{ (hệ nhị phân } 1024^2\text{)}$$
* **Kích thước MP3 128 kbps dài 60 s:**
  $$\text{Size}_{\text{MP3}} = \frac{128,000 \times 60}{8} = 960,000\text{ bytes} = \mathbf{0.96\text{ MB}} \text{ (hay } \mathbf{0.916\text{ MiB}}\text{)}$$
* **So sánh:**
  $$\text{Compression Ratio} = \frac{1,411.2}{128} \approx \mathbf{11.03 : 1}, \qquad \text{Saving} = \left(1 - \frac{960,000}{10,584,000}\right) \times 100\% \approx \mathbf{90.93\%}$$
  MP3 128 kbps nhỏ gọn hơn gấp **11 lần** và tiết kiệm hơn **90.9%** dung lượng lưu trữ so với WAV PCM chuẩn CD.

### Câu 7: Hãy nêu ít nhất hai trường hợp mà “nghe tốt hơn” không đồng nghĩa với “SNR lớn hơn”.
1. **Mã hóa âm thanh theo cảm nhận (Perceptual Audio Coding như MP3/AAC):** Bộ mã hóa tận dụng hiệu ứng che lấp thính giác (auditory masking) để loại bỏ các thành phần tần số mà tai người không nghe thấy và phân bố nhiễu lượng tử vào các dải bị che lấp. Khi so sánh từng mẫu dạng sóng với bản gốc, sai số $e[n] = \hat{x}[n] - x[n]$ có thể khá lớn (SNR thấp hơn nhiều so với PCM nén nhẹ), nhưng tai người nghe lại gần như trong suốt, tự nhiên hơn hẳn một tín hiệu lượng tử đều có SNR tương đương.
2. **Lọc nhiễu/Khử tạp âm hoặc Dịch pha/Trễ thời gian (Phase Shift / Group Delay / Spectral Enhancement):** Một bộ lọc làm sạch tiếng nói có thể gây trễ pha hoặc thay đổi nhẹ biên độ dạng sóng (làm SNR từng mẫu giảm mạnh, thậm chí âm nếu không căn chỉnh trễ), nhưng người nghe lại cảm thấy tiếng nói sạch sẽ, dễ nghe hơn nhiều. Ngược lại, một đoạn âm thanh bị nhiễu "musical noise" hoặc tiếng rít đơn tần có năng lượng nhiễu nhỏ (SNR cao) nhưng lại gây khó chịu cho tai người hơn so với nhiễu nền trắng dịu nhẹ có SNR thấp hơn.

---

## 11. KẾT LUẬN VÀ TÀI LIỆU THAM KHẢO

### Kết luận
Bài thực hành Lab 1 đã hoàn thiện trọn vẹn quy trình phân tích và xử lý tín hiệu âm thanh số từ A đến G trên dữ liệu thực tế `input/input.mp3` ($24\text{ kHz}$, Mono, $7.440\text{ s}$), đầy đủ các biểu đồ trực quan có trục/đơn vị rõ ràng, các tệp WAV đầu ra có kiểm soát và đối chứng số liệu định lượng chặt chẽ với lý thuyết DSP.

### Tài liệu tham khảo
1. Đề cương chi tiết học phần CSE457 – *Xử lý âm thanh và tiếng nói*, Trường Đại học Thủy lợi, 2023.
2. X. Huang, A. Acero, H.-W. Hon, *Spoken Language Processing*, Prentice-Hall, 2001.
3. D. Jurafsky, J. H. Martin, *Speech and Language Processing*, Prentice-Hall, 2008.
4. L. R. Rabiner, R. W. Schafer, *Theory and Applications of Digital Speech Processing*, Prentice-Hall/Pearson, 2010.
