# Công cụ Giải Bài toán Quy hoạch Tuyến tính (Linear Programming Solver)

Đây là một ứng dụng web được xây dựng bằng Flask (Python) nhằm cung cấp một công cụ mạnh mẽ và trực quan để giải các bài toán Quy hoạch Tuyến tính (QHTT). Người dùng có thể dễ dàng nhập bài toán, lựa chọn phương pháp giải và nhận được kết quả chi tiết từng bước.

*\[Hình ảnh giao diện nhập liệu của ứng dụng\]*

## ✨ Tính năng nổi bật

* **Giải các loại bài toán QHTT**: Hỗ trợ cả bài toán **Tối đa hóa (Max)** và **Tối thiểu hóa (Min)**.
* **Ràng buộc linh hoạt**: Chấp nhận các loại ràng buộc khác nhau: `≤` (nhỏ hơn hoặc bằng), `≥` (lớn hơn hoặc bằng), và `=` (bằng).
* **Đa dạng loại biến**: Hỗ trợ các biến không âm (`>=0`), không dương (`<=0`), và biến tự do (URS - Unrestricted in Sign).
* **Hai phương pháp giải mạnh mẽ**:
    1.  **Đơn hình với Quy tắc Bland**: Giải các bài toán dạng chuẩn tắc, sử dụng quy tắc Bland để tránh hiện tượng xoay vòng (cycling).
    2.  **Đơn hình Hai Pha (Two-Phase Simplex)**: Một phương pháp tổng quát để giải bất kỳ bài toán QHTT nào, đặc biệt hữu ích khi không có phương án cơ sở ban đầu.
* **Kết quả chi tiết**:
    * Hiển thị trạng thái cuối cùng của bài toán: Tối ưu, Vô số nghiệm, Không giới nội, hoặc Vô nghiệm.
    * Cung cấp giá trị tối ưu của hàm mục tiêu (`z*`) và nghiệm tối ưu của các biến.
    * Trình bày **chi tiết từng bước lặp (tableau)** của thuật toán, giúp người dùng dễ dàng theo dõi và kiểm tra quá trình giải.
* **Trực quan hóa đồ thị**: Đối với các bài toán có 2 biến, ứng dụng sẽ tự động vẽ đồ thị biểu diễn miền khả thi, các đường ràng buộc và điểm tối ưu.
* **Giao diện thân thiện**: Giao diện web được thiết kế gọn gàng, dễ sử dụng với các chú thích và hướng dẫn rõ ràng.

*\[Hình ảnh trang kết quả với đồ thị và các bước giải chi tiết\]*

## 🌐 Triển khai & Trạng thái Trực tuyến (Deployment & Status)

Ứng dụng này đã được triển khai và đang hoạt động trực tuyến trên nền tảng **Render.com**.

* **Link truy cập**: **[https://project-lps.onrender.com](https://project-lps.onrender.com)**

**Lưu ý quan trọng**:
Dự án được host trên gói **Miễn phí (Free Tier)** của Render, do đó sẽ có một số giới hạn:
* **Giới hạn thời gian hoạt động**: Gói miễn phí cung cấp **750 giờ** hoạt động mỗi tháng. Nếu vượt quá giới hạn này, trang web có thể tạm thời không truy cập được cho đến chu kỳ tiếp theo. (Trạng thái hiện tại: đã sử dụng **12.33 / 750 giờ**).
* **Chế độ ngủ (Sleep Mode)**: Nếu không có ai truy cập trang web trong một khoảng thời gian, nó sẽ tự động chuyển sang chế độ ngủ để tiết kiệm tài nguyên. Khi bạn truy cập lần đầu, có thể sẽ mất khoảng **30-60 giây** để máy chủ "thức dậy" và tải trang. Vui lòng kiên nhẫn chờ.

## 🛠️ Công nghệ sử dụng

* **Backend**:
    * **Python**: Ngôn ngữ lập trình chính.
    * **Flask**: Một web framework nhẹ để xây dựng ứng dụng.
    * **SymPy**: Thư viện cho tính toán biểu tượng, dùng để xử lý các biểu thức toán học trong bảng đơn hình.
    * **NumPy**: Thư viện cho tính toán số học, hỗ trợ các phép toán ma trận.
    * **Matplotlib**: Thư viện để vẽ đồ thị cho các bài toán 2 biến.
* **Frontend**:
    * **HTML5**: Ngôn ngữ đánh dấu cấu trúc trang web.
    * **Tailwind CSS**: Một CSS framework để xây dựng giao diện người dùng một cách nhanh chóng.
    * **JavaScript**: Xử lý các tương tác phía client như cập nhật form động.

## 🚀 Cài đặt và Chạy dự án

Để chạy dự án này trên máy cục bộ của bạn, hãy làm theo các bước sau:

**1. Clone Repository**
    ```
    git clone https://github.com/ldgk2712/project-lps.git
    ```
**2. Tạo và kích hoạt môi trường ảo (Khuyến khích)**

* Đối với Windows:
    ```
    python -m venv venv
    .\venv\Scripts\activate
    ```
* Đối với macOS/Linux:
    ```
    python3 -m venv venv
    source venv/bin/activate
    ```

**3. Cài đặt các thư viện cần thiết**
 ```
pip install -r requirements.txt
 ```

**4. Chạy ứng dụng Flask**
 ```
Sau khi cài đặt xong, bạn có thể khởi chạy server Flask:
python app.py
 ```

**5. Truy cập ứng dụng**

Mở trình duyệt web của bạn và truy cập vào địa chỉ sau:
<http://127.0.0.1:5000>

Bây giờ bạn đã có thể bắt đầu sử dụng công cụ để giải các bài toán QHTT của mình!

## 📂 Cấu trúc Thư mục
```
├── app.py                     # File chính của Flask, xử lý logic web
├── simplex_bland.py           # Module chứa thuật toán Đơn hình (Quy tắc Bland)
├── simplex_two_phase.py       # Module chứa thuật toán Đơn hình Hai Pha
├── requirements.txt           # Danh sách các thư viện Python cần thiết
└── templates\
    ├── index.html             # Giao diện trang nhập liệu
    └── result.html            # Giao diện trang hiển thị kết quả
```
