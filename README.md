# Bài toán
Nhận diện thủ ngữ - ký hiệu tay .<br>
Đầu vào: dữ liệu thủ ngữ từ webcam .<br>
Đầu ra: chuyển thể thủ ngữ sang ký hiệu chữ viết và hiển thị ra màn hình .<br>

### Các thư mục và file trong project
preprocessing_and_training.py dùng để xử lý dữ liệu training model và thực hiện training model.<br>
sign_langguage: model sau khi đã train được lưu tại đây.<br>
real_time.py dùng để nhận dữ liệu thực tế đầu vào từ webcam, đưa và model đã lưu để nhận dạng và hiển thị dự đoán cho người dùng.<br>
data/...: Folder chứa dưa liệu để train và test model.<br>


### Cách giải quyết bài toán
Dữ liệu từ webcam sẽ được detect thành các ảnh liên tiếp để chuyển thể video sang dạng ảnh.<br>
Sử dụng deep learning model phân loại ảnh thủ ngữ đầu vào thành 24 ký tự Alphabel.<br>
Data ảnh train model: https://www.kaggle.com/datamunge/sign-language-mnist.<br>

### Mô tả dữ liệu đầu vào train model
Các ảnh có size 28x28 pixel = 784 pixel, giá trị của từng pixel tương ứng là các cột.<br>
Có 24 phân loại Aphabet tương ứng là các hàng.<br>
Tập train có 27455 ảnh => dữ liệu train lưu dưới dạng csv có 27455 hàng x 784 cột.<br>
Tập test có 7172 ảnh => dữ liệu test lưu dạng csv có 7172 hàng x 784 cột.<br>

### Mô tả các layer trong mạng neuron network
CONV2D->RELU->MAXPOOLING->CONV2D->RELU->MAXPOOLING->DROPOUT->CONV2D->RELU->MAXPOOLING->DROPOUT->FLATTEN->DENSE->DROPOUT-> DENSE->SOFTMAX


### Độ chính xác dữ liệu train = 99.64% 
### Độ chính xác dữ liệu test    = 97.02%




## Các bước 
B1: Cài đặt python >= 3.10 và git.<br>
B2: Mở terminal tại nơi bạn định lưu project và chạy lệnh: "git clone https://github.com/trangtran-dev/ThuNguToText".<br>
B3: Tạo folder data trong project.<br>
B4: Tải và giải nén các file và folder trong dữ liệu từ link https://www.kaggle.com/datamunge/sign-language-mnist và đưa vào folder data.<br>
Vào trong project, chạy từ terminal bên trong project, hoặc chạy từ trình biên dịch VSCode .<br>
B5: chạy lệnh "pip install -r setup.txt".<br>
B6: Chạy file preprocessing_and_training.py. Sau khi chạy xong, ta được model sau train lưu ở file sign_langguage.<br>
B7: Chạy file real_time.py, sẽ có 1 ảnh hướng dẫn các ký tự thủ ngữ và cửa sổ webcam, bạn làm theo anhrn hướng dẫn và đưa hành động mô tả thủ ngữ vào khung detect bên trong webcam, kết quả nhận diện sẽ hiển thị ra.<br>
B8: Stop file real_time.py để ngừng chương trình.<br>





