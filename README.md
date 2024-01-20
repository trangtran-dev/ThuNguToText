# Bài toán
Nhận diện thủ ngữ - ký hiệu tay
Đầu vào: dữ liệu thủ ngữ từ webcam
Đầu ra: chuyển thể thủ ngữ sang ký hiệu chữ viết và hiển thị ra màn hình

### Các thư mục và file trong project
preprocessing_and_training.py dùng để xử lý dữ liệu training model và thực hiện training model
sign_langguage: model sau khi đã train được lưu tại đây
real_time.py dùng để nhận dữ liệu thực tế đầu vào từ webcam, đưa và model đã lưu để nhận dạng và hiển thị dự đoán cho người dùng
data/...: Folder chứa dưa liệu để train và test model


### Cách giải quyết bài toán
Dữ liệu từ webcam sẽ được detect thành các ảnh liên tiếp để chuyển thể video sang dạng ảnh
Sử dụng deep learning model phân loại ảnh thủ ngữ đầu vào thành 24 ký tự Alphabel
Data ảnh train model: https://www.kaggle.com/datamunge/sign-language-mnist

### Mô tả dữ liệu đầu vào train model
Các ảnh có size 28x28 pixel = 784 pixel, giá trị của từng pixel tương ứng là các cột
Có 24 phân loại Aphabet tương ứng là các hàng
Tập train có 27455 ảnh => dữ liệu train lưu dưới dạng csv có 27455 hàng x 784 cột
Tập test có 7172 ảnh => dữ liệu test lưu dạng csv có 7172 hàng x 784 cột

### Mô tả các layer trong mạng neuron network
CONV2D->RELU->MAXPOOLING->CONV2D->RELU->MAXPOOLING->DROPOUT->CONV2D->RELU->MAXPOOLING->DROPOUT->FLATTEN->DENSE->DROPOUT-> DENSE->SOFTMAX


### Độ chính xác dữ liệu train = 99.64% 
### Độ chính xác dữ liệu test    = 97.02%
