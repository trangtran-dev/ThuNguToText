

# thư viện xử lý dữ liệu dạng dataframe
import pandas as pd
# thư viện xử lý dữ liệu dạng mảng, ma trận
import numpy as np
from sklearn.preprocessing import LabelBinarizer
# thư viện xử lý đồ họa
import matplotlib.pyplot as plt
# thư viện hỗ trợ viết các layer trong mạng deep learning và xử lý dữ liệu đầu vào của mạng
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.preprocessing.image import img_to_array
# thư viện opencv xử lý ảnh
import cv2


# đọc dữ liệu train và test thành 2 dataframe train và test
df_train=pd.read_csv('data/sign_mnist_train/sign_mnist_train.csv')
df_test=pd.read_csv('data/sign_mnist_test/sign_mnist_test.csv')

# danh mục nhãn đầu ra: 24 ký tự Alphabet
y_train=df_train['label'].values
y_test=df_test['label'].values

# lược bỏ cột label trong dữ liệu train và test 
df_train.drop('label',axis=1,inplace=True)
df_test.drop('label',axis=1,inplace=True)
df_test.head()

# xử lý dữ liệu đầu vào để train model để chuyển mỗi điểm dữ liệu đầu vào(ứng với mỗi ảnh) dạng matrix 28*28
x_train=df_train.values
x_test=df_test.values
unique_val = np.array(y_train)
np.unique(unique_val)
x_train=np.array(x_train.reshape(-1,28,28,1))
x_test=np.array(x_test.reshape(-1,28,28,1))
print(x_train.shape)
print(x_test.shape)

# chuyển đổi label Aphabet sang label dạng số
lb_train= LabelBinarizer()
lb_test=LabelBinarizer()
y_train=lb_train.fit_transform(y_train)
y_test=lb_test.fit_transform(y_test)
print(y_train)
print(y_test)
plt.imshow(x_train[10].reshape(28,28),cmap='gray')

# chuẩn hóa dữ liệu train và test
x_train=x_train/255
x_test=x_test/255


# số lần chạy model
batch_size = 128
num_classes = 24
epochs = 100

# add các layer sử dụng trong model
model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu', input_shape=(28, 28 ,1) ))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.20))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.20))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.20))
model.add(Dense(num_classes, activation = 'softmax'))


# hàm mất mát sử dụng để train model
model.compile(loss = keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# train model với dữ liệu đầu vào
history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=epochs, batch_size=batch_size)


# hiển thị kết quả độ chính xác của model đã train
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title("Accuracy")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train','test'])
plt.show()


# save model 
model.save('sign_language')


# phân loại ảnh đầu vào size 28*28 sang 24 ký tự
alphabet=['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
def classify(image):
    image = cv2.resize(image, (28, 28))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    proba=model.predict(image)
    idx = np.argmax(proba)
    return alphabet[idx]


# hiển thị kết quả phân loại với ảnh thứ 4 trong tập dữ liệu train 
classify(x_train[3]*255)
print(y_train[3])





