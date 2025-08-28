import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
(x_train,y_train), (x_test,y_test)=tf.keras.datasets.cifar10.load_data()
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
print(x_train.size)
x_train=x_train/255.0
x_test=x_test/255.0
x_train=tf.image.resize(x_train,(96,96))
x_test=tf.image.resize(x_test,(96,96))
base_model=MobileNetV2(weights="imagenet", include_top=False, input_shape=(96,96,3))
base_model.trainable=False
model=models.Sequential([base_model, 
                         layers.GlobalAveragePooling2D(), 
                         layers.Dense(10, activation ='softmax')])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history=model.fit(x_train,y_train, epochs=2, batch_size=64, validation_split=0.1, verbose=1)
test_loss, test_acc=model.evaluate(x_test,y_test, verbose=2)
print(f'Test Accuracy: {test_acc:.4f}')
print(f'Test Accuracy: {test_acc*100:.4f}%')
