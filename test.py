from extern_lib import *
from load_data import *
from UNet import *

if __name__ == '__main__':
    X_test=load_test()
    model = UNet()
    batch_size=1
    model.build(input_shape=(batch_size,IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
    # model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.load_weights('./model.h5')
    plt.figure(figsize=(20,20))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X_test[i], cmap=plt.cm.binary)
    plt.show()
    plt.clf()
    print(X_test.shape)
    print(X_test[0:10].shape)
    
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        out=tf.reduce_max(model(X_test[i:i+1]),axis=-1).numpy()
        plt.imshow(out[0], cmap=plt.cm.binary)
    plt.show()