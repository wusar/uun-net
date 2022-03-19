from extern_lib import *
from load_data import *
from UNet import *
batch_size = 1
np.random.seed = 0
if __name__ == '__main__':
    X_train, Y_train = load_data()
    plot_images(X_train)
    plot_images(Y_train)
    X_test = load_test()
    model = UNet()
    
    model.build(input_shape=(batch_size, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    results = model.fit(X_train, Y_train, batch_size=batch_size, epochs=10)
    model.save_weights('./model.h5')
