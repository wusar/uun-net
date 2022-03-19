from msilib import sequence
from extern_lib import *


from sklearn.metrics import f1_score


def InputBlock(filters, kernel_size=3, strides=1, padding='same'):
    layers = [
        keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                            strides=strides, padding=padding, activation='relu'),
        keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                            strides=strides, padding=padding, activation='relu')
    ]
    return Sequential(layers=layers)


def ContractingPathBlock(filters, kernel_size=3, strides=1, padding='same'):
    layers = [tf.keras.layers.MaxPool2D((2, 2)),
              tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                     activation='relu'),
              tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                     activation='relu')
              ]
    return Sequential(layers=layers)

# 扩张（恢复）路径模块


class ExpansivePathBlock(keras.Model):
    def __init__(self, filters, tran_filters, kernel_size=3, tran_kernel_size=2, strides=1,
                 tran_strides=2, padding='same', tran_padding='same'):
        super(ExpansivePathBlock, self).__init__()
        self.convtrans = keras.layers.Conv2DTranspose(filters=tran_filters, kernel_size=tran_kernel_size,
                                                      strides=tran_strides, padding=tran_padding)
        self.conv1 = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                         activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                            activation='relu')

    def call(self, inputs):
        input, con_feature = inputs
        upsampling = self.convtrans(input)
        con_feature = tf.image.resize(con_feature, ((upsampling.shape)[1], (upsampling.shape)[2]),
                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        concat_feature = tf.concat([con_feature, upsampling], axis=3)
        conv_1 = self.conv1(concat_feature)
        return self.conv2(conv_1)


class UNet(keras.Model):
    def __init__(self):
        super(UNet, self).__init__()
        self.input_block = InputBlock(64)
        self.con_b1 = ContractingPathBlock(128)
        self.con_b2 = ContractingPathBlock(256)
        self.con_b3 = ContractingPathBlock(512)
        self.con_b4 = ContractingPathBlock(1024)

        # expansive path
        self.exp_b4 = ExpansivePathBlock(512, 512)
        self.exp_b3 = ExpansivePathBlock(256, 256)
        self.exp_b2 = ExpansivePathBlock(128, 128)
        self.exp_b1 = ExpansivePathBlock(64, 64)
        self.out_layer=keras.layers.Conv2D(2, 1)

    def call(self,inputs):
        s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

        # input block
        input_block = self.input_block(s)
        con_1=self.con_b1(input_block)
        con_2=self.con_b2(con_1)
        con_3=self.con_b3(con_2)
        con_4=self.con_b4(con_3)

        exp_4=self.exp_b4((con_4,con_3))
        exp_3=self.exp_b3((exp_4,con_2))
        exp_2=self.exp_b2((exp_3,con_1))
        exp_1=self.exp_b1((exp_2,input_block))
        outputs = self.out_layer(exp_1)
        return outputs

if __name__ == '__main__':
    pass
