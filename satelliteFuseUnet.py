

TF = 1

if TF:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        BatchNormalization, Conv2D, Conv2DTranspose,
        MaxPooling2D, UpSampling2D, Input,
        concatenate, Add
    )
else:
    from keras.models import Model
    from keras.layers import (
        BatchNormalization, Conv2D, Conv2DTranspose,
        MaxPooling2D, UpSampling2D, Input,
        concatenate
    )


def bn_conv_relu(input, filters, bachnorm_momentum, **conv2d_args):
    x = BatchNormalization(momentum=bachnorm_momentum)(input)
    x = Conv2D(filters, **conv2d_args)(x)
    return x


def bn_upconv_relu(input, filters, bachnorm_momentum, **conv2d_trans_args):
    x = BatchNormalization(momentum=bachnorm_momentum)(input)
    x = Conv2DTranspose(filters, **conv2d_trans_args)(x)
    return x


def satellite_unet(input_shapes2,input_shapes1, num_classes=1,output_activation='sigmoid',num_layers=4,sum=1):
    inputs1 = Input(input_shapes1)
    inputs2 = Input(input_shapes2)



    filters = 64
    upconv_filters = 96

    kernel_size = (3, 3)
    activation = 'relu'
    strides = (1, 1)
    padding = 'same'
    kernel_initializer = 'he_normal'

    conv2d_args = {
        'kernel_size': kernel_size,
        'activation': activation,
        'strides': strides,
        'padding': padding,
        'kernel_initializer': kernel_initializer
    }

    conv2d_trans_args = {
        'kernel_size': kernel_size,
        'activation': activation,
        'strides': (2, 2),
        'padding': padding,
        'output_padding': (1, 1)
    }

    bachnorm_momentum = 0.01

    pool_size = (2, 2)
    pool_strides = (2, 2)
    pool_padding = 'valid'

    maxpool2d_args = {
        'pool_size': pool_size,
        'strides': pool_strides,
        'padding': pool_padding,
    }

    x1 = Conv2D(filters, **conv2d_args)(inputs1)
    c1_1 = bn_conv_relu(x1, filters, bachnorm_momentum, **conv2d_args)
    x1 = bn_conv_relu(c1_1, filters, bachnorm_momentum, **conv2d_args)
    x1 = MaxPooling2D(**maxpool2d_args)(x1)


    x2 = Conv2D(filters, **conv2d_args)(inputs2)
    c1_2 = bn_conv_relu(x2, filters, bachnorm_momentum, **conv2d_args)
    x2 = bn_conv_relu(c1_2, filters, bachnorm_momentum, **conv2d_args)
    x2 = MaxPooling2D(**maxpool2d_args)(x2)
    if sum:
        c1=Add()([c1_1, c1_2])
    else:
        c1= concatenate([c1_1, c1_2])
        c1=Conv2D(filters, **conv2d_args)(c1)


    down_layers = []

    for l in range(num_layers):
        x1 = bn_conv_relu(x1, filters, bachnorm_momentum, **conv2d_args)
        x1 = bn_conv_relu(x1, filters, bachnorm_momentum, **conv2d_args)
        x2 = bn_conv_relu(x2, filters, bachnorm_momentum, **conv2d_args)
        x2 = bn_conv_relu(x2, filters, bachnorm_momentum, **conv2d_args)
        if sum:
            x=Add()([x1, x2])
        else:
            x = concatenate([x1, x2])
            x = Conv2D(filters, **conv2d_args)(x)
        down_layers.append(x)
        x1 = bn_conv_relu(x1, filters, bachnorm_momentum, **conv2d_args)
        x1 = MaxPooling2D(**maxpool2d_args)(x1)
        x2 = bn_conv_relu(x2, filters, bachnorm_momentum, **conv2d_args)
        x2 = MaxPooling2D(**maxpool2d_args)(x2)

    x1 = bn_conv_relu(x1, filters, bachnorm_momentum, **conv2d_args)
    x1 = bn_conv_relu(x1, filters, bachnorm_momentum, **conv2d_args)

    x2 = bn_conv_relu(x2, filters, bachnorm_momentum, **conv2d_args)
    x2 = bn_conv_relu(x2, filters, bachnorm_momentum, **conv2d_args)
    if sum:
        x = Add()([x1, x2])
    else:
        x = concatenate([x1, x2])
        x = Conv2D(filters, **conv2d_args)(x)
    x = bn_upconv_relu(x, filters, bachnorm_momentum, **conv2d_trans_args)

    for conv in reversed(down_layers):

        x = concatenate([x, conv])
        x = bn_conv_relu(x, upconv_filters, bachnorm_momentum, **conv2d_args)
        x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
        x = bn_upconv_relu(x, filters, bachnorm_momentum, **conv2d_trans_args)


    x = concatenate([x, c1])
    x = bn_conv_relu(x, upconv_filters, bachnorm_momentum, **conv2d_args)
    x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)

    outputs = Conv2D(num_classes, kernel_size=(1, 1), strides=(1, 1), activation=output_activation, padding='valid')(x)

    model = Model(inputs=[inputs2, inputs1], outputs=[outputs])
    return model




def encoder (inputs, num_layers=4, attention=0):
    filters = 64
    upconv_filters = 96

    kernel_size = (3, 3)
    activation = 'relu'
    strides = (1, 1)
    padding = 'same'
    kernel_initializer = 'he_normal'

    conv2d_args = {
        'kernel_size': kernel_size,
        'activation': activation,
        'strides': strides,
        'padding': padding,
        'kernel_initializer': kernel_initializer
    }

    bachnorm_momentum = 0.01

    pool_size = (2, 2)
    pool_strides = (2, 2)
    pool_padding = 'valid'

    maxpool2d_args = {
        'pool_size': pool_size,
        'strides': pool_strides,
        'padding': pool_padding,
    }

    x = Conv2D(filters, **conv2d_args)(inputs)
    c1 = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
    x = bn_conv_relu(c1, filters, bachnorm_momentum, **conv2d_args)
    x = MaxPooling2D(**maxpool2d_args)(x)

    down_layers = []

    for l in range(num_layers):
        x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
        x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
        down_layers.append(x)
        x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
        x = MaxPooling2D(**maxpool2d_args)(x)

    x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
    x = bn_conv_relu(x, filters, bachnorm_momentum, **conv2d_args)
    return x





