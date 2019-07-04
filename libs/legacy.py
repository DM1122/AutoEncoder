# Legacy Functions

def build_autoencoder_funcapi(input_x, input_y):
    '''
    Builds an autoencoder using the Functional api

    Args:
        input_x: input width
        input_y: input height
    '''

    # Create model
    input_layer = keras.layers.Input(shape=(input_x,input_y,))      # [width, height, batch]

    encoder_layer = keras.layers.Flatten()(input_layer)

    encoder_layer = keras.layers.Dense(
        units= autoencoder_bottleneck * encoder_layers_ratio**(autoencoder_layers),
        activation='relu',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        name='Encoder_')(encoder_layer)
    
    encoder_layer = keras.layers.Dense(
        units=autoencoder_bottleneck,
        activation='relu',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        name='Bottleneck')(encoder_layer)

    decoder_layer = keras.layers.Dense(
        units= autoencoder_bottleneck * encoder_layers_ratio**(3),
        activation='relu',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        name='Decoder_')(encoder_layer)
    
    decoder_layer = keras.layers.Dense(
        units=input_x * input_y,
        activation='relu',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        name='Output')(decoder_layer)
    
    decoder_layer = keras.layers.Reshape((input_x,input_y))(decoder_layer)


    # Build autoencoder
    autoencoder = keras.models.Model(inputs=input_layer, outputs=decoder_layer)
    
    # Build encoder
    encoder = keras.models.Model(inputs=input_layer, outputs=encoder_layer)

    # Build decoder
    # create a placeholder for an encoded (32-dimensional) input
    decoder_input = keras.layers.Input(shape=(autoencoder_bottleneck,))
    # retrieve the last layer of the autoencoder model
    decoder_output = autoencoder.layers[-1]
    # create the decoder model
    decoder = keras.models.Model(inputs=decoder_input, outputs=decoder_output(decoder_input))


    # input_layer_encoded = keras.layers.Input(shape=(autoencoder_bottleneck,))
    # decoder = keras.models.Model(inputs=input_layer_encoded, outputs=autoencoder.layers[-1](input_layer_encoded))

    print(autoencoder.layers)
    print(encoder.layers)
    print(decoder.layers)

    return autoencoder, encoder, decoder

def build_autoencoder_funcapi_seg(input_x, input_y):
    '''
    Builds an encoder and decoder separately, then returns an autoencoder model using the functional api

    Args:
        input_x: input width
        input_y: input height
    '''


    # Encoder
    encoder_input = keras.layers.Input(shape=(input_x,input_y,))      # [width, height, batch]

    encoder_layer = keras.layers.Flatten()(encoder_input)

    encoder_layer = keras.layers.Dense(
        units= autoencoder_bottleneck * autoencoder_ratio**(autoencoder_layers),
        activation='relu',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        name='Encoder_')(encoder_layer)

    encoder_layer = keras.layers.Dense(
        units=autoencoder_bottleneck,
        activation='relu',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        name='Bottleneck')(encoder_layer)

    encoder = keras.models.Model(inputs=encoder_input, outputs=encoder_layer)


    # Decoder
    decoder_input = keras.layers.Input(shape=(autoencoder_bottleneck,))

    decoder_layer = keras.layers.Dense(
        units= autoencoder_bottleneck * autoencoder_ratio**(3),
        activation='relu',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        name='Decoder_')(decoder_input)

    decoder_layer = keras.layers.Dense(
        units=input_x * input_y,
        activation='relu',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        name='Output')(decoder_layer)
    
    decoder_layer = keras.layers.Reshape((input_x,input_y))(decoder_layer)

    decoder = keras.models.Model(inputs=decoder_input, outputs=decoder_layer)

    # Autoencoder
    autoencoder = keras.models.Model(inputs=encoder_input, outputs=decoder(encoder(encoder_input)))

    return autoencoder, encoder, decoder


def split_autoencoder(model):
    '''
    Splits an autoencoder model into its encoder and decoder components.

    Args:
        model: compiled keras autoencoder
    '''

    model_encoder = keras.Model(inputs=model.input, outputs=model.get_layer('Bottleneck').output)
    # model_decoder = keras.Model(inputs=keras.Input(shape=model.get_layer('Bottleneck').output.shape), outputs=model.output)


    input_layer_encoded = keras.layers.Input(shape=(model.get_layer('Bottleneck').output.shape[1],))

    # model_decoder = keras.models.Model(inputs=input_layer_encoded, outputs=model.layers[-1](input_layer_encoded))
    model_decoder = keras.models.Model(inputs=input_layer_encoded, outputs=model.layers[-1].output)

    print(model_encoder.layers)
    print(model_decoder.layers)

    return model_encoder, model_decoder

    
    log_dir = './logs/' + os.path.basename(__file__) + '/{0}/'.format(dt.datetime.now().strftime('%Y%m%d-%H%M%S'))
    model_dir = './models/' + os.path.basename(__file__) + '/model.keras'