import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import random
import os

# --- Utility Functions ---
def _set_seed(seed: int):
    """Set global random seeds for reproducibility"""
    tf.keras.mixed_precision.set_global_policy('float32')
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ['PYTHONHASHSEED'] = f"{seed}"
    tf.random.set_seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

def _residual_block(x, filters: int, stride: int, seed: int):

    shortcut = x

    y = layers.Conv2D(
        filters,
        3,
        strides=stride,
        padding="same",
        use_bias=False,
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
    )(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)

    y = layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed + 1),
    )(y)
    y = layers.BatchNormalization()(y)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters,
            1,
            strides=stride,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed + 2),
        )(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    y = layers.Add()([y, shortcut])
    return layers.Activation("relu")(y)


def build_tabtransformer_model(categorical_columns, numerical_columns, num_categories, embed_dim=32, num_heads=4, ff_dim=64, transformer_blocks=2, activation="softmax", num_classes=7, seed=24):
    
    # TabTransformer following: https://keras.io/examples/structured_data/tabtransformer/

    # Set the seeds for reproducibility
    keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Embedding layers for categorical features
    inputs = []
    cat_embeds = []
    for i, col in enumerate(categorical_columns):
        input_cat = layers.Input(shape=(1,), name=f'{col}_input')
        embed = layers.Embedding(
            input_dim=num_categories[col],
            output_dim=embed_dim,
            embeddings_initializer=keras.initializers.GlorotUniform(seed=seed)
        )(input_cat)
        cat_embeds.append(embed)
        inputs.append(input_cat)
    cat_stack = layers.Concatenate(axis=1)(cat_embeds)

    # Transformer blocks
    x = cat_stack
    for block_idx in range(transformer_blocks):
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim,
            kernel_initializer=keras.initializers.GlorotUniform(seed=seed),
            bias_initializer=keras.initializers.Zeros(),
            seed=seed
        )(x, x)
        attn_output = layers.Dropout(0.1, seed=seed)(attn_output)
        out1 = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
        ffn_output = layers.Dense(
            ff_dim, 
            activation='relu',
            kernel_initializer=keras.initializers.GlorotUniform(seed=seed)
        )(out1)
        ffn_output = layers.Dense(
            embed_dim,
            kernel_initializer=keras.initializers.GlorotUniform(seed=seed)
        )(ffn_output)
        ffn_output = layers.Dropout(0.1, seed=seed)(ffn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
    x = layers.Flatten()(x)

    # Numerical features / Continuous features
    input_num = layers.Input(shape=(len(numerical_columns),), name='num_input')
    inputs.append(input_num)
    input_num = layers.LayerNormalization()(input_num)

    # Combine categorical and numerical features
    x = layers.Concatenate()([x, input_num])

    # Final MLP
    x = layers.Dense(
        128, 
        activation='relu'
    )(x)
    x = layers.Dropout(0.2, seed=seed)(x)
    x = layers.Dense(
        64, 
        activation='relu',
    )(x)
    x = layers.Dropout(0.2, seed=seed)(x)
    output = layers.Dense(
        num_classes,
        activation=activation
    )(x)
    model = keras.Model(inputs=inputs, outputs=output, seed=seed)
    return model

def _build_cifar_resnet(num_classes: int, n: int, name: str, seed: int):
    # Build a ResNet32 dataset for CIFAR-10, CIFAR-100 or CINIC-10

    _set_seed(seed=seed)
    inputs = layers.Input(shape=(32, 32, 3))

    x = layers.Conv2D(
        16,
        3,
        padding="same",
        use_bias=False,
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    block_seed = seed + 3
    for filters, first_stride in ((16, 1), (32, 2), (64, 2)):
        for block in range(n):
            x = _residual_block(x, filters, first_stride if block == 0 else 1, block_seed)
            block_seed += 3

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=block_seed),
    )(x)

    return models.Model(inputs, outputs, name=name)

class MLModel():
    @staticmethod
    def get_model():
        raise NotImplementedError("Subclasses must implement get_model()")

# --- Image Models ---
class CIFAR10(MLModel):
    # Followin: https://www.tensorflow.org/tutorials/images/cnn
    @staticmethod
    def get_model():
        _set_seed(24)
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        return model


class CIFAR100(MLModel):
    
    @staticmethod
    def get_model(seed:int):
       
        _set_seed(seed=seed)
        
        model = models.Sequential()
 
        model.add(layers.Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Conv2D(32, (3, 3), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.2, seed=24))
 
        model.add(layers.Conv2D(64, (3, 3), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Conv2D(64, (3, 3), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.2, seed=24))
        
        # Dense layers
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(100, activation='softmax'))
        return model

# --- Tabular Models ---
   
class AdultModel():
    
    @staticmethod
    def get_model():
        
        seed = 24
        _set_seed(seed)

        categorical_columns = [
            'workclass', 'education', 'marital-status', 'occupation', 'relationship',
            'race', 'sex', 'native-country'
        ]
        
        num_categories = {'workclass': 9,
                        'education': 16,
                        'marital-status': 7,
                        'occupation': 15,
                        'relationship': 6,
                        'race': 5,
                        'sex': 2,
                        'native-country': 42}

        numerical_columns = ['age',
                            'fnlwgt',
                            'education-num',
                            'capital-gain',
                            'capital-loss',
                            'hours-per-week']
    
        model = build_tabtransformer_model(
                    categorical_columns=categorical_columns,
                    numerical_columns=numerical_columns,
                    num_categories=num_categories,
                    num_classes=1,
                    activation='sigmoid',
                    seed=seed
              )
        
        return model
    
class CoverTypeModel():
    
    @staticmethod
    def get_model():
        
        seed = 24
        _set_seed(seed)

        categorical_columns = ['Wilderness_Area', 'Soil_Type']
        num_categories = {'Wilderness_Area': 4, 'Soil_Type': 40}
        numerical_columns = [
            'Elevation',
            'Aspect',
            'Slope',
            'Horizontal_Distance_To_Hydrology',
            'Vertical_Distance_To_Hydrology',
            'Horizontal_Distance_To_Roadways',
            'Hillshade_9am',
            'Hillshade_Noon',
            'Hillshade_3pm',
            'Horizontal_Distance_To_Fire_Points']

        model = build_tabtransformer_model(
                    categorical_columns=categorical_columns,
                    numerical_columns=numerical_columns,
                    num_categories=num_categories,
                    embed_dim=32,
                    num_heads=4,
                    ff_dim=32,
                    num_classes=7,
                    transformer_blocks=2,
                    seed=seed
                )
        
        return model

# --- Time Series Models ---

class PVODModel():
    
    @staticmethod
    def get_model():
        seed = 24
        _set_seed(seed)
        
        model = models.Sequential([
            layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(96, 13), kernel_initializer=tf.keras.initializers.glorot_uniform(seed=24), padding="causal"),
            layers.Conv1D(32, kernel_size=3, activation='relu', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=24), padding="causal", dilation_rate=2),
            layers.GRU(64),
            layers.Flatten(),
            layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=24)),
            layers.Dense(1, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=24))
        ])

        return model
    
class GermanSolarFarm():

    @staticmethod
    def get_model():
    
        seed = 24
        _set_seed(seed)
        
        model = tf.keras.Sequential(name='lstm_regression_model')
        model.add(layers.LSTM(
            32,
            activation='tanh',
            recurrent_activation='sigmoid',
            dropout=0.25,
            input_shape=(16, 72),
        ))
    
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.25))
        model.add(layers.Dense(1, activation='linear'))
        
        return model
    
class GermanSolarFarmCNN():
    
    @staticmethod
    def get_model():
        seed = 24
        _set_seed(seed)
        
        model = models.Sequential([
            layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(16, 72), kernel_initializer=tf.keras.initializers.glorot_uniform(seed=24), padding="causal"),
            layers.Conv1D(32, kernel_size=3, activation='relu', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=24), padding="causal", dilation_rate=2),
            layers.Conv1D(32, kernel_size=3, activation='relu', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=24), padding="causal", dilation_rate=4),
            layers.Conv1D(32, kernel_size=3, activation='relu', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=24), padding="causal", dilation_rate=8),
            layers.GRU(64),
            layers.Flatten(),
            layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=24)),
            layers.Dense(1, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=24))
        ])
        
        return model