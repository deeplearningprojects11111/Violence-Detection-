from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def build_model(input_shape):
    base_model = VGG16(input_shape=input_shape, weights="imagenet", include_top=False)
    
    for layer in base_model.layers:
        layer.trainable = False 

    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Better than Flatten for generalization
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)  
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)  

    model = Model(inputs=base_model.input, outputs=output)

    optimizer = Adam(learning_rate=1e-4) 
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model