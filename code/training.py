import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from dataloader import get_generator  

train_paths = [
     '/vols/cms/yl13923/masterproject/data_chunks/train_chunk_1.pkl',
     '/vols/cms/yl13923/masterproject/data_chunks/train_chunk_2.pkl',
     '/vols/cms/yl13923/masterproject/data_chunks/train_chunk_3.pkl',
     '/vols/cms/yl13923/masterproject/data_chunks/train_chunk_4.pkl',
     '/vols/cms/yl13923/masterproject/data_chunks/train_chunk_5.pkl'
]
               
val_paths = [
     '/vols/cms/yl13923/masterproject/data_chunks/val_chunk_1.pkl',
     '/vols/cms/yl13923/masterproject/data_chunks/val_chunk_2.pkl',
     '/vols/cms/yl13923/masterproject/data_chunks/val_chunk_3.pkl',
     '/vols/cms/yl13923/masterproject/data_chunks/val_chunk_4.pkl',
     '/vols/cms/yl13923/masterproject/data_chunks/val_chunk_5.pkl'
]

data_train = tf.data.Dataset.from_generator(
    get_generator(train_paths),
    output_signature=(
        tf.TensorSpec(shape=(2,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )
).batch(150)

data_val = tf.data.Dataset.from_generator(
    get_generator(val_paths),
    output_signature=(
        tf.TensorSpec(shape=(2,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )
).batch(150)

def simple_model(input_dimension):
    model = Sequential()
    model.add(Dense(input_dimension, input_dim=input_dimension, kernel_initializer='normal', activation='relu'))
    model.add(Dense(input_dimension*2, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation="sigmoid"))

    optimizer = Adam(learning_rate=0.01)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model

model = simple_model(2)

early_stop = EarlyStopping(monitor='val_loss', patience=3)

history = model.fit(
    data_train, 
    validation_data=data_val,
    epochs=10, 
    callbacks=[early_stop]
)

# save model
model_save_path = '/vols/cms/yl13923/masterproject/my_model.h5'
model.save(model_save_path)

history_save_path = '/vols/cms/yl13923/masterproject/train_history.json'
with open(history_save_path, 'w') as f:
    json.dump(history.history, f)
