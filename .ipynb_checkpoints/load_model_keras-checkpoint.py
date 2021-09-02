import tensorflow as tf 


model = tf.keras.models.load_model("kerasModel.h5")


print(model.summary())