# from model_pytorch import MySSD
# import torch
# from torch.autograd import Variable
# import torchvision
# from onnx_tf.backend import prepare
# import onnx
# import tensorflow as tf
# import onnx2keras
# from onnx2keras import onnx_to_keras
import torch 
import torch.nn as nn


model = torch.load("model_ssd_V2.pth")
print(model)

# model = MySSD(in_channels=3, num_classes=1)
# # model = torchvision.models.mobilenet_v2()

# # model = Tiny()

# # torch.save(mymodel, "model_ssd.pth")


# dummy_input = Variable(torch.randn(1, 3, 256, 256))


# torch.onnx.export(model, dummy_input, "model.onnx")

# # model = onnx.load('model.onnx')

# onnx_model = onnx.load('model.onnx')
# k_model = onnx_to_keras(onnx_model, ['input.1'])

# tf.keras.models.save_model(k_model,'kerasModel.h5',overwrite=True,include_optimizer=True)

# # tf_rep = prepare(model)
# # tf_rep.export_graph('mnist_tf')


# # model = tf.keras.models.load_model("model_tf")

# # print(model)

# # print(model.summary())