import tensorflow.keras as keras 
import tensorflow as tf 
from tensorflow.keras import layers 


class Block(keras.Model):

    
    
    def __init__(self, out_dim_1, out_dim_2):
        super().__init__()
        self.net = keras.Sequential(
            [
                layers.Conv2D(out_dim_1, (3,3), padding="same"),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Conv2D(out_dim_2, (3,3), padding="same"),
                layers.BatchNormalization(),
                layers.ReLU(),   
            ]
        )
    
    def call(self, X):
        return self.net(X)
        


class SSD(keras.Model):

    def down(self, x):
        return layers.MaxPool2D()(x)
    
    def __init__(self):
        super().__init__()

        self.my_input = keras.Input(shape=(512,512,3))
        
        self.env1 = Block(16,16)
        self.env2 = Block(16, 32)
        self.env3 = Block(32, 64)
        self.env4 = Block(64, 128)

    def call(self, X):
        
        X = self.my_input(X)

        X = self.env1(X)
        X = self.env2(X)
        X = self.env3(X)
        X = self.env4(X)

        return X 


if __name__ == "__main__":
    
    model = SSD()
    print(model.summary())