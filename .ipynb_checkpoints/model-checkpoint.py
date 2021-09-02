# import tensorflow
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras import layers 
from tqdm import tqdm





# import nump

cls_loss = keras.losses.SparseCategoricalCrossentropy()
bbox_loss = keras.losses.MeanSquaredError()

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    cls = cls_loss(cls_preds, cls_labels)
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks)
    return cls + bbox


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



def cls_predictor(num_anchors, num_classes):
    return layers.Conv2D(num_anchors * (num_classes + 1), kernel_size=3,
                     padding=1)

def bbox_predictor(num_anchors):
    return layers.Conv2D(num_anchors * 4, kernel_size=3, padding=1)




class SSD(keras.Model):

    def down(self, x):
        return layers.MaxPool2D()(x)
    
    def __init__(self):
        super().__init__()

        # self.my_input = keras.Input(shape=(512,512,3))
        
        self.env1 = Block(16,16)
        self.env2 = Block(16, 32)
        self.env3 = Block(32, 64)
        self.env4 = Block(64, 128)
        
        self.sizes = [[0.2, 0.272], [0.37, 0.447], 
                        [0.54, 0.619], [0.71, 0.79], 
                        [0.88, 0.961]]
        self.ratios = [[1, 2, 0.5]] * 5
        self.num_anchors = [len(size) + len(ratio) - 1 for size, ratio in zip(self.sizes, self.ratios)]
        
        self.auxiliaryconvs = AuxiliaryConvolutions()
        
        self.cls_1 = cls_predictor(self.num_anchors[0], self.num_anchors[0], self.num_classes)
        self.bbox_1 = bbox_predictor(self.num_anchors[0],self.num_classes)
        
        self.cls_2 = cls_predictor(self.num_anchors[1], self.num_anchors[1], self.num_classes)
        self.bbox_2 = bbox_predictor(self.num_anchors[1],self.num_classes)
        
        self.cls_3 = cls_predictor(self.num_anchors[2], self.num_anchors[2], self.num_classes)
        self.bbox_3 = bbox_predictor(self.num_anchors[2],self.num_classes)
        
        self.cls_4 = cls_predictor(self.num_anchors[3], self.num_anchors[3], self.num_classes)
        self.bbox_4 = bbox_predictor(self.num_anchors[3],self.num_classes)
        
        self.cls_5 = cls_predictor(self.num_anchors[4], self.num_anchors[4], self.num_classes)
        self.bbox_5 = bbox_predictor(self.num_anchors[4],self.num_classes)
        
        
    
    def compile(self, optimizer,
                metrics, 
                # cls_loss_fn,
                # box_loss_fn,
                calc_loss):
        """[summary]

        Args:
            optimizer ([type]): [description]
            metrics ([type]): [description]
            cls_loss_fn ([type]): [description]
            box_loss_fn ([type]): [description]
        """
        self.optimizer = optimizer
        self.metrics = metrics
        # self.cls_loss_fn = cls_loss_fn
        # self.box_loss_fn = box_loss_fn
        self.calc_loss = calc_loss
        
    def train_step(self, data):
        
        x, y = data 
        
        with tf.GradientTape as tape:
            anchors, cls_preds, bbox_preds = self(X) 
            bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, y)
            loss = self.box_loss_fn
        
        trainable_vars = self.trainable_variables
        gradient = tape.gradient(loss, trainable_vars)
        
        self.optimizer.apply_gradients(zip(gradient, trainable_vars))
        self.compiled_metrics.update_state(y, prediction)
        
        results.update(
          {"loss": loss,}
        )
        return results


    def call(self, X):
        
        # X = self.my_input(X)

        X = self.env1(X) # 512x512x16
        X = self.down(self.env2(X)) # 200x200x32
        X = self.down(self.env3(X)) # 100x100x64
        X = self.down(self.env4(X)) # 50x50x128
        
        X = self.down(self.env4(X))
        
        

        return X 
    

def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # The Adam optimization algorithm is used here
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    # net.compile(loss=loss, optimizer=optimizer)
    for epoch in range(num_epochs):
        for X, y in tqdm(train_iter):
            with tf.GradientTape() as tape:
                y_hat = net(X)
                l = loss(y, y_hat)
            params = net.trainable_variables
            grads = tape.gradient(l, params)
            optimizer.apply_gradients(zip(grads, params))
        train_ls.append(log_rmse(train_labels, net(train_features)))
        if test_labels is not None:
            test_ls.append(log_rmse(test_labels, net(test_features)))
    return train_ls, test_ls


class AuxiliaryConvolutions(nn.Module):
    """
    Additional convolutions to produce higher-level feature maps.
    """

    def __init__(self):
        super(AuxiliaryConvolutions, self).__init__()

        # Auxiliary/additional convolutions on top of the VGG base
        self.conv8_1 = nn.Conv2d(128, 128, kernel_size=1, padding=0)  # stride = 1, by default
        self.conv8_2 = nn.Conv2d(128, 128, kernel_size=1, stride=2, padding=1)  # dim. reduction because stride > 1

        self.conv9_1 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.conv9_2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0)  # dim. reduction because stride > 1

        self.conv10_1 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.conv10_2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0)  # dim. reduction because padding = 0

        self.conv11_1 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.conv11_2 = nn.Conv2d(128, 128, kernel_size=3, padding=0)  # dim. reduction because padding = 0

    def forward(self, conv4):
        """
        Forward propagation.

        :param conv7_feats: lower-level conv7 feature map, a tensor of dimensions (N, 1024, 19, 19)
        :return: higher-level feature maps conv8_2, conv9_2, conv10_2, and conv11_2
        """
        # print(conv7_feats.shape)
        out = F.relu(self.conv8_1(conv4))  # (N, 128, 37, 37)
        out = F.relu(self.conv8_2(out))  # (N, 128, 18, 18)
        # print(out.shape)
        conv8_2_feats = out  # (N, 128, 10, 10)
        

        out = F.relu(self.conv9_1(out))  # (N, 128, 18, 18)
        out = F.relu(self.conv9_2(out))  # (N, 128, 9, 9)
        # print(out.shape)
        conv9_2_feats = out  # (N, 128, 5, 5)

        out = F.relu(self.conv10_1(out))  # (N, 128, 9, 9)
        out = F.relu(self.conv10_2(out))  # (N, 128, 4, 4)
        # print(out.shape)
        conv10_2_feats = out  # (N, 128, 3, 3)

        out = F.relu(self.conv11_1(out))  # (N, 128, 4, 4)
        conv11_2_feats = F.relu(self.conv11_2(out))  # (N, 128, 2, 2)
        # print(conv11_2_feats.shape)

        # Higher-level feature maps
        return conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats


# if __name__ == "__main__":
    
#     model = SSD()
#     inputs = keras.Input(shape=(512,512,3))
#     input_shape = inputs.shape
#     print(inputs.shape)
#     model.build(input_shape)
    
    # model.compile(
    #     optimizer = keras.optimizers.Adam(), 
    #     metrics = [keras.metrics.SparseCategoricalAccuracy()], 
    #     calc_loss=calc_loss,
        
    # )
    
    
    
    
    # print(model.summary())
