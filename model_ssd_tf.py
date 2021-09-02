import tensorflow as tf 
from tensorflow.keras import layers


def cls_predictor(num_inputs, num_anchors, num_classes):
    return layers.Conv2D(num_anchors * (num_classes + 1),
                     kernel_size=3, padding="same")

def bbox_predictor(num_inputs, num_anchors):
    return layers.Conv2d(num_anchors * 4, kernel_size=3, padding="same")


def flatten_pred(pred):
    return tf.reshape(pred, shape=(pred.shape[0], -1))

def concat_preds(preds):
    return tf.concat([flatten_pred(p) for p in preds], axis=1)


class Block_tf(tf.keras.Model):
    """Create Block convlution for Encode. Use layers convlution

    Args:
        torch ([Module pytorch]): [create block conv with class Module of pytorch]
    """
    def __init__(self, mid_channel, out_channels, batch_norm=False):
        super().__init__()
        
        self.conv1 = layers.Conv2d(mid_channel, kernel_size=3, padding='same')
        self.conv2 = layers.Conv2d(out_channels, kernel_size=3, padding='same')
        
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn1 = layers.BatchNormalization()
            self.bn2 = layers.BatchNormalization()
            
    def forward(self, x):
        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = tf.nn.relu(x)
        
        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        out = tf.nn.relu(x)
        return out

    @property
    def _get_conv(self):
        name_layer = ["conv1", "conv2"]
        
        # [getattr(self, i) for i in name_layer]
        
        return [getattr(self, i) for i in name_layer]

class MySSD_tf(tf.keras.Model):
    def down(self, x):
        return layers.MaxPool2D(2)
    def __init__(self, num_classes=None, batch_norm=False, upscale_mode="nearest"):
        super().__init__()

        self.batch_norm = batch_norm
        self.upscale_mode = upscale_mode

        self.sizes = [[0.2, 0.272], [0.37, 0.447], 
                        [0.54, 0.619], [0.71, 0.79], 
                        [0.88, 0.961]]
        self.ratios = [[1, 2, 0.5]] * 5
        self.num_anchors = [len(size) + len(ratio) - 1 for size, ratio in zip(self.sizes, self.ratios)]

        self.auxiliaryconvs = AuxiliaryConvolutions_tf()
        
        # encoder
        self.enc1 = Block_tf(16, 16, batch_norm)
        self.enc2 = Block_tf(32, 32, batch_norm)
        self.enc3 = Block_tf(64, 64, batch_norm)
        self.enc4 = Block_tf(128, 128, batch_norm)
        
        self.num_classes = num_classes
        idx_to_in_channels = [128, 128, 128, 128, 128]
        for i in range(len(idx_to_in_channels)):
            setattr(self, f'cls_{i}', cls_predictor(self.num_anchors[i],
                              self.num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(self.num_anchors[i]))
        


    def forward(self, x):
        self.anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5

        enc1 = self.enc1(x) # 300x300
        enc2 = self.enc2(self.down(enc1)) # 150x150
        enc3 = self.enc3(self.down(enc2)) # 75x75
        enc4 = self.enc4(self.down(enc3)) # 37x37

        enc5, enc6, enc7, enc8 = self.auxiliaryconvs(enc4)

        for i, enc in enumerate([env4, enc5, enc6, enc7, enc8]):

            anchor = multibox_prior(enc, self.sizes[i], self.ratios[i])

            cls_pred = getattr(self, f'cls_{i}')(enc)

            bbox_pred = getattr(self, f'bbox_{i}')(enc)
            self.anchors[i] = anchor
            cls_preds[i] = cls_pred
            bbox_preds[i] = bbox_pred

        self.anchors = tf.concat(self.anchors, dims=1)

        cls_preds = concat_preds(cls_preds)

        cls_preds = tf.reshape(cls_preds, (cls_preds.shape[0], -1,
                                      self.num_classes + 1))
        bbox_preds = concat_preds(bbox_preds)

        assert cls_preds.shape[2] == self.num_classes + 1
        assert self.anchors.shape[1] == cls_preds.shape[1] == int(bbox_preds.shape[1]/4)
        return [self.anchors, cls_preds, bbox_preds]

    @property
    def _get_conv(self):
        name_layer = ["enc1", "enc2", "enc3", "enc4"]
        return [getattr(self, i) for i in name_layer]

    @property
    def _get_cls(self):
        name_layer = ["cls_1", "cls_2", "cls_3", "cls_4"]
        return [getattr(self, i) for i in name_layer]

    @property
    def _get_auxiliaryconv(self):
        name_layer = ["bbox_1", "bbox_2", "bbox_3", "bbox_4"]
        return [getattr(self, i) for i in name_layer]

    @property
    def _get_bbox(self):
        name_layer = ["auxiliaryconvs"]
        return [getattr(self, i) for i in name_layer]


class AuxiliaryConvolutions_tf(tf.keras.Model):
    """
    Additional convolutions to produce higher-level feature maps.
    """

    def __init__(self):
        super(AuxiliaryConvolutions_tf, self).__init__()

        # Auxiliary/additional convolutions on top of the VGG base
        self.conv1_1 = layers.Conv2D(128, kernel_size=1)  # stride = 1, by default
        self.conv1_2 = layers.Conv2D(128, kernel_size=1, stride=2)  # dim. reduction because stride > 1

        self.conv2_1 = layers.Conv2D(128, kernel_size=1,)
        self.conv2_2 = layers.Conv2D(128, kernel_size=3, stride=2)  # dim. reduction because stride > 1

        self.conv3_1 = layers.Conv2D(128, kernel_size=1, )
        self.conv3_2 = layers.Conv2D(128, kernel_size=3, stride=2)  # dim. reduction because padding = 0

        self.conv4_1 = layers.Conv2D(128, kernel_size=1)
        self.conv4_2 = layers.Conv2D(128, kernel_size=3)  # dim. reduction because padding = 0

    def forward(self, conv4):
        """
        Forward propagation.

        :param conv7_feats: lower-level conv7 feature map, a tensor of dimensions (N, 1024, 19, 19)
        :return: higher-level feature maps conv8_2, conv9_2, conv10_2, and conv11_2
        """
        # print(conv7_feats.shape)
        out = tf.nn.relu(self.conv1_1(conv4))  # (N, 128, 37, 37)
        out = tf.nn.relu(self.conv1_2(out))  # (N, 128, 18, 18)
        # print(out.shape)
        conv1_2_feats = out  # (N, 128, 10, 10)
        

        out = tf.nn.relu(self.conv2_1(out))  # (N, 128, 18, 18)
        out = tf.nn.relu(self.conv2_2(out))  # (N, 128, 9, 9)
        # print(out.shape)
        conv2_2_feats = out  # (N, 128, 5, 5)

        out = F.relu(self.conv3_1(out))  # (N, 128, 9, 9)
        out = F.relu(self.conv3_2(out))  # (N, 128, 4, 4)
        # print(out.shape)
        conv3_2_feats = out  # (N, 128, 3, 3)

        out = F.relu(self.conv4_1(out))  # (N, 128, 4, 4)
        conv4_2_feats = F.relu(self.conv4_2(out))  # (N, 128, 2, 2)
        # print(conv11_2_feats.shape)

        # Higher-level feature maps
        return conv1_2_feats, conv2_2_feats, conv3_2_feats, conv4_2_feats

    @property
    def _get_bbox(self):
        name_layer = ["auxiliaryconvs"]
        return [getattr(self, i) for i in name_layer]

    @property
    def _get_bbox(self):
        name_layer = ["conv1_1", "conv1_2", "conv2_1", "conv2_2", 
                        "conv3_1", "conv3_2", "conv4_1", "conv4_2"]
        return [getattr(self, i) for i in name_layer]


if __name__ == '__main__':
    # main()
    mymodel = MySSD_tf(num_classes=1)

    # torch.save(mymodel, "model_ssd.pth")

    # torch.load("model_ssd.pth")

    example = torch.rand(1, 256, 256, 3)

    print(mymodel(example)[0].shape)
    # traced_script_module = torch.jit.trace(mymodel, example)


    # traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    # traced_script_module_optimized._save_for_lite_interpreter("model_ssd_v2.ptl")
    

