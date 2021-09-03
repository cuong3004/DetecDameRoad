import torch
import torch.nn as nn 
import torch.nn.functional as F
from loss import calc_loss
from utils import *
from pytorch_lightning import LightningModule
from torch.utils.mobile_optimizer import optimize_for_mobile






def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                     kernel_size=3, padding=1)

def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)


def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)


def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use `is_grad_enabled` to determine whether the current mode is training
    # mode or prediction mode
    if not torch.is_grad_enabled():
        # If it is prediction mode, directly use the mean and variance
        # obtained by moving average
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully-connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(dim=0)
            var = ((X - mean)**2).mean(dim=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of `X`, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean)**2).mean(dim=(0, 2, 3), keepdim=True)
        # In training mode, the current mean and variance are used for the
        # standardization
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean.data, moving_var.data


class BatchNorm(nn.Module):
    # `num_features`: the number of outputs for a fully-connected layer
    # or the number of output channels for a convolutional layer. `num_dims`:
    # 2 for a fully-connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # The variables that are not model parameters are initialized to 0 and 1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # If `X` is not on the main memory, copy `moving_mean` and
        # `moving_var` to the device where `X` is located
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # Save the updated `moving_mean` and `moving_var`
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean, self.moving_var,
            eps=1e-5, momentum=0.9)
        return Y


class Block(torch.nn.Module):
    """Create Block convlution for Encode. Use layers convlution

    Args:
        torch ([Module pytorch]): [create block conv with class Module of pytorch]
    """
    def __init__(self, in_channels, mid_channel, out_channels, batch_norm=False):
        super().__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=mid_channel, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, padding=1)
        
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn1 = BatchNorm(mid_channel, num_dims=4)
            self.bn2 = BatchNorm(out_channels, num_dims=4)
            
    def forward(self, x):
        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = torch.relu(x)
        
        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = torch.relu(x)
        return x

    @property
    def _get_conv(self):
        name_layer = ["conv1", "conv2"]
        
        # [getattr(self, i) for i in name_layer]
        
        return [getattr(self, i) for i in name_layer]

class MySSD(torch.nn.Module):
    """Encode is Blocks conv, Decode is UpSample with mode 'nearest'. Model
        use BatchNorm2d

    Args:
        torch ([Module pytorch]): [create model Unet with class Module of pytorch]
    """
    def up(self, x, size):
        return torch.nn.functional.interpolate(x, size=size, mode=self.upscale_mode)
    
    def down(self, x):
        return torch.nn.MaxPool2d(2)(x)
    
    def __init__(self, in_channels, num_classes=None, batch_norm=False, upscale_mode="nearest"):
        super().__init__()
        
        # config class model
        self.in_channels = in_channels
        # self.out_channels = out_channels
        self.batch_norm = batch_norm
        self.upscale_mode = upscale_mode

        self.sizes = [[0.1, 0.141], [0.2, 0.272], 
                        [0.37, 0.449], [0.54, 0.619], 
                        [0.71, 0.79]]
        self.ratios = [[1, 2, 0.5, 3, 0.33]] * 5
        self.num_anchors = [len(size) + len(ratio) - 1 for size, ratio in zip(self.sizes, self.ratios)]

        self.auxiliaryconvs = AuxiliaryConvolutions()
        
        # encoder
        self.enc1 = Block(in_channels, 16, 16, batch_norm)
        self.enc2 = Block(16, 32, 32, batch_norm)
        self.enc3 = Block(32, 64, 64, batch_norm)
        self.enc4 = Block(64, 128, 128, batch_norm)
        
        self.num_classes = num_classes
        idx_to_in_channels = [128, 128, 128, 128, 128]
        for i in range(len(idx_to_in_channels)):
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i], self.num_anchors[i],
                              self.num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i], self.num_anchors[i]))
        
        self._gradient = None 


    def forward(self, x):
        self.anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5

        enc1 = self.enc1(x) # 300x300
        enc2 = self.enc2(self.down(enc1)) # 150x150
        enc3 = self.enc3(self.down(enc2)) # 75x75
        enc4 = self.enc4(self.down(enc3)) # 37x37

        self._gradient = enc4

        enc5, enc6, enc7, enc8 = self.auxiliaryconvs(enc4)

        for i, enc in enumerate([enc4, enc5, enc6, enc7, enc8]):

            anchor = multibox_prior(enc, self.sizes[i], self.ratios[i])

            cls_pred = getattr(self, f'cls_{i}')(enc)

            bbox_pred = getattr(self, f'bbox_{i}')(enc)
            self.anchors[i] = anchor
            cls_preds[i] = cls_pred
            bbox_preds[i] = bbox_pred

        self.anchors = torch.cat(self.anchors, dim=1)

        cls_preds = concat_preds(cls_preds)

        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1,
                                      self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)

        assert cls_preds.shape[2] == self.num_classes + 1
        assert self.anchors.shape[1] == cls_preds.shape[1] == int(bbox_preds.shape[1]/4)
        return [self.anchors, cls_preds, bbox_preds]

    @property
    def _get_conv_block(self):
        name_layer = ["enc1", "enc2", "enc3", "enc4"]
        return [getattr(self, i) for i in name_layer]

    @property
    def _get_cls(self):
        name_layer = ["cls_0", "cls_1", "cls_2", "cls_3", "cls_4"]
        return [getattr(self, i) for i in name_layer]

    @property
    def _get_bbox(self):
        name_layer = ["bbox_0", "bbox_1", "bbox_2", "bbox_3", "bbox_4"]
        return [getattr(self, i) for i in name_layer]

    @property
    def _get_auxiliaryconv(self):
        name_layer = ["auxiliaryconvs"]
        return [getattr(self, i) for i in name_layer]


class AuxiliaryConvolutions(nn.Module):
    """
    Additional convolutions to produce higher-level feature maps.
    """

    def __init__(self):
        super(AuxiliaryConvolutions, self).__init__()

        # Auxiliary/additional convolutions on top of the VGG base
        self.conv1_1 = nn.Conv2d(128, 128, kernel_size=1, padding=0)  # stride = 1, by default
        self.conv1_2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0)  # dim. reduction because stride > 1

        self.conv2_1 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0)  # dim. reduction because stride > 1

        self.conv3_1 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0)  # dim. reduction because padding = 0

        self.conv4_1 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.conv4_2 = nn.Conv2d(128, 128, kernel_size=3, padding=0)  # dim. reduction because padding = 0

    def forward(self, conv4):

        out = F.relu(self.conv1_1(conv4)) 
        out = F.relu(self.conv1_2(out))  
        # print(out.shape)
        conv1_2_feats = out 

        out = F.relu(self.conv2_1(out)) 
        out = F.relu(self.conv2_2(out)) 

        conv2_2_feats = out 

        out = F.relu(self.conv3_1(out)) 
        out = F.relu(self.conv3_2(out))

        conv3_2_feats = out 

        out = F.relu(self.conv4_1(out))
        conv4_2_feats = F.relu(self.conv4_2(out)) 

        return conv1_2_feats, conv2_2_feats, conv3_2_feats, conv4_2_feats

    @property
    def _get_conv_anxi(self):
        name_layer = ["conv1_1", "conv1_2", "conv2_1", "conv2_2", 
                        "conv3_1", "conv3_2", "conv4_1", "conv4_2"]
        return [getattr(self, i) for i in name_layer]



class LitObjectDetect(LightningModule):
    def __init__(self, backbone, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = backbone

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.backbone(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        anchors, cls_preds, bbox_preds = self.backbone(x)

        bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, y)

        loss = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                      bbox_masks)

        loss = loss.mean()
        # print(loss)

        # calc_loss
        # # y_hat = torch.sigmoid(y_out)
        # loss = F.binary_cross_entropy_with_logits(y_hat, y)
        # # loss = tversky_loss(y_hat, y)

        # acc = dice_score(y_hat, y)
        self.log('train_loss', loss.item())
        # self.log('train_acc', acc.item())
        return {'loss': loss, 'anchor': anchors.detach(), 
                                        'cls_pred':cls_preds.detach(), 
                                        'bbox_pred':bbox_preds.detach()}
        # return loss
    
    # def on_train_start(self):
        # self.logger.log_hyperparams(self.hparams, {"valid_acc": 0})

    def validation_step(self, batch, batch_idx):
        x, y = batch
        anchors, cls_preds, bbox_preds = self.backbone(x)

        bbox_labels, bbox_masks, cls_labels = multibox_target(anchors, y)

        loss = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                      bbox_masks)

        loss = loss.mean()
        # print(loss)

        # calc_loss
        # # y_hat = torch.sigmoid(y_out)
        # loss = F.binary_cross_entropy_with_logits(y_hat, y)
        # # loss = tversky_loss(y_hat, y)

        # acc = dice_score(y_hat, y)
        self.log('valid_loss', loss.item())
        # self.log('train_acc', acc.item())
        return {'loss': loss, 'anchor': anchors, 
                                        'cls_pred':cls_preds, 
                                        'bbox_pred':bbox_preds}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        acc = dice_score(y_hat, y)
        self.log('test_acc', acc, on_step=True)
        return {'output': y_hat}
    
    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=7, eta_min=5e-6)


        return [optimizer], [scheduler]


class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # Equivalent to the assignment statement `self.blk_i = get_blk(i)`
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(
                self, f'cls_{i}',
                cls_predictor(idx_to_in_channels[i], num_anchors,
                              num_classes))
            setattr(self, f'bbox_{i}',
                    bbox_predictor(idx_to_in_channels[i], num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # Here `getattr(self, 'blk_%d' % i)` accesses `self.blk_i`
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1,
                                      self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds

from torch.utils.mobile_optimizer import optimize_for_mobile

if __name__ == '__main__':
    # main()
    mymodel = MySSD(in_channels=3, num_classes=len(voc_labels))

    # torch.save(mymodel, "model_ssd.pth")

    # torch.load("model_ssd.pth")

    example = torch.rand(1, 3, 400, 400)

    print(mymodel(example)[0].shape)
    # traced_script_module = torch.jit.trace(mymodel, example)


    # traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    # traced_script_module_optimized._save_for_lite_interpreter("model_ssd_v2.ptl")
    

