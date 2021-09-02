import torch
import torch.nn as nn 
import torch.nn.functional as F
from loss import calc_loss
from utils import *
# from pytorch_lightning import LightningModule
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
            self.bn1 = torch.nn.BatchNorm2d(mid_channel)
            self.bn2 = torch.nn.BatchNorm2d(out_channels)
            
    def forward(self, x):
        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = torch.nn.functional.relu(x, inplace=True)
        
        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        out = torch.nn.functional.relu(x, inplace=True)
        return out

class Tiny(torch.nn.Module):
    def forward(self, X):
        anchor = multibox_prior(X, [0.2, 0.272], [1, 2, 0.5])
        
        return anchor
        

class MySSD(torch.nn.Module):
    """Encode is Blocks conv, Decode is UpSample with mode 'nearest'. Model
        use BatchNorm2d

    Args:
        torch ([Module pytorch]): [create model Unet with class Module of pytorch]
    """
    def up(self, x, size):
        return torch.nn.functional.interpolate(x, size=size, mode=self.upscale_mode)
    
    def down(self, x):
        return torch.nn.functional.max_pool2d(x, kernel_size=2)
    
    def __init__(self, in_channels, num_classes=None, batch_norm=False, upscale_mode="nearest"):
        super().__init__()
        
        # config class model
        self.in_channels = in_channels
        # self.out_channels = out_channels
        self.batch_norm = batch_norm
        self.upscale_mode = upscale_mode

        self.sizes = [torch.tensor([0.2, 0.272]), 
                        torch.tensor([0.37, 0.447]), 
                        torch.tensor([0.54, 0.619]),
                      torch.tensor([0.71, 0.79]), 
                        torch.tensor([0.88, 0.961])]
        self.ratios = [torch.tensor([1, 2, 0.5])] * 5
        self.num_anchors = [len(size) + len(ratio) - 1 for size, ratio in zip(self.sizes, self.ratios)]

        self.auxiliaryconvs = AuxiliaryConvolutions()
        
        # encoder
        self.enc1 = Block(in_channels, 16, 16, batch_norm)
        self.enc2 = Block(16, 32, 32, batch_norm)
        self.enc3 = Block(32, 64, 64, batch_norm)
        self.enc4 = Block(64, 128, 128, batch_norm)
        # self.enc5 = Block(128, 256, 256, batch_norm)
        
        self.num_classes = num_classes
        idx_to_in_channels = [128, 128, 128, 128]
        for i in range(len(idx_to_in_channels)):
            # Equivalent to the assignment statement `self.blk_i = get_blk(i)`
            setattr(
                self, f'cls_{i}',
                cls_predictor(idx_to_in_channels[i], self.num_anchors[i],
                              self.num_classes))
            setattr(self, f'bbox_{i}',
                    bbox_predictor(idx_to_in_channels[i], self.num_anchors[i]))
        
        # output layer
        # self.out = torch.nn.Conv3d(in_channels=16, out_channels=out_channels, kernel_size=1)

        self.gradients = None


    def forward(self, x):
        anchors, cls_preds, bbox_preds = [None] * 4, [None] * 4, [None] * 4

        enc1 = self.enc1(x) # 300x300
        enc2 = self.enc2(self.down(enc1)) # 150x150
        enc3 = self.enc3(self.down(enc2)) # 75x75
        enc4 = self.enc4(self.down(enc3)) # 

        self.gradients = enc4
        # enc5 = self.enc5(self.down())

        enc5, enc6, enc7, enc8 = self.auxiliaryconvs(enc4)

        for i, enc in enumerate([enc5, enc6, enc7, enc8]):


            anchor = multibox_prior(enc, self.sizes[i], self.ratios[i])

            cls_pred = getattr(self, f'cls_{i}')(enc)

            bbox_pred = getattr(self, f'bbox_{i}')(enc)
            # print(anchor.shape)
            anchors[i] = anchor
            cls_preds[i] = cls_pred
            bbox_preds[i] = bbox_pred

        # print(anchors)
        anchors = torch.cat(anchors, dim=1)

        cls_preds = concat_preds(cls_preds)
        # print("OK")
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1,
                                      self.num_classes + 1)
        # print("OK")
        bbox_preds = concat_preds(bbox_preds)
        # print("OK")
        # AuxiliaryConvolutions()
        

        # print(anchors.shape, cls_preds.shape, bbox_preds.shape)
        assert cls_preds.shape[2] == self.num_classes + 1
        assert anchors.shape[1] == cls_preds.shape[1] == int(bbox_preds.shape[1]/4)
        return [cls_preds, bbox_preds]


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
"""
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
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

"""
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

# from torch.utils.mobile_optimizer import optimize_for_mobile
# mymodel = MySSD(in_channels=3, num_classes=1)

# torch.save(mymodel, "model_ssd.pth")

# # torch.load("model_ssd.pth")

# example = torch.rand(1, 3, 300, 300)

# print(mymodel(example)[0].shape)
# traced_script_module = torch.jit.tr
"""
if __name__ == '__main__':
    # main()
    mymodel = MySSD(in_channels=3, num_classes=1)

    torch.save(mymodel, "model_ssd.pth")

    # torch.load("model_ssd.pth")

    example = torch.rand(1, 3, 300, 300)

    print(mymodel(example)[0].shape)
    # traced_script_module = torch.jit.trace(mymodel, example)


    # traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    # traced_script_module_optimized._save_for_lite_interpreter("model_ssd_v2.ptl")
    

"""