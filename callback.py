import pytorch_lightning as pl
import torch.nn.functional as F
from utils import *
from torchvision.utils import make_grid
import cv2
import numpy as np


def get_heat_map_from_fearture_map(x, features_map):

    imgs = torch.clone(x)

    batch_size = x.shape[0]

    imgTensor_new = []

    for i in range(batch_size):

        img = imgs[i]
        # print(img.shape)
        img = img.permute(1,2,0).cpu().numpy()
        img_draw = np.copy(img*255).astype(np.uint8)
        img_draw = Image.fromarray(img_draw)
        img_draw = np.asarray(img_draw)

        grad = features_map[i]

        heatmap = torch.mean(grad, dim=0)
        heatmap = torch.maximum(heatmap, torch.zeros_like(heatmap))
        heatmap /= torch.max(heatmap)

        heatmap = heatmap.cpu().numpy()
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

        heatmap = np.uint8(255 * heatmap)

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        img_draw = heatmap * 0.4 + img_draw

        img_draw = torch.from_numpy(img_draw).permute(2,0,1).unsqueeze(0)

        imgTensor_new.append(img_draw)

    imgTensor_new = torch.cat(imgTensor_new)

    return imgTensor_new

def draw_img_label(imgTensor, y):

    imgs = torch.clone(imgTensor)
    imgTensor_new = []
    batch = imgs.shape[0]
    for i in range(batch):
        img = imgs[i]
        img = img.permute(1,2,0).cpu().numpy()
        img_draw = np.copy(img*255).astype(np.uint8)
        img_draw = Image.fromarray(img_draw)
        img_draw = np.asarray(img_draw)
        output = y[i]
        for row in output:

            h, w = img.shape[:2]
            x1 = int(row[1]*w)
            y1 = int(row[2]*h)
            x2 = int(row[3]*w)
            y2 = int(row[4]*h)
            label_idx = int(row[0])
            label = rev_label_map[label_idx]
            img_draw = cv2.rectangle(img_draw, (x1, y1), (x2, y2), distinct_colors[label_idx], 2)
            img_draw = cv2.putText(img_draw, f'{label}', (x1, y1-10), font, fontScale, distinct_colors[label_idx])
        img_draw = torch.from_numpy(img_draw).permute(2,0,1).unsqueeze(0)
        imgTensor_new.append(img_draw)
    imgTensor_new = torch.cat(imgTensor_new)
    return imgTensor_new



def predict_from_output(anchors, cls_preds, bbox_preds):
    batch_size = cls_preds.shape[0]
    # net.eval()
    # anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds.cpu(), dim=2).permute(0, 2, 1)
    output = multibox_detection(cls_probs, bbox_preds.cpu(), anchors.cpu())


    idxes = []
    outputs = []
    for batch in range(batch_size):
        idx = [i for i, row in enumerate(output[batch]) if int(row[0]) != -1]
        idxes.append(idx)
        outputs.append(output[batch, idx])



    # print(outputs[0][:5])
    # output_idx = torch.where(output[:,:,:1] >= 0  )
    # output_idx_1 = torch.where(output[:,:,:1] == -1  )
    # # print(output[0,:5])
    # print(output[:,:,:1][:,])
    # print(output_idx[1][27:32])
    # print(output_idx_1[1][:5])
    # print(output_idx.shape)
    # idx = [out[] for i, out in enumerate(output)]
    # idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    # out_new = [out[output_idx[1]] for out in output]
    # for out in out_new:
    #     print(torch.where(output[:,:,:1] >= 0)[0][:5])
        # print(torch.unique(out[output_idx[1]][:,:,:1]))
    # print(torch.unique([out[output_idx[1]] for out in output]))
    return outputs

def draw_img_predict(imgTensor, outputs, threshold=0.9):
    imgs = torch.clone(imgTensor)
    imgTensor_new = []
    batch = imgs.shape[0]
    for i in range(batch):
        img = imgs[i]
        img = img.permute(1,2,0).cpu().numpy()
        img_draw = np.copy(img*255).astype(np.uint8)
        img_draw = Image.fromarray(img_draw)
        img_draw = np.asarray(img_draw)
        output = outputs[i]
        for row in output:
            score = float(row[1])
            if score < threshold:
                continue
            h, w = img.shape[:2]
            x1 = int(row[2]*w)
            y1 = int(row[3]*h)
            x2 = int(row[4]*w)
            y2 = int(row[5]*h)
            label_idx = int(row[0])
            label = rev_label_map[label_idx]
            img_draw = cv2.rectangle(img_draw, (x1, y1), (x2, y2), distinct_colors[label_idx], 2)
            img_draw = cv2.putText(img_draw, f'{int(score*100)/100}% {label}', (x1, y1-10), font, fontScale, distinct_colors[label_idx])
        img_draw = torch.from_numpy(img_draw).permute(2,0,1).unsqueeze(0)
        imgTensor_new.append(img_draw)
    imgTensor_new = torch.cat(imgTensor_new)
    return imgTensor_new


class TrainInputOutputMonitor(pl.Callback):

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # return
        if (batch_idx + 1) % trainer.log_every_n_steps == 0:
        # if True:
            
            x, y = batch
            anchors = outputs['anchor']
            cls_preds = outputs['cls_pred']
            bbox_preds = outputs['bbox_pred']

            logger = trainer.logger
            logger.experiment.add_histogram("train_input", x, global_step=trainer.global_step)
            # logger.experiment.add_histogram("train_target", y, global_step=trainer.global_step)
            logger.experiment.add_histogram("train_anchor", anchors, global_step=trainer.global_step)
            logger.experiment.add_histogram("train_cls_pred", cls_preds, global_step=trainer.global_step)
            logger.experiment.add_histogram("train_bbox_pred", bbox_preds, global_step=trainer.global_step)


class ValidInputOutputMonitor(pl.Callback):

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # return
        if (batch_idx) % 5 == 0:
        # if True:
            
            x, y = batch

            # x_draw, y_draw = x[:5],
            # print(x.shape)
            anchors = outputs['anchor']
            cls_preds = outputs['cls_pred']
            bbox_preds = outputs['bbox_pred']

            output_data = predict_from_output(anchors, cls_preds, bbox_preds)
            # print(x.shape)
            predict_draw = draw_img_predict(x, output_data)[:16]
            label_draw = draw_img_label(x, y)[:16]
            label_grid = make_grid(label_draw, nrow=4)
            predict_grid = make_grid(predict_draw, nrow=4)

            gram_cam_draw = get_heat_map_from_fearture_map(x, pl_module.backbone._gradient)[:16]
            gram_cam_grip = make_grid(gram_cam_draw, nrow=4)

            logger = trainer.logger
            logger.experiment.add_histogram("valid_input", x, global_step=trainer.global_step)
            # logger.experiment.add_histogram("train_target", y, global_step=trainer.global_step)
            logger.experiment.add_histogram("valid_anchor", anchors, global_step=trainer.global_step)
            logger.experiment.add_histogram("valid_cls_pred", cls_preds, global_step=trainer.global_step)
            logger.experiment.add_histogram("valid_bbox_pred", bbox_preds, global_step=trainer.global_step)

            logger.experiment.add_image("valid_predict", predict_grid, global_step=trainer.global_step)
            logger.experiment.add_image("valid_label", label_grid, global_step=trainer.global_step)
            logger.experiment.add_image("valid_gram_cam", gram_cam_grip, global_step=trainer.global_step)