
from utils import *
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from model import MySSD, LitObjectDetect
import torchvision
from torchsummary import summary 
import callback as call
import os
# from data_download_banana import load_data_bananas
from custom_data import CustomData
import pandas as pd

# def transforms_banana(a)

batch_size, edge_size = 32, 256
# train_iter, valid_iter = load_data_bananas(batch_size)
# for x, y in train_iter:
#     print(x.dtype)
#     print(x.max())
#     break
def cli_main():
    pl.seed_everything(1234)
    """Combin all, load config and load dataset, them trainning
    """
    # ---------------------Load Config -------------------------------

    # conf = OmegaConf.load('../config.yaml')["train"]
    # print(OmegaConf.to_yaml(conf))

    # gpusVISIBLE = conf.gpus
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpusVISIBLE)

    # ---------------\\\\\\==========//////-------------------------

    # ---------------------Load Dataset ------------------------------
    
    df = pd.read_csv("df.csv")


    dataset = CustomData(df, label_map, transform=transform_albu, keep_difficult=False)
    # dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
    # load_data_bananas
    
    train_size = int(0.9 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True , num_workers=0, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)
    

    # test_input_dataset(train_dataset)
    # test_input_dataloader(train_loader)
    # ---------------\\\\\\============//////-------------------------
    # shape_input = test_train_data_loader(train_loader)
    # ---------------------Load Model---------------------------------
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_ssd = MySSD(
        in_channels=3,
        num_classes=1, 
        batch_norm=True,
    )
    # ------------
    # model
    # ------------
    model = LitObjectDetect(model_ssd)
    # model = LitObjectDetect.load_from_checkpoint("tb_logs/default/version_76/checkpoints/epoch=19-step=639.ckpt")
    
    # print(summary(model, (3,300,300), device="cpu"))
    # ------------
    # training
    # ------------
    logger = pl.loggers.TensorBoardLogger("tb_logs", default_hp_metric=False)
    
    folder_checkpoint = 'checkpoint'
    if not os.path.exists(folder_checkpoint): # create folder
        os.mkdir(folder_checkpoint)

    callbacks = [
                call.TrainInputOutputMonitor(),
                call.ValidInputOutputMonitor(),
                # call.WeightMonitor(),
                # call.checkpoint_callback,
                # call.TestMonitor(),
            ]


    trainer = pl.Trainer(gpus=1, 
                        logger=logger, 
                        max_epochs=20,
                        # limit_train_batches=limit_train_batches,
                        # limit_val_batches=limit_val_batches,
                        callbacks=callbacks,
                        )

    # trainer.fit(model, train_loader, val_loader) # val_loader
    # trainer.fit(model, train_loader, val_loader)
    trainer.fit(model, train_loader, val_loader)
    trainer.validate(model, valid_iter)

    torch.save(model.backbone.state_dict(), "model_ssd_V3.pt")

    # result = trainer.test(test_dataloaders=val_loader)

    # ------------
    # testing
    # ------------
    # result = trainer.test(test_dataloaders=val_loader)
    #print(result)


if __name__ == '__main__':
    

    cli_main()