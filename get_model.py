import torch.nn as nn
from monai.networks.nets import SwinUNETR
from src.model.Class.HWAUNETR_class import HWAUNETR as TFM_UNET_class
from src.model.Seg.HWAUNETR_seg import HWAUNETR as TFM_UNET_seg
from src.model.Class.ResNet import resnet50
from src.model.Class.Vit import Vit as Vit
from src.model.Class.TP_Mamba import SAM_MS
from src.model.Multi_Tasks.HSL_Net import HSL_Net
from src.model.Multi_Tasks.HWAUNETR_Mu import HWAUNETRV2 as HWAUNETR
from src.model.Seg.uxnet_model import UXNET
from src.model.Seg.sabnet_model import SaBNet

# from src.model.Seg.nnFormer_model import nnFormer
from src.model.Seg.alien_model import ALIEN
from src.model.Seg.CoTr import ResTranUnet
from src.model.Seg.unetr_plus_plus import UNETR_PP


def get_model(config):
    # data choose
    if config.trainer.choose_dataset == "GCM":
        use_config = config.GCM_loader
    if config.trainer.choose_dataset == "Colon":
        use_config = config.colon_loader
    elif config.trainer.choose_dataset == "GCNC":
        use_config = config.GCNC_loader
    elif config.trainer.choose_dataset == "GICC":
        use_config = config.GICC_loader
    elif config.trainer.choose_dataset == "FS":
        use_config = config.FS_loader

    # Multitask choose model will return first
    if "HSL_Net" in config.trainer.choose_model:
        model = HSL_Net(
            in_channels=len(use_config.checkModels),
            out_channels=len(use_config.checkModels),
            num_tasks=1,
            hidden_size=768,
            depths=[2, 2, 2, 2],
            kernel_sizes=[4, 2, 2, 2],
            dims=[48, 96, 192, 384],
            out_dim=64,
            heads=[1, 2, 4, 4],
            out_indices=[0, 1, 2, 3],
            num_slices_list=[64, 32, 16, 8],
        )
        print("HSL_Net for multitask")
        return model
    elif "HWAUNETR" in config.trainer.choose_model:
        model = HWAUNETR(
            in_chans=len(use_config.checkModels),
            out_chans=len(use_config.checkModels),
            fussion=[1, 2, 4, 8],
            kernel_sizes=[4, 2, 2, 2],
            depths=[2, 2, 2, 2],
            dims=[48, 96, 192, 384],
            heads=[1, 2, 4, 4],
            hidden_size=768,
            num_slices_list=[64, 32, 16, 8],
            out_indices=[0, 1, 2, 3],
        )
        return model

    # Single task choose model return now
    if config.trainer.task == "Segmentation":
        if "TFM_UNET" in config.trainer.choose_model:
            model = TFM_UNET_seg(
                in_chans=len(use_config.checkModels),
                out_chans=len(use_config.checkModels),
                fussion=[1, 2, 4, 8],
                kernel_sizes=[4, 2, 2, 2],
                depths=[2, 2, 2, 2],
                dims=[48, 96, 192, 384],
                heads=[1, 2, 4, 4],
                hidden_size=768,
                num_slices_list=[64, 32, 16, 8],
                out_indices=[0, 1, 2, 3],
            )
            print("TFM_UNET for segmentation")
        elif "SwinUNETR" in config.trainer.choose_model:
            model = SwinUNETR(
                in_channels=len(use_config.checkModels),
                out_channels=len(use_config.checkModels),
                img_size=use_config.target_size,
                feature_size=48,
            )
            print("SwinUNETR for segmentation")
        elif "UXNET" in config.trainer.choose_model:
            model = UXNET(
                in_chans=len(use_config.checkModels),
                out_chans=len(use_config.checkModels),
                depths=[2, 2, 2, 2],
                feat_size=[48, 96, 192, 384],
                drop_path_rate=0.0,
                layer_scale_init_value=1e-6,
                spatial_dims=3,
            )
            print("UXNET for segmentation")
        elif "SaBNet" in config.trainer.choose_model:
            model = SaBNet(
                in_chs=len(use_config.checkModels),
                out_chs=len(use_config.checkModels),
                num_heads=2,
            )
            print("SaBNet for segmentation")
        #
        elif "ALIEN" in config.trainer.choose_model:
            model = ALIEN(
                n_channels=len(use_config.checkModels),
                n_classes=len(use_config.checkModels),
                trilinear=True,
            )
            print("ALIEN for segmentation")
        elif "CoTr" in config.trainer.choose_model:
            model = ResTranUnet(
                in_channels=len(use_config.checkModels),
                num_classes=len(use_config.checkModels),
                img_size=(
                    use_config.target_size[0],
                    use_config.target_size[1],
                    use_config.target_size[2],
                ),
                deep_supervision=False,
            )
            print("CoTr for segmentation")
        elif "UNETR_PP" in config.trainer.choose_model:
            model = UNETR_PP(
                in_channels=len(use_config.checkModels),
                out_channels=len(use_config.checkModels),
                img_size=(
                    use_config.target_size[0],
                    use_config.target_size[1],
                    use_config.target_size[2],
                ),
                feature_size=16,
                num_heads=4,
                depths=[3, 3, 3, 3],
                dims=[32, 64, 128, 256],
                do_ds=True,
            )
            print("UNETR++ for segmentation")
    elif config.trainer.task == "Classification":
        if "ResNet" in config.trainer.choose_model:
            model = resnet50(
                in_classes=len(use_config.checkModels),
                num_classes=1,
                shortcut_type="B",
                spatial_size=64,
                sample_count=128,
            )
            print("ResNet for classification")

        elif "Vit" in config.trainer.choose_model:
            model = Vit(
                in_channels=len(use_config.checkModels),
                out_channels=len(use_config.checkModels),
                embed_dim=96,
                embedding_dim=32,
                channels=(24, 48, 60),
                blocks=(1, 2, 3, 2),
                heads=(1, 2, 4, 4),
                r=(4, 2, 2, 1),
                dropout=0.3,
            )
            print("ViT for classification")

        elif "TFM_UNET" in config.trainer.choose_model:
            model = TFM_UNET_class(
                in_chans=len(use_config.checkModels),
                fussion=[1, 2, 4, 8],
                kernel_sizes=[4, 2, 2, 2],
                depths=[1, 1, 1, 1],
                dims=[48, 96, 192, 384],
                heads=[1, 2, 4, 4],
                hidden_size=768,
                num_slices_list=[64, 32, 16, 8],
                out_indices=[0, 1, 2, 3],
            )
            print("TFM_UNET for classification")
        elif config.trainer.choose_model == "TP_Mamba":
            model = SAM_MS(
                in_classes=len(use_config.checkModels), num_classes=2, dr=16.0
            )
            print("TP_Mamba for classification")

        elif config.trainer.choose_model == "X3D":
            from src.model.Class.X3D_Efficient import X3D_Classifier

            model = X3D_Classifier(
                in_channels=len(use_config.checkModels),
                num_classes=1,
                dropout_rate=0.5,
            )
            print("X3D for classification")
        elif config.trainer.choose_model == "AMSNET":
            from src.model.Class.AMSNet import AMSNet

            model = AMSNet(
                in_channels=len(use_config.checkModels),
                num_classes=1,
            )
            print("AMSNet for classification")
        elif config.trainer.choose_model == "HCNN":
            from src.model.Class.Hybrid_CNN_Transformer import Hybrid_CNN_Transformer

            model = Hybrid_CNN_Transformer(
                in_channels=len(use_config.checkModels),
                img_size=(
                    use_config.target_size[0],
                    use_config.target_size[1],
                    use_config.target_size[2],
                ),
                num_classes=1,
                embed_dim=256,
            )
            print("HCNN for classification")
        elif config.trainer.choose_model == "M3T":
            from src.model.Class.M3T import M3T

            model = M3T(
                in_channels=len(use_config.checkModels),
                img_size=(
                    use_config.target_size[0],
                    use_config.target_size[1],
                    use_config.target_size[2],
                ),
                num_classes=1,
            )
            print("M3T for classification")
        elif config.trainer.choose_model == "Med3D":
            from src.model.Class.Med3D_ResNet import generate_model

            model = generate_model(
                model_depth=18, in_channels=len(use_config.checkModels), num_classes=1
            )
            print("Med3D for classification")
        elif config.trainer.choose_model == "Swin3D":
            from src.model.Class.Swin3D_Classifier import Swin3D_Classifier

            model = Swin3D_Classifier(
                in_channels=len(use_config.checkModels),
                img_size=(
                    use_config.target_size[0],
                    use_config.target_size[1],
                    use_config.target_size[2],
                ),
                num_classes=1,
            )
            print("Swin3D for classification")
    else:
        raise ValueError(
            "Invalid task type. Choose either 'segmentation' or 'classification'."
        )

    return model
