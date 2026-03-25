from calendar import c
from json import load
import os
import re
import math
import yaml
import torch
import monai
import random
import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
from datetime import datetime
from easydict import EasyDict
import torch.nn.functional as F
from monai.utils import ensure_tuple_rep
from monai.networks.utils import one_hot
from typing import List, Dict, Any
from monai.transforms import MapTransform
sitk.ProcessObject.SetGlobalWarningDisplay(False)
from typing import Tuple, List, Mapping, Hashable, Dict
from monai.data import DataLoader, pad_list_data_collate
from monai.transforms import (
    Compose,
    LoadImage,
    LoadImaged,
    EnsureTyped,
    MapTransform,
    ScaleIntensityRanged,
    EnsureChannelFirstd,
    Spacingd,
    RandAffined,
    RandRotate90d,
    Orientationd,
    ResampleToMatchd,
    ResizeWithPadOrCropd,
    Resize,
    ConcatItemsd,
    DeleteItemsd,
    Resized,
    RandFlipd,
    NormalizeIntensityd,
    ToTensord,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ScaleIntensityRangePercentilesd,
)


class ConvertToMultiChannelBasedOnBratsClassesd_For_BraTS(
    monai.transforms.MapTransform
):
    """
    TC WT ET
    Dictionary-based wrapper of :py:class:`monai.transforms.ConvertToMultiChannelBasedOnBratsClasses`.
    Convert labels to multi channels based on brats18 classes:
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the peritumoral edema
    label 4 is the GD-enhancing tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    backend = [monai.utils.TransformBackends.TORCH, monai.utils.TransformBackends.NUMPY]

    def __init__(
        self,
        keys: monai.config.KeysCollection,
        is2019: bool = False,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.is2019 = is2019

    def converter(self, img: monai.config.NdarrayOrTensor):
        # TC WT ET
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)
        if self.is2019:
            result = [
                (img == 2) | (img == 3),
                (img == 1) | (img == 2) | (img == 3),
                (img == 2),
            ]
        else:
            # TC WT ET
            result = [
                (img == 1) | (img == 4),
                (img == 1) | (img == 4) | (img == 2),
                img == 4,
            ]
            # merge labels 1 (tumor non-enh) and 4 (tumor enh) and 2 (large edema) to WT
            # label 4 is ET
        return (
            torch.stack(result, dim=0)
            if isinstance(img, torch.Tensor)
            else np.stack(result, axis=0)
        )

    def __call__(
        self, data: Mapping[Hashable, monai.config.NdarrayOrTensor]
    ) -> Dict[Hashable, monai.config.NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


class SafeLoadDICOMd(MapTransform):
    """
    读取 DICOM 序列（list[str]）为 (H, W, Z) 的 numpy，并写入完整 meta_dict：
      - spacing: (sx, sy, sz)
      - direction: 9 或 16 长度的方向余弦
      - origin: (ox, oy, oz)
      - affine: 4x4 齐次矩阵（由 direction/spacing/origin 构造）
      - original_channel_dim: "no_channel"（提示 EnsureChannelFirstd 需要新建通道维）

    用法：放在 Compose 最前面，对应 keys 为 DICOM 的键（每个值是 list[str]）。
    """

    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            file_list = d[key]
            if not isinstance(file_list, (list, tuple)) or len(file_list) == 0:
                raise ValueError(f"{key} must be a non-empty list of DICOM paths.")

            # 用 SimpleITK 读取 3D 序列
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(list(file_list))
            try:
                sitk_img = reader.Execute()  # SimpleITK.Image
            except Exception as e:
                raise RuntimeError(
                    f"SimpleITK failed to read DICOM series for key '{key}'."
                ) from e

            # SimpleITK: GetArrayFromImage -> (Z, Y, X)
            arr_zyx = sitk.GetArrayFromImage(sitk_img)
            # 转为 (H, W, Z) = (Y, X, Z)
            arr_hwz = np.transpose(arr_zyx, (1, 2, 0))

            # 提取元信息
            spacing_xyz = sitk_img.GetSpacing()  # (sx, sy, sz) in X,Y,Z
            origin_xyz = sitk_img.GetOrigin()  # (ox, oy, oz)
            direction = sitk_img.GetDirection()  # len=9 (3x3) 或 16 (4x4)

            # 构造 4x4 affine：R * diag(spacing) 作为旋转缩放，origin 作为平移
            if len(direction) == 9:
                R = np.array(direction, dtype=np.float64).reshape(3, 3)
            elif len(direction) == 16:
                R = np.array(direction, dtype=np.float64).reshape(4, 4)[:3, :3]
            else:
                # 不常见情况，退化为 I
                R = np.eye(3, dtype=np.float64)
            S = np.diag(spacing_xyz)  # diag(sx, sy, sz)
            A = np.eye(4, dtype=np.float64)
            A[:3, :3] = R @ S
            A[:3, 3] = np.array(origin_xyz, dtype=np.float64)

            # ✅ 新增 spatial_shape 字段
            spatial_shape = arr_hwz.shape  # (H, W, Z)
            
            # 写回数据与 meta
            d[key] = arr_hwz  # (H, W, Z)
            d[f"{key}_meta_dict"] = {
                "spacing": spacing_xyz,
                "direction": tuple(direction),
                "origin": origin_xyz,
                "affine": A,  # 提供 affine，利于 Orientationd/Spacingd
                "spatial_shape": spatial_shape,  # ✅ 加这一行
                "original_channel_dim": "no_channel",
            }
        return d


class PreserveMetaD(MapTransform):
    def __init__(self, img_keys, lab_keys):
        super().__init__(img_keys + lab_keys)
        self.img_keys = img_keys
        self.lab_keys = lab_keys

    def __call__(self, data):
        d = dict(data)
        # 保留任意一个模态的 meta 作为总体 meta
        first_img = self.img_keys[0]
        first_lab = self.lab_keys[0]
        d["image_meta_dict"] = d.get(f"{first_img}_meta_dict", {})
        d["label_meta_dict"] = d.get(f"{first_lab}_meta_dict", {})
        return d


class AssertPairAlignedd(MapTransform):
    """
    在 ConcatItemsd 之前调用：
    对每个模态的 (img_key, lab_key) 检查 shape/spacing/direction/affine 是否对齐。
    """

    def __init__(
        self, img_keys, lab_keys, atol_spacing=1e-4, atol_affine=1e-3, atol_dir=1e-4
    ):
        super().__init__(img_keys + lab_keys)
        assert len(img_keys) == len(lab_keys)
        self.img_keys = img_keys
        self.lab_keys = lab_keys
        self.atol_spacing = atol_spacing
        self.atol_affine = atol_affine
        self.atol_dir = atol_dir

    def __call__(self, data):
        d = dict(data)
        for ik, lk in zip(self.img_keys, self.lab_keys):
            # 形状
            ishape = np.array(d[ik].shape, dtype=int)
            lshape = np.array(d[lk].shape, dtype=int)
            if not np.array_equal(ishape, lshape):
                raise ValueError(
                    f"Shape mismatch between {ik} and {lk}: {tuple(ishape)} vs {tuple(lshape)}"
                )

            # meta
            im = d.get(f"{ik}_meta_dict", {}) or {}
            lm = d.get(f"{lk}_meta_dict", {}) or {}

            # spacing
            isp = np.array(im.get("spacing", ()), dtype=float)
            lsp = np.array(lm.get("spacing", ()), dtype=float)
            if (
                isp.size
                and lsp.size
                and not np.allclose(isp, lsp, atol=self.atol_spacing, rtol=0)
            ):
                raise ValueError(
                    f"Spacing mismatch between {ik} and {lk}: {isp} vs {lsp}"
                )

            # direction（方向余弦长度可能是9或16，统一到3x3）
            def _dir_to_3x3(md):
                direction = np.array(md.get("direction", ()), dtype=float)
                if direction.size == 9:
                    return direction.reshape(3, 3)
                if direction.size == 16:
                    return direction.reshape(4, 4)[:3, :3]
                return None

            idr = _dir_to_3x3(im)
            ldr = _dir_to_3x3(lm)
            if (
                idr is not None
                and ldr is not None
                and not np.allclose(idr, ldr, atol=self.atol_dir, rtol=0)
            ):
                raise ValueError(f"Direction mismatch between {ik} and {lk}")

            # affine（若都有就比）
            ia = np.array(im.get("affine", ()), dtype=float)
            la = np.array(lm.get("affine", ()), dtype=float)
            if (
                ia.size
                and la.size
                and not np.allclose(ia, la, atol=self.atol_affine, rtol=0)
            ):
                raise ValueError(f"Affine mismatch between {ik} and {lk}")
        return d


def sort_dcm_paths(paths):
    def extract_number(filename):
        # 提取文件名中的数字部分
        match = re.search(r"(\d+)\.dcm$", filename)
        return int(match.group(1)) if match else -1

    return sorted(paths, key=extract_number)


def read_csv_for_GCM(config):
    csv_path = config.GCM_loader.root + "/" + "Classification.xlsx"
    # 定义dtype转换，将第二列（索引为1）读作str
    dtype_converters = {1: str}

    df1 = pd.read_excel(
        csv_path, engine="openpyxl", dtype=dtype_converters, sheet_name="腹膜转移分类"
    )
    df2 = pd.read_excel(
        csv_path,
        engine="openpyxl",
        dtype=dtype_converters,
        sheet_name="淋巴结转移分类",
    )

    # 创建空字典
    content_dict1 = {}
    content_dict2 = {}
    content_dict3 = {}
    # 遍历DataFrame的每一行，从第二行开始
    for index, row in df1.iterrows():

        key = row[1]  # 第2列作为键
        values = row[2]  # 第3列的数据读为label
        time = row[4]  # 第4列的数据读为time，用于划分数据
        content_dict1[key] = [values, time]

    for index, row in df2.iterrows():

        key = row[1]  # 第2列作为键
        values1 = row[3]  # 第4列的数据读为label
        values2 = row[4]  # 第5列的数据读为label
        time = row[4]
        content_dict2[key] = [values1, values2, time]


    return content_dict1, content_dict2


def read_csv_for_GICC(csv_path: str) -> List[Dict[str, str]]:
    dtype_converters = {0: str}
    train_df = pd.read_excel(
        os.path.join(csv_path, "train.xlsx"), engine="openpyxl", dtype={0: str}, sheet_name="Sheet1"
    )
    val_df = pd.read_excel(
        os.path.join(csv_path, "val.xlsx"), engine="openpyxl", dtype={0: str}, sheet_name="Sheet1"
    )
    test_df = pd.read_excel(
        os.path.join(csv_path, "test.xlsx"), engine="openpyxl", dtype={4: str}, sheet_name="Sheet1"
    )
    
    # 创建空字典
    train_dict = {}
    val_dict = {}
    test_dict = {}
    
    # 遍历DataFrame的每一行，从第二行开始
    for index, row in train_df.iterrows():
        key = row[0]  # 第2列作为键
        values = row[16]  # TODO: 随便找个值先填上，此处可以改读为分类标签
        if pd.isna(values):   # NaN / None 都能识别
            values = 0
        train_dict[key] = [values]
    
    for index, row in val_df.iterrows():
        key = row[0]  # 第2列作为键
        values = row[14]  # TODO: 随便找个值先填上，此处可以改读为分类标签
        if pd.isna(values):   # NaN / None 都能识别
            values = 0
        val_dict[key] = [values]
    
    for index, row in test_df.iterrows():
        key = row[4]  # 第2列作为键
        values = row[13]  # TODO: 随便找个值先填上，此处可以改读为分类标签
        if pd.isna(values):   # NaN / None 都能识别
            values = 0
        test_dict[key] = [values]
    
    return train_dict, val_dict, test_dict

def read_csv_for_GCNC(config):

    csv_path1 = config.GCNC_loader.root + "/" + "NPC_new.xlsx"

    # 定义dtype转换，将第四列（索引为3）读作str
    dtype_converters = {3: str}

    df = pd.read_excel(csv_path1, engine="openpyxl", dtype=dtype_converters)

    # 创建空列表
    content_dict = {}

    # 遍历DataFrame的每一行，从第二行开始
    for index, row in df.iterrows():

        key = str(row["ID"])

        # TODO: 中心3以0开头设计ID，ID需要特殊读取避免删0
        if row["中心"] == 3:
            key = key.zfill(10)

        # TODO: 此处只做高低表达分类记录。后续数据补全可以添加阴阳性分类
        # 如果指标缺失，则跳过读取
        if isinstance(row["label"], (int, float)) != True:
            print(f"{key} miss label")
            continue

        # try:
        #     row["PD-L1"] = float(row["PD-L1"])
        # except:
        #     try:
        #         if "未做" in row["PD-L1"]:
        #             print(f"{key} miss PD-L1")
        #             continue
        #         elif "＜1" in row["PD-L1"]:
        #             row["PD-L1"] = 0
        #     except:
        #         print(f"{key} miss PD-L1")
        #         continue

        # label_dict = {"PD-L1": -1, "label": -1, "center": -1}
        label_dict = {"label": -1, "center": -1}

        # if row["PD-L1"] > 20:
        #     label_dict["PD-L1"] = 1
        # else:
        #     label_dict["PD-L1"] = 0

        label_dict["M"] = row["label"]

        label_dict["center"] = row["中心"]

        if label_dict["M"] != -1:
            content_dict[key] = label_dict
        else:
            print(f"{key} miss label")
    return content_dict


def read_usedata(file_path):
    read_flas = False
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if read_flas == True:
                result = line.replace("\n", "").split(",")
                result = [data.replace(" ", "") for data in result]
                return result
            elif "Useful data" in line:
                read_flas = True
                continue


def load_MR_dataset_images(
    root, use_data, use_models, use_data_dict={}, data_choose="GCM", test_center=3
):
    images_path = os.listdir(root)
    images_list = []
    test_images_list = []
    images_lack_list = []

    for path in use_data:
        path = str(path)
        if path in images_path:
            models = os.listdir(root + "/" + path + "/")
        else:
            print(f"{path} is not in {root}. ")
            continue
        lack_flag = False
        lack_model_flag = False
        image = []
        label = []

        for model in models:
            if "WI" in model and data_choose != "GCM":
                check_model = model.replace("WI", "")
            # elif "WI" in model and data_choose == "GCM":
            #     if "T2" in model:
            #         check_model = model.replace("WI", "_FS")
            elif "T2" in model and data_choose == "GCM":
                check_model = "T2_FS"
            elif model == "CT1":
                check_model = "T1+C"
            elif model == "T1+c":
                check_model = "T1+C"
            else:
                check_model = model

            if check_model in use_models:
                if not os.path.exists(root + "/" + path + "/" + model):
                    print(f"{path} does not have {model} file. ")
                    lack_model_flag = True
                    break
                elif not os.path.exists(
                    root + "/" + path + "/" + model + "/" + path + ".nii.gz"
                ):
                    print(f"{path} does not have {model} image file.")
                    lack_model_flag = True
                    break

                image.append(root + "/" + path + "/" + model + "/" + path + ".nii.gz")
                if not os.path.exists(
                    root + "/" + path + "/" + model + "/" + path + "seg.nii.gz"
                ):  
                    if os.path.exists(
                    root + "/" + path + "/" + model + "/" + path + "seg.nii"):
                        label.append(
                            root + "/" + path + "/" + model + "/" + path + "seg.nii"
                        )
                    elif os.path.exists(
                        root + "/" + path + "/" + model + "/" + path + "SEG.nii"
                    ):
                        label.append(
                            root + "/" + path + "/" + model + "/" + path + "SEG.nii"
                        )
                    else:
                        label.append(
                            root + "/" + path + "/" + model + "/" + path + ".nii.gz"
                        )
                        print(f"Label file not found for {path} in model {model}. ")
                        lack_flag = True
                else:
                    label.append(
                        root + "/" + path + "/" + model + "/" + path + "seg.nii.gz"
                    )

        if image == [] or len(image) < len(use_models):
            print(f"{path} does not have image file or not enough modals. ")
            lack_model_flag = True

        if data_choose == "GCM":
            if lack_flag == False and lack_model_flag == False:
                if use_data_dict != {}:
                    images_list.append(
                        {
                            "image": image,
                            "label": label,
                            "class_label": use_data_dict[path][0],
                            "PFS_label": use_data_dict[path][1],
                        }
                    )
                else:
                    images_list.append(
                        {
                            "image": image,
                            "label": label,
                        }
                    )
            elif lack_model_flag == False:
                if use_data_dict != {}:
                    images_lack_list.append(
                        {
                            "image": image,
                            "label": label,
                            "class_label": use_data_dict[path][0],
                            "PFS_label": use_data_dict[path][1],
                        }
                    )
                else:
                    images_lack_list.append(
                        {
                            "image": image,
                            "label": label,
                        }
                    )
        else:
            if lack_flag == False and lack_model_flag == False:
                if use_data_dict != {}:
                    if use_data_dict[path]["center"] == test_center:
                        test_images_list.append(
                            {
                                "image": image,
                                "label": label,
                                # "pdl1_label": use_data_dict[path]["PD-L1"],
                                "m_label": use_data_dict[path]["M"],
                                "center": use_data_dict[path]["center"],
                            }
                        )
                    else:
                        images_list.append(
                            {
                                "image": image,
                                "label": label,
                                # "pdl1_label": use_data_dict[path]["PD-L1"],
                                "m_label": use_data_dict[path]["M"],
                                "center": use_data_dict[path]["center"],
                            }
                        )
                else:
                    images_list.append(
                        {
                            "image": image,
                            "label": label,
                        }
                    )
            elif lack_model_flag == False:
                if use_data_dict != {}:
                    images_lack_list.append(
                        {
                            "image": image,
                            "label": label,
                            # "pdl1_label": use_data_dict[path]["PD-L1"],
                            "m_label": use_data_dict[path]["M"],
                            "center": use_data_dict[path]["center"],
                        }
                    )
                else:
                    images_lack_list.append(
                        {
                            "image": image,
                            "label": label,
                        }
                    )
        # print(f'{path} example has been loaded')

    if data_choose == "GCM":
        return images_list, images_lack_list
    else:
        return images_list, test_images_list, images_lack_list


def load_MR_tif_dataset_images(root, use_data, use_models):
    images_path = os.listdir(root)
    images_list = []

    for path in use_data:
        path = str(path)
        images = []
        labels = []

        if path not in images_path:
            print(f"{path} is not in {root}. ")
            continue

        for modal in use_models:
            image = []
            label = []

            for img in os.listdir(root + "/" + path):
                if (modal in img) and ("mask" not in img):
                    image.append(root + "/" + path + "/" + img)
                elif (modal in img) and ("mask" in img):
                    label.append(root + "/" + path + "/" + img)

            images.append(image)
            labels.append(label)

        images_list.append({"image": images, "label": labels})

    return images_list


def load_MR_dcm_dataset_images(root, use_data, use_models):
    images_path = os.listdir(root)
    images_list = []
    for path in use_data:
        path = str(path)
        images = {}
        labels = {}

        if path not in images_path:
            print(f"{path} is not in {root}. ")
            continue

        for modal in use_models:
            image = []
            label = []

            if "CE-T1" in modal:
                label_flag = "ROI-CE-T1"
            elif "T2" in modal:
                label_flag = "ROI-T2"
            else:
                label_flag = "ROI-T1"

            for img in os.listdir(root + "/" + path + "/" + modal):
                image.append(root + "/" + path + "/" + modal + "/" + img)

            image = sort_dcm_paths(image)  # 将dcm文件按数字顺序排序

            images[modal] = image
            labels[modal] = root + "/" + path + "/" + label_flag + ".nii"

        images_list.append({"image": images, "label": labels})

    return images_list


def load_brats2021_dataset_images(root):
    images_path = os.listdir(root)
    images_list = []
    for path in images_path:
        image_path = root + "/" + path + "/" + path
        flair_img = image_path + "_flair.nii.gz"
        t1_img = image_path + "_t1.nii.gz"
        t1ce_img = image_path + "_t1ce.nii.gz"
        t2_img = image_path + "_t2.nii.gz"
        seg_img = image_path + "_seg.nii.gz"
        images_list.append(
            {"image": [flair_img, t1_img, t1ce_img, t2_img], "label": seg_img}
        )
    return images_list


def get_GCM_transforms(
    config: EasyDict,
) -> Tuple[monai.transforms.Compose, monai.transforms.Compose]:
    load_transform = []
    for model_scale in config.GCM_loader.model_scale:
        load_transform.append(
            monai.transforms.Compose(
                [
                    LoadImaged(
                        keys=["image", "label"], image_only=False, simple_keys=True
                    ),
                    EnsureChannelFirstd(keys=["image", "label"]),
                    Resized(
                        keys=["image", "label"],
                        spatial_size=config.GCM_loader.target_size,
                        mode=("trilinear", "nearest-exact"),
                    ),
                    ScaleIntensityRanged(
                        keys=["image"],  # 对图像应用变换
                        a_min=model_scale[0],  # 输入图像的最小强度值
                        a_max=model_scale[1],  # 输入图像的最大强度值
                        b_min=0.0,  # 输出图像的最小强度值
                        b_max=1.0,  # 输出图像的最大强度值
                        clip=True,  # 是否裁剪超出范围的值
                    ),
                    ToTensord(keys=["image", "label"]),
                ]
            )
        )

    train_transform = monai.transforms.Compose(
        [
            # 训练集的额外增强
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = monai.transforms.Compose(
        [
            ToTensord(keys=["image", "label"]),
        ]
    )
    return load_transform, train_transform, val_transform


def get_GICC_transforms(
    config: EasyDict,
) -> Tuple[monai.transforms.Compose, monai.transforms.Compose]:
    load_transform = []
    for model_scale in config.GICC_loader.model_scale:
        load_transform.append(
            monai.transforms.Compose(
                [
                    LoadImaged(
                        keys=["image", "label"], image_only=False, simple_keys=True
                    ),
                    EnsureChannelFirstd(keys=["image", "label"]),
                    Resized(
                        keys=["image", "label"],
                        spatial_size=config.GICC_loader.target_size,
                        mode=("trilinear", "nearest-exact"),
                    ),
                    ScaleIntensityRanged(
                        keys=["image"],  # 对图像应用变换
                        a_min=model_scale[0],  # 输入图像的最小强度值
                        a_max=model_scale[1],  # 输入图像的最大强度值
                        b_min=0.0,  # 输出图像的最小强度值
                        b_max=1.0,  # 输出图像的最大强度值
                        clip=True,  # 是否裁剪超出范围的值
                    ),
                    ToTensord(keys=["image", "label"]),
                ]
            )
        )

    train_transform = monai.transforms.Compose(
        [
            # 训练集的额外增强
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = monai.transforms.Compose(
        [
            ToTensord(keys=["image", "label"]),
        ]
    )
    return load_transform, train_transform, val_transform


def get_GCNC_transforms(
    config: EasyDict,
) -> Tuple[monai.transforms.Compose, monai.transforms.Compose]:
    load_transform = []
    for model_scale in config.GCNC_loader.model_scale:
        load_transform.append(
            monai.transforms.Compose(
                [
                    LoadImaged(
                        keys=["image", "label"], image_only=False, simple_keys=True
                    ),
                    EnsureChannelFirstd(keys=["image", "label"]),
                    Resized(
                        keys=["image", "label"],
                        spatial_size=config.GCNC_loader.target_size,
                        mode=("trilinear", "nearest-exact"),
                    ),
                    Orientationd(keys=["image", "label"], axcodes="RAS"),
                    ScaleIntensityRanged(
                        keys=["image"],  # 对图像应用变换
                        a_min=model_scale[0],  # 输入图像的最小强度值
                        a_max=model_scale[1],  # 输入图像的最大强度值
                        b_min=0.0,  # 输出图像的最小强度值
                        b_max=1.0,  # 输出图像的最大强度值
                        clip=True,  # 是否裁剪超出范围的值
                    ),
                    ToTensord(keys=["image", "label"]),
                ]
            )
        )

    train_transform = monai.transforms.Compose(
        [
            # 训练集的额外增强
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = monai.transforms.Compose(
        [
            ToTensord(keys=["image", "label"]),
        ]
    )
    return load_transform, train_transform, val_transform



def get_FS_transforms(
    modalities,
    target_spacing=(1.0, 1.0, 1.0),
    target_size=(256, 256, 64),  # (W,H,Z)
    dtype_image=torch.float32,
    dtype_label=torch.float32,  # True => 含 Rand* 增强
    orientation_axcodes="RAS",
) -> Tuple[monai.transforms.Compose, monai.transforms.Compose]:

    img_keys = [f"img_{m}" for m in modalities]
    lab_keys = [f"lab_{m}" for m in modalities]
    pairs = list(zip(img_keys, lab_keys))

    # —— 共同的“确定性预处理” ——（读、对齐、重采样、尺寸统一、对齐断言、拼接）
    deterministic = [
        SafeLoadDICOMd(keys=img_keys),  # DICOM 用 SimpleITK 读取 + meta
        LoadImaged(keys=lab_keys, reader="NibabelReader", image_only=True),  # NIfTI
        EnsureChannelFirstd(keys=img_keys + lab_keys, channel_dim="no_channel"),
        Orientationd(keys=img_keys + lab_keys, axcodes=orientation_axcodes),
        Spacingd(
            keys=img_keys + lab_keys,
            pixdim=target_spacing,
            mode=tuple(["bilinear"] * len(img_keys) + ["nearest"] * len(lab_keys)),
        ),
        Resized(keys=img_keys, spatial_size=target_size, mode="trilinear"),
        Resized(keys=lab_keys, spatial_size=target_size, mode="nearest"),
        AssertPairAlignedd(
            img_keys=img_keys,
            lab_keys=lab_keys,
            atol_spacing=1e-4,
            atol_affine=1e-3,
            atol_dir=1e-4,
        ),  # 拼接前强校验
        ConcatItemsd(keys=img_keys, name="image", dim=0),  # (C,H,W,Z)
        ConcatItemsd(keys=lab_keys, name="label", dim=0),  # (C,H,W,Z)
        PreserveMetaD(img_keys, lab_keys),
        DeleteItemsd(keys=img_keys + lab_keys),
    ]

    # —— 训练增强（只在 train=True 时追加；空间类对 image+label，同步；强度类只对 image）——

    aug = [
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.3, max_k=3),
        RandAffined(
            keys=["image", "label"],
            prob=0.25,
            rotate_range=(0.1, 0.1, 0.1),
            scale_range=(0.1, 0.1, 0.1),
            translate_range=(4, 4, 2),
            mode=("bilinear", "nearest"),
            padding_mode="border",
        ),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
    ]

    tail = [
        EnsureTyped(keys=["image"], dtype=dtype_image),
        EnsureTyped(keys=["label"], dtype=dtype_label),
    ]

    return Compose(deterministic + aug + tail), Compose(deterministic + tail)


def get_Brats_transforms(
    config: EasyDict,
) -> Tuple[monai.transforms.Compose, monai.transforms.Compose]:
    train_transform = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=["image", "label"]),
            monai.transforms.EnsureChannelFirstd(keys="image"),
            monai.transforms.EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd_For_BraTS(
                keys=["label"], is2019=False
            ),
            monai.transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            monai.transforms.SpatialPadD(
                keys=["image", "label"],
                spatial_size=(255, 255, config.BraTS_loader.image_size),
                method="symmetric",
                mode="constant",
            ),
            monai.transforms.Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            monai.transforms.CenterSpatialCropD(
                keys=["image", "label"],
                roi_size=ensure_tuple_rep(config.BraTS_loader.image_size, 3),
            ),
            # monai.transforms.Resized(keys=["image", "label"], spatial_size=ensure_tuple_rep(config.model.image_size, 3)),
            monai.transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                num_samples=2,
                spatial_size=ensure_tuple_rep(config.BraTS_loader.image_size, 3),
                pos=1,
                neg=1,
                image_key="image",
                image_threshold=0,
            ),
            # monai.transforms.RandSpatialCropd(keys=["image", "label"], roi_size=config.model.image_size, random_size=False),
            monai.transforms.RandFlipd(
                keys=["image", "label"], prob=0.5, spatial_axis=0
            ),
            monai.transforms.RandFlipd(
                keys=["image", "label"], prob=0.5, spatial_axis=1
            ),
            monai.transforms.RandFlipd(
                keys=["image", "label"], prob=0.5, spatial_axis=2
            ),
            monai.transforms.NormalizeIntensityd(
                keys="image", nonzero=True, channel_wise=True
            ),
            monai.transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            monai.transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            monai.transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=["image", "label"]),
            monai.transforms.EnsureChannelFirstd(keys="image"),
            monai.transforms.EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd_For_BraTS(
                keys="label", is2019=False
            ),
            monai.transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            monai.transforms.Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            # monai.transforms.Resized(keys=["image", "label"], spatial_size=ensure_tuple_rep(config.model.image_size, 3)),
            monai.transforms.NormalizeIntensityd(
                keys="image", nonzero=True, channel_wise=True
            ),
        ]
    )
    return train_transform, val_transform


class MultiModalityDataset(monai.data.Dataset):
    def __init__(
        self,
        data,
        loadforms,
        transforms,
        over_label=False,
        over_add=0,
        use_class=True,
        data_choose="GCM",
    ):
        self.data = data
        self.transforms = transforms
        self.loadforms = loadforms
        self.over_label = over_label
        self.over_add = over_add
        self.use_class = use_class
        self.data_choose = data_choose

    def extract_and_resize(self, image, label, over_add=0):
        # 获取label中值为1的点的坐标
        indices = torch.nonzero(label[0])  # 去掉batch维度，找到非零索引

        # 找到最长的维度
        max_size = max(image[0].shape[0], image[0].shape[1], image[0].shape[2])

        # 计算各个维度上的 over_add，按比例缩小较短维度的扩展量
        over_add_x = round(over_add * (image[0].shape[0] / max_size))
        over_add_y = round(over_add * (image[0].shape[1] / max_size))
        over_add_z = round(over_add * (image[0].shape[2] / max_size))

        # 获取在每个维度上的最小和最大索引
        min_x, min_y, min_z = indices.min(dim=0).values.tolist()
        max_x, max_y, max_z = indices.max(dim=0).values.tolist()

        # 计算扩展后的坐标，并限制在合法范围内
        min_x = max(0, min_x - over_add_x)
        max_x = min(image[0].shape[0] - 1, max_x + over_add_x)
        min_y = max(0, min_y - over_add_y)
        max_y = min(image[0].shape[1] - 1, max_y + over_add_y)
        min_z = max(0, min_z - over_add_z)
        max_z = min(image[0].shape[2] - 1, max_z + over_add_z)

        # 切割image和label
        cropped_image = image[
            :, min_x : max_x + 1, min_y : max_y + 1, min_z : max_z + 1
        ]

        # 直接使用 interpolate 进行 resize 到 (1, 128, 128, 64)
        # resized_image = F.interpolate(cropped_image.unsqueeze(0), size=(128, 128, 64), mode='trilinear', align_corners=False).squeeze(0)
        resized_image = monai.transforms.Resize(
            spatial_size=(label.shape[1], label.shape[2], label.shape[3]),
            mode=("trilinear"),
        )(cropped_image)
        return resized_image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        combined_data = {}

        
        for i in range(0, len(item["image"])):
            # print('Loading ', item['image'][i])
            
            # print(f'Processing sample {item["image"][i]}')
            
            globals()[f"data_{i}"] = self.loadforms[i](
                {"image": item["image"][i], "label": item["label"][i]}
            )

            combined_data[f"model_{i}_image"] = globals()[f"data_{i}"]["image"]
            combined_data[f"model_{i}_label"] = globals()[f"data_{i}"]["label"]

            if self.over_label == True:
                imgae = self.extract_and_resize(
                    combined_data[f"model_{i}_image"],
                    combined_data[f"model_{i}_label"],
                    self.over_add,
                )
            else:
                imgae = combined_data[f"model_{i}_image"]
            combined_data[f"model_{i}_image"] = imgae

        images = []
        labels = []

        for i in range(0, len(item["image"])):
            images.append(combined_data[f"model_{i}_image"])
            labels.append(combined_data[f"model_{i}_label"])
            
        image_tensor = torch.cat(images, dim=0)
        label_tensor = torch.cat(labels, dim=0)
        result = {"image": image_tensor, "label": label_tensor}
        result = self.transforms(result)

        if self.use_class == True:
            if self.data_choose == "GCM":
                class_label = item["class_label"]
                PFS_label = item["PFS_label"]
                if class_label != 0:
                    class_label = 1
                return {
                    "image": result["image"],
                    "label": result["label"],
                    "class_label": torch.tensor(class_label).unsqueeze(0).long(),
                    "PFS_label": torch.tensor(PFS_label).unsqueeze(0).long(),
                }
            else:
                # pdl1_label = item["pdl1_label"]
                # if pdl1_label != 0:
                #     pdl1_label = 1
                m_label = item["m_label"]
                if m_label != 0:
                    m_label = 1
                return {
                    "image": result["image"],
                    "label": result["label"],
                    # "pdl1_label": torch.tensor(pdl1_label).unsqueeze(0).long(),
                    "m_label": torch.tensor(m_label).unsqueeze(0).long(),
                    "center": torch.tensor(item["center"]).unsqueeze(0).long(),
                }
        else:
            return {
                "image": result["image"],
                "label": result["label"],
                "center": result["center"],
            }


class DCMDataset(monai.data.Dataset):

    def __init__(self, data: List[Dict[str, Any]], transform=None):
        if not data:
            raise ValueError("Empty dataset.")
        # 固定模态顺序
        self.modalities = sorted(list(data[0]["image"].keys()))

        # 扁平化
        flat = []
        for s in data:
            item = {}
            for m in self.modalities:
                item[f"img_{m}"] = s["image"][m]  # list[str] of .dcm
                item[f"lab_{m}"] = s["label"][m]  # str of .nii
            flat.append(item)

        super().__init__(data=flat, transform=transform)


def split_list(data, ratios):
    # 计算每个部分的大小
    sizes = [math.ceil(len(data) * r) for r in ratios]

    # 调整大小以确保总大小与原列表长度匹配
    total_size = sum(sizes)
    if total_size != len(data):
        sizes[-1] -= total_size - len(data)

    # 分割列表
    start = 0
    parts = []
    for size in sizes:
        end = start + size
        parts.append(data[start:end])
        start = end

    return parts


def check_example(data, dcm=False):
    index = []
    for d in data:
        if dcm != True:
            num = d["image"][0].split("/")[-1].split(".")[0]
        else:
            num = d["image"][list(d["image"].keys())[0]][0].split("/")[-3]
        index.append(num)
    return index


def sort_keys_by_time(data: dict) -> list:
    def parse_time(t):
        if isinstance(t, datetime):
            return t
        if isinstance(t, (float, int)):
            return datetime.fromtimestamp(t)
        if not isinstance(t, str):
            raise ValueError(f"Unsupported time format: {t}")

        t = t.strip()
        time_formats = ["%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d  %H:%M:%S"]

        for fmt in time_formats:
            try:
                return datetime.strptime(t, fmt)
            except ValueError:
                continue

        raise ValueError(f"Unrecognized datetime format: {t}")

    return sorted(data.keys(), key=lambda k: parse_time(data[k][1]))


def split_examples_to_data(data, config, lack_flag=False, loding=False):
    def read_file_to_list(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            # 去除每行末尾的换行符
        lines = [line.strip() for line in lines]
        return lines

    def select_example_to_data(data, example_list):
        selected_data = []
        for d in data:
            num = d
            if num in example_list:
                selected_data.append(d)
        return selected_data

    def load_example_to_data(data, example_path, loding=False):
        data_list = read_file_to_list(example_path)
        print(f"Loading examples from {example_path}")
        if loding == True:
            data_list = select_example_to_data(data, data_list)
        return data_list

    if config.trainer.choose_dataset == "GCM":
        data_root = config.GCM_loader.root
    elif config.trainer.choose_dataset == "GCNC":
        data_root = config.GCNC_loader.root
    elif config.trainer.choose_dataset == "FS":
        data_root = config.FS_loader.root

    train_example = data_root + "/" + "train_examples.txt"
    val_example = data_root + "/" + "val_examples.txt"
    test_example = data_root + "/" + "test_examples.txt"

    train_data, val_data, test_data = (
        load_example_to_data(data, train_example, loding),
        load_example_to_data(data, val_example, loding),
        load_example_to_data(data, test_example, loding),
    )

    if lack_flag == True:
        train_data_lack, val_data_lack, test_data_lack = (
            load_example_to_data(data, train_example, loding),
            load_example_to_data(data, val_example, loding),
            load_example_to_data(data, test_example, loding),
        )
        return (
            train_data,
            val_data,
            test_data,
            train_data_lack,
            val_data_lack,
            test_data_lack,
        )

    return train_data, val_data, test_data


def get_dataloader_GCM(
    config: EasyDict,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    root = config.GCM_loader.root
    datapath = root + "/" + "ALL" + "/"
    use_models = config.GCM_loader.checkModels

    # data1: 腹膜转移分类; data2: 淋巴结同时序（手术）; data3: 淋巴结异时序（化疗后）
    data1, data2 = read_csv_for_GCM(config)
    if config.GCM_loader.task == "DS":
        use_data_dict = data1
    elif config.GCM_loader.task == "LM":
        use_data_dict = data2
        use_data_dict.update(data2)
    else:
        use_data_dict = data1

    # 按时间顺序划分数据集
    # use_data_list = sort_keys_by_time(use_data_dict)
    use_data_list = use_data_dict.keys()
    
    # 剔除不需要的病历号
    remove_list = config.GCM_loader.leapfrog
    use_data = [item for item in use_data_list if item not in remove_list]

    # 在use_data处划分数据，避免并行导致的读取问题
    if config.GCM_loader.fix_example != True:
        if config.GCM_loader.time_limit != True:
            random.shuffle(use_data)
            print("Random Loading!")
        train_use_data, val_use_data, test_use_data = split_list(
            use_data,
            [
                config.GCM_loader.train_ratio,
                config.GCM_loader.val_ratio,
                config.GCM_loader.test_ratio,
            ],
        )
        if config.GCM_loader.fusion == True:
            need_val_data = val_use_data + test_use_data
            val_use_data = need_val_data
            test_use_data = need_val_data
    else:
        train_use_data, val_use_data, test_use_data = split_examples_to_data(
            use_data, config, loding=True
        )

    # 加载MR数据
    train_data, _ = load_MR_dataset_images(
        datapath, train_use_data, use_models, use_data_dict, data_choose="GCM"
    )
    val_data, _ = load_MR_dataset_images(
        datapath, val_use_data, use_models, use_data_dict, data_choose="GCM"
    )
    test_data, _ = load_MR_dataset_images(
        datapath, test_use_data, use_models, use_data_dict, data_choose="GCM"
    )

    # data, _ = load_MR_dataset_images(datapath, use_data, use_models, use_data_dict)

    load_transform, train_transform, val_transform = get_GCM_transforms(config)

    # if config.GCM_loader.fix_example == True:
    #     train_data, val_data, test_data = split_examples_to_data(data, config)
    # else:
    #     # shuffle data for objective verification
    #     if config.GCM_loader.time_limit != True:
    #         random.shuffle(data)
    #         print('Random Loading!')

    #     train_data, val_data, test_data = split_list(data, [config.GCM_loader.train_ratio, config.GCM_loader.val_ratio, config.GCM_loader.test_ratio])

    #     # if not need test, can use fusion to fuse two data
    #     if config.GCM_loader.fusion == True:
    #         need_val_data = val_data + test_data
    #         val_data = need_val_data
    #         test_data = need_val_data

    train_example = check_example(train_data)
    val_example = check_example(val_data)
    test_example = check_example(test_data)

    train_dataset = MultiModalityDataset(
        data=train_data,
        over_label=config.GCM_loader.over_label,
        over_add=config.GCM_loader.over_add,
        loadforms=load_transform,
        transforms=train_transform,
        data_choose="GCM",
    )
    val_dataset = MultiModalityDataset(
        data=val_data,
        over_label=config.GCM_loader.over_label,
        over_add=config.GCM_loader.over_add,
        loadforms=load_transform,
        transforms=val_transform,
        data_choose="GCM",
    )
    test_dataset = MultiModalityDataset(
        data=test_data,
        over_label=config.GCM_loader.over_label,
        over_add=config.GCM_loader.over_add,
        loadforms=load_transform,
        transforms=val_transform,
        data_choose="GCM",
    )

    train_loader = monai.data.DataLoader(
        train_dataset,
        num_workers=config.GCM_loader.num_workers,
        batch_size=config.trainer.batch_size,
        shuffle=True,
    )
    val_loader = monai.data.DataLoader(
        val_dataset,
        num_workers=config.GCM_loader.num_workers,
        batch_size=config.trainer.batch_size,
        shuffle=False,
    )
    test_loader = monai.data.DataLoader(
        test_dataset,
        num_workers=config.GCM_loader.num_workers,
        batch_size=config.trainer.batch_size,
        shuffle=False,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        (train_example, val_example, test_example),
    )


def get_dataloader_GICC(
    config: EasyDict,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    root = config.GICC_loader.root
    datapath = root + "/" + "ALL" + "/"
    use_models = config.GICC_loader.checkModels
    # 读取数据集划分信息 / 去除不使用的样本
    train_d, val_d, test_d = read_csv_for_GICC(root)
    remove_list = config.GICC_loader.leapfrog
    train_dict = {k: v for k, v in train_d.items() if k not in remove_list}
    val_dict = {k: v for k, v in val_d.items() if k not in remove_list}
    test_dict = {k: v for k, v in test_d.items() if k not in remove_list}
    
    train_use_data = list(train_dict.keys())
    val_use_data = list(val_dict.keys())
    test_use_data = list(test_dict.keys())
    
    # 不需要划分数据，直接加载
    # 加载MR数据，形式上与GCM一致
    train_data, _ = load_MR_dataset_images(
        datapath, train_use_data, use_models, train_dict, data_choose="GCM"
    )
    val_data, _ = load_MR_dataset_images(
        datapath, val_use_data, use_models, val_dict, data_choose="GCM"
    )
    test_data, _ = load_MR_dataset_images(
        datapath, test_use_data, use_models, test_dict, data_choose="GCM"
    )
    
    # 和胃癌共用数据增强
    load_transform, train_transform, val_transform = get_GICC_transforms(config)
    
    train_example = check_example(train_data)
    val_example = check_example(val_data)
    test_example = check_example(test_data)

    train_dataset = MultiModalityDataset(
        data=train_data,
        loadforms=load_transform,
        transforms=train_transform,
        data_choose="GCM",
    )
    val_dataset = MultiModalityDataset(
        data=val_data,
        loadforms=load_transform,
        transforms=val_transform,
        data_choose="GCM",
    )
    test_dataset = MultiModalityDataset(
        data=test_data,
        loadforms=load_transform,
        transforms=val_transform,
        data_choose="GCM",
    )

    train_loader = monai.data.DataLoader(
        train_dataset,
        num_workers=config.GICC_loader.num_workers,
        batch_size=config.trainer.batch_size,
        shuffle=True,
        collate_fn=pad_list_data_collate,
    )
    val_loader = monai.data.DataLoader(
        val_dataset,
        num_workers=config.GICC_loader.num_workers,
        batch_size=config.trainer.batch_size,
        shuffle=False,
        collate_fn=pad_list_data_collate,
    )
    test_loader = monai.data.DataLoader(
        test_dataset,
        num_workers=config.GICC_loader.num_workers,
        batch_size=config.trainer.batch_size,
        shuffle=False,
        collate_fn=pad_list_data_collate,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        (train_example, val_example, test_example),
    )
    
    
    
    

def get_dataloader_GCNC(
    config: EasyDict,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    root = config.GCNC_loader.root
    datapath = root + "/" + "ALL" + "/"
    use_models = config.GCNC_loader.checkModels

    content_dict = read_csv_for_GCNC(config)

    use_data_list = list(content_dict.keys())

    remove_list = config.GCNC_loader.leapfrog
    use_data = [item for item in use_data_list if item not in remove_list]

    data, test_data, data_lack = load_MR_dataset_images(
        datapath,
        use_data,
        use_models,
        content_dict,
        data_choose="GCNC",
        test_center=config.GCNC_loader.test_center,
    )
    load_transform, train_transform, val_transform = get_GCNC_transforms(config)

    if config.GCNC_loader.fix_example == True:
        (train_data, val_data, _) = split_examples_to_data(
            data, config, lack_flag=False, loding=True
        )
    else:
        random.shuffle(data)
        print("Random Loading!")
        # TODO：使用第三中心数据作为测试，避免test数据划分
        train_data, val_data, _ = split_list(
            data,
            [
                config.GCNC_loader.train_ratio,
                config.GCNC_loader.val_ratio + config.GCNC_loader.test_ratio,
                0,
            ],
        )

        if config.GCNC_loader.fusion == True:
            need_val_data = val_data + test_data
            val_data = need_val_data
            test_data = need_val_data

    train_example = check_example(train_data)
    val_example = check_example(val_data)
    test_example = check_example(test_data)

    # train_lack_example = check_example(train_data_lack)
    # val_lack_example = check_example(val_data_lack)
    # test_lack_example = check_example(test_data_lack)

    train_dataset = MultiModalityDataset(
        data=train_data,
        over_label=config.GCNC_loader.over_label,
        over_add=config.GCNC_loader.over_add,
        loadforms=load_transform,
        transforms=train_transform,
        use_class=True,
        data_choose="GCNC",
    )
    val_dataset = MultiModalityDataset(
        data=val_data,
        over_label=config.GCNC_loader.over_label,
        over_add=config.GCNC_loader.over_add,
        loadforms=load_transform,
        transforms=val_transform,
        use_class=True,
        data_choose="GCNC",
    )
    test_dataset = MultiModalityDataset(
        data=test_data,
        over_label=config.GCNC_loader.over_label,
        over_add=config.GCNC_loader.over_add,
        loadforms=load_transform,
        transforms=val_transform,
        use_class=True,
        data_choose="GCNC",
    )

    train_loader = monai.data.DataLoader(
        train_dataset,
        num_workers=config.GCNC_loader.num_workers,
        batch_size=config.trainer.batch_size,
        shuffle=True,
    )
    val_loader = monai.data.DataLoader(
        val_dataset,
        num_workers=config.GCNC_loader.num_workers,
        batch_size=config.trainer.batch_size,
        shuffle=False,
    )
    test_loader = monai.data.DataLoader(
        test_dataset,
        num_workers=config.GCNC_loader.num_workers,
        batch_size=config.trainer.batch_size,
        shuffle=False,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        (train_example, val_example, test_example),
    )


def get_dataloader_FS(
    config: EasyDict,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    root = config.FS_loader.root
    datapath = root + "/" + "primary_data/data/MRI-Segments/"
    use_data_list = [
        d for d in os.listdir(datapath) if os.path.isdir(os.path.join(datapath, d))
    ]
    remove_list = config.FS_loader.leapfrog
    use_data = [item for item in use_data_list if item not in remove_list]
    data = load_MR_dcm_dataset_images(datapath, use_data, config.FS_loader.checkModels)
    if config.FS_loader.fix_example == True:
        (train_data, val_data, test_data) = split_examples_to_data(
            data, config, lack_flag=False, loding=True
        )
    else:
        random.shuffle(data)
        print("Random Loading!")
        # TODO：使用第三中心数据作为测试，避免test数据划分
        train_data, val_data, test_data = split_list(
            data,
            [
                config.FS_loader.train_ratio,
                config.FS_loader.val_ratio,
                config.FS_loader.test_ratio,
            ],
        )

        if config.FS_loader.fusion == True:
            need_val_data = val_data + test_data
            val_data = need_val_data
            test_data = need_val_data

    train_example = check_example(train_data, dcm=True)
    val_example = check_example(val_data, dcm=True)
    test_example = check_example(test_data, dcm=True)

    train_trasform, val_trasform = get_FS_transforms(
        modalities=config.FS_loader.checkModels,
        target_spacing=(1.0, 1.0, 1.0),
        target_size=config.FS_loader.target_size,
        dtype_image=torch.float32,
        dtype_label=torch.float32,
    )

    # train_dataset = DCMDataset(
    #     train_data,
    #     target_spacing=(1.0, 1.0, 1.0),  # 你需要的体素间距
    #     target_size=(128, 128, 64),      # 你需要的 W,H,Z
    #     dtype_image=torch.float32,
    #     dtype_label=torch.float32,       # 若是离散 mask 想要整数：改为 torch.long
    # )

    train_dataset = DCMDataset(
        train_data,
        transform=train_trasform,  # 若是离散 mask 想要整数：改为 torch.long
    )
    val_dataset = DCMDataset(
        val_data,
        transform=val_trasform,  # 若是离散 mask 想要整数：改为 torch.long
    )
    test_dataset = DCMDataset(
        test_data,
        transform=val_trasform,  # 若是离散 mask 想要整数：改为 torch.long
    )

    train_loader = monai.data.DataLoader(
        train_dataset,
        num_workers=config.FS_loader.num_workers,
        batch_size=config.trainer.batch_size,
        shuffle=True,
        pin_memory=True,
    )

    val_loader = monai.data.DataLoader(
        val_dataset,
        num_workers=config.FS_loader.num_workers,
        batch_size=config.trainer.batch_size,
        shuffle=False,
        pin_memory=True,
    )

    test_loader = monai.data.DataLoader(
        test_dataset,
        num_workers=config.FS_loader.num_workers,
        batch_size=config.trainer.batch_size,
        shuffle=False,
        pin_memory=True,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        (train_example, val_example, test_example),
    )


def get_dataloader_BraTS(
    config: EasyDict,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_images = load_brats2021_dataset_images(config.BraTS_loader.dataPath)
    train_transform, val_transform = get_Brats_transforms(config)
    
    data = train_images.copy()
    random.shuffle(data)
    
    n = len(data)

    train_end = int(n * config.BraTS_loader.train_ratio)
    val_end = int(n * (config.BraTS_loader.train_ratio + config.BraTS_loader.val_ratio))
    train_data = data[:train_end]
    val_data   = data[train_end:val_end]
    test_data  = data[val_end:]
    
    train_dataset = monai.data.Dataset(
        data=train_data,
        transform=train_transform,
    )
    val_dataset = monai.data.Dataset(
        data=val_data,
        transform=val_transform,
    )
    test_dataset = monai.data.Dataset(
        data=test_data,
        transform=val_transform,
    )

    train_loader = monai.data.DataLoader(
        train_dataset,
        num_workers=config.BraTS_loader.num_workers,
        batch_size=config.trainer.batch_size,
        shuffle=True,
    )

    val_loader = monai.data.DataLoader(
        val_dataset,
        num_workers=config.BraTS_loader.num_workers,
        batch_size=config.trainer.batch_size,
        shuffle=False,
    )
    test_loader = monai.data.DataLoader(
        test_dataset,
        num_workers=config.BraTS_loader.num_workers,
        batch_size=config.trainer.batch_size,
        shuffle=False,
    )


    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    config = EasyDict(
        yaml.load(
            open("/workspace/HWA/config.yml", "r", encoding="utf-8"),
            Loader=yaml.FullLoader,
        )
    )

    train_loader, val_loader, test_loader = get_dataloader_BraTS(config)
    

    
    
    train_count = 0
    val_count = 0
    test_count = 0
    
    for batch_data in train_loader:
        print(batch_data["image"].shape)
        print(batch_data["label"].shape)
        # print(batch_data["class_label"].shape)
        # print(batch_data["class_label"].shape)
        
        
        train_count += 1
        
    for batch_data in val_loader:
        print(batch_data["image"].shape)
        print(batch_data["label"].shape)
        # print(batch_data["class_label"].shape)
        # print(batch_data["PFS_label"].shape)
        val_count += 1
        
    for batch_data in test_loader:
        print(batch_data["image"].shape)
        print(batch_data["label"].shape)
        # print(batch_data["class_label"].shape)
        # print(batch_data["PFS_label"].shape)
        test_count += 1
    
    print(f"Train batches: {train_count}")
    print(f"val batches: {val_count}")
    print(f"test batches: {test_count}")
