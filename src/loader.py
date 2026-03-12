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

    load_transform, train_transform, val_transform = get_GCM_transforms(config)


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




if __name__ == "__main__":
    config = EasyDict(
        yaml.load(
            open("/workspace/HWA/config.yml", "r", encoding="utf-8"),
            Loader=yaml.FullLoader,
        )
    )
    
    train_loader, val_loader, test_loader, _ = get_dataloader_GCM(config)
    
    train_count = 0
    val_count = 0
    test_count = 0
    
    for batch_data in train_loader:
        print(batch_data["image"].shape)
        print(batch_data["label"].shape)
        print(batch_data["class_label"].shape)  # batch size, 1
        print(batch_data["class_label"].max())
        print(batch_data["class_label"].shape)
        print(batch_data["class_label"].max())
        print(batch_data["PFS_label"].shape)
        print(batch_data["PFS_label"].max())
        
        train_count += 1
        
    for batch_data in val_loader:
        print(batch_data["image"].shape)
        print(batch_data["label"].shape)
        print(batch_data["class_label"].shape)
        print(batch_data["class_label"].max())
        print(batch_data["PFS_label"].shape)
        print(batch_data["PFS_label"].max())
        val_count += 1
        
    for batch_data in test_loader:
        print(batch_data["image"].shape)
        print(batch_data["label"].shape)
        print(batch_data["class_label"].shape)
        print(batch_data["class_label"].max())
        print(batch_data["PFS_label"].shape)
        print(batch_data["PFS_label"].max())
        test_count += 1
    
    print(f"Train batches: {train_count}")
    print(f"val batches: {val_count}")
    print(f"test batches: {test_count}")
