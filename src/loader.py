import os
import math
import logging
import torch
import monai
import random
import pandas as pd
import SimpleITK as sitk
from easydict import EasyDict
from typing import Tuple

sitk.ProcessObject.SetGlobalWarningDisplay(False)
LOGGER = logging.getLogger(__name__)
from monai.transforms import (
    LoadImaged,
    ScaleIntensityRanged,
    EnsureChannelFirstd,
    RandFlipd,
    ToTensord,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Resized,
)


def read_csv_for_GCM(config):
    csv_path = config.GCM_loader.root + "/" + "Classification.xlsx"
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

    content_dict1 = {}
    content_dict2 = {}
    for _, row in df1.iterrows():
        key = row.iloc[1]
        values = row.iloc[2]
        time = row.iloc[4]
        content_dict1[key] = [values, time]

    for _, row in df2.iterrows():
        key = row.iloc[1]
        values1 = row.iloc[3]
        values2 = row.iloc[4]
        time = row.iloc[4]
        content_dict2[key] = [values1, values2, time]

    return content_dict1, content_dict2


def load_MR_dataset_images(root, use_data, use_models, use_data_dict=None):
    use_data_dict = use_data_dict or {}
    images_path = os.listdir(root)
    images_list = []
    images_lack_list = []

    for path in use_data:
        path = str(path)
        if path in images_path:
            models = os.listdir(root + "/" + path + "/")
        else:
            LOGGER.warning("Case not found: %s under %s", path, root)
            continue
        lack_flag = False
        lack_model_flag = False
        image = []
        label = []

        for model in models:
            if "T2" in model:
                check_model = "T2_FS"
            elif model == "CT1":
                check_model = "T1+C"
            elif model == "T1+c":
                check_model = "T1+C"
            else:
                check_model = model

            if check_model in use_models:
                if not os.path.exists(root + "/" + path + "/" + model):
                    LOGGER.warning("Missing modality directory: case=%s, modality=%s", path, model)
                    lack_model_flag = True
                    break
                elif not os.path.exists(
                    root + "/" + path + "/" + model + "/" + path + ".nii.gz"
                ):
                    LOGGER.warning("Missing image file: case=%s, modality=%s", path, model)
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
                        LOGGER.warning("Missing label file: case=%s, modality=%s", path, model)
                        lack_flag = True
                else:
                    label.append(
                        root + "/" + path + "/" + model + "/" + path + "seg.nii.gz"
                    )

        if image == [] or len(image) < len(use_models):
            LOGGER.warning("Incomplete modality set: case=%s", path)
            lack_model_flag = True
        
        if not lack_flag and not lack_model_flag:
            if use_data_dict:
                images_list.append(
                    {
                        "image": image,
                        "label": label,
                        "class_label": use_data_dict[path][0],
                        "PFS_label": use_data_dict[path][1],
                    }
                )
            else:
                images_list.append({"image": image, "label": label})
        elif not lack_model_flag:
            if use_data_dict:
                images_lack_list.append(
                    {
                        "image": image,
                        "label": label,
                        "class_label": use_data_dict[path][0],
                        "PFS_label": use_data_dict[path][1],
                    }
                )
            else:
                images_lack_list.append({"image": image, "label": label})

    return images_list, images_lack_list


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
                        keys=["image"],
                        a_min=model_scale[0],
                        a_max=model_scale[1],
                        b_min=0.0,
                        b_max=1.0,
                        clip=True,
                    ),
                    ToTensord(keys=["image", "label"]),
                ]
            )
        )

    train_transform = monai.transforms.Compose(
        [
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
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
        cache_loaded=False,
    ):
        self.data = data
        self.transforms = transforms
        self.loadforms = loadforms
        self.over_label = over_label
        self.over_add = over_add
        self.use_class = use_class
        self.cache_loaded = bool(cache_loaded)
        self._loaded_cache = {}

    def extract_and_resize(self, image, label, over_add=0):
        indices = torch.nonzero(label[0])
        if indices.numel() == 0:
            return image, label

        max_size = max(image[0].shape[0], image[0].shape[1], image[0].shape[2])

        over_add_x = round(over_add * (image[0].shape[0] / max_size))
        over_add_y = round(over_add * (image[0].shape[1] / max_size))
        over_add_z = round(over_add * (image[0].shape[2] / max_size))

        min_x, min_y, min_z = indices.min(dim=0).values.tolist()
        max_x, max_y, max_z = indices.max(dim=0).values.tolist()

        min_x = max(0, min_x - over_add_x)
        max_x = min(image[0].shape[0] - 1, max_x + over_add_x)
        min_y = max(0, min_y - over_add_y)
        max_y = min(image[0].shape[1] - 1, max_y + over_add_y)
        min_z = max(0, min_z - over_add_z)
        max_z = min(image[0].shape[2] - 1, max_z + over_add_z)

        cropped_image = image[
            :, min_x : max_x + 1, min_y : max_y + 1, min_z : max_z + 1
        ]
        cropped_label = label[
            :, min_x : max_x + 1, min_y : max_y + 1, min_z : max_z + 1
        ]

        target_spatial_size = (label.shape[1], label.shape[2], label.shape[3])
        resized_image = monai.transforms.Resize(
            spatial_size=target_spatial_size,
            mode=("trilinear"),
        )(cropped_image)
        resized_label = monai.transforms.Resize(
            spatial_size=target_spatial_size,
            mode=("nearest"),
        )(cropped_label)
        return resized_image, resized_label

    def __len__(self):
        return len(self.data)

    def _load_modalities(self, idx, item):
        if self.cache_loaded and idx in self._loaded_cache:
            cached = self._loaded_cache[idx]
            return {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in cached.items()
            }

        combined_data = {}
        for i in range(0, len(item["image"])):
            data_i = self.loadforms[i]({"image": item["image"][i], "label": item["label"][i]})
            image_i = data_i["image"]
            label_i = data_i["label"]

            if self.over_label:
                image_i, label_i = self.extract_and_resize(image_i, label_i, self.over_add)

            combined_data[f"model_{i}_image"] = image_i
            combined_data[f"model_{i}_label"] = label_i

        if self.cache_loaded:
            self._loaded_cache[idx] = {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in combined_data.items()
            }
        return combined_data

    def __getitem__(self, idx):
        item = self.data[idx]

        combined_data = self._load_modalities(idx, item)

        images = []
        labels = []

        for i in range(0, len(item["image"])):
            images.append(combined_data[f"model_{i}_image"])
            labels.append(combined_data[f"model_{i}_label"])
            
        image_tensor = torch.cat(images, dim=0)
        label_tensor = torch.cat(labels, dim=0)
        result = {"image": image_tensor, "label": label_tensor}
        result = self.transforms(result)

        if self.use_class:
            class_label = item["class_label"]
            pfs_label = item["PFS_label"]
            if class_label != 0:
                class_label = 1
            return {
                "image": result["image"],
                "label": result["label"],
                "class_label": torch.tensor(class_label).unsqueeze(0).long(),
                "PFS_label": torch.tensor(pfs_label).unsqueeze(0).long(),
            }
        else:
            return {"image": result["image"], "label": result["label"]}


def split_list(data, ratios):
    sizes = [math.ceil(len(data) * r) for r in ratios]

    total_size = sum(sizes)
    if total_size != len(data):
        sizes[-1] -= total_size - len(data)

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
        if not dcm:
            num = d["image"][0].split("/")[-1].split(".")[0]
        else:
            num = d["image"][list(d["image"].keys())[0]][0].split("/")[-3]
        index.append(num)
    return index


def split_examples_to_data(data, config, lack_flag=False, loading=False):
    def _cfg_value(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def read_file_to_list(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
        lines = [line.strip() for line in lines]
        return lines

    def select_example_to_data(data, example_list):
        selected_data = []
        for d in data:
            num = d
            if num in example_list:
                selected_data.append(d)
        return selected_data

    def load_example_to_data(data, example_path, loading=False):
        data_list = read_file_to_list(example_path)
        LOGGER.info("Loading split file: %s", example_path)
        if loading:
            data_list = select_example_to_data(data, data_list)
        return data_list

    data_root = config.GCM_loader.root
    train_example = _cfg_value(
        config.GCM_loader,
        "train_examples_path",
        data_root + "/" + "train_examples.txt",
    )
    val_example = _cfg_value(
        config.GCM_loader,
        "val_examples_path",
        data_root + "/" + "val_examples.txt",
    )
    test_example = _cfg_value(
        config.GCM_loader,
        "test_examples_path",
        data_root + "/" + "test_examples.txt",
    )

    train_data, val_data, test_data = (
        load_example_to_data(data, train_example, loading),
        load_example_to_data(data, val_example, loading),
        load_example_to_data(data, test_example, loading),
    )

    if lack_flag:
        train_data_lack, val_data_lack, test_data_lack = (
            load_example_to_data(data, train_example, loading),
            load_example_to_data(data, val_example, loading),
            load_example_to_data(data, test_example, loading),
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

    data1, data2 = read_csv_for_GCM(config)
    if config.GCM_loader.task == "DS":
        use_data_dict = data1
    elif config.GCM_loader.task == "LM":
        use_data_dict = data2
    else:
        use_data_dict = data1

    use_data_list = use_data_dict.keys()
    
    remove_list = config.GCM_loader.leapfrog
    use_data = [item for item in use_data_list if item not in remove_list]

    if not config.GCM_loader.fix_example:
        if not config.GCM_loader.time_limit:
            random.shuffle(use_data)
            LOGGER.info("Using randomized split assignment.")
        train_use_data, val_use_data, test_use_data = split_list(
            use_data,
            [
                config.GCM_loader.train_ratio,
                config.GCM_loader.val_ratio,
                config.GCM_loader.test_ratio,
            ],
        )
        if config.GCM_loader.fusion:
            need_val_data = val_use_data + test_use_data
            val_use_data = need_val_data
            test_use_data = need_val_data
    else:
        train_use_data, val_use_data, test_use_data = split_examples_to_data(
            use_data, config, loading=True
        )

    train_data, _ = load_MR_dataset_images(datapath, train_use_data, use_models, use_data_dict)
    val_data, _ = load_MR_dataset_images(datapath, val_use_data, use_models, use_data_dict)
    test_data, _ = load_MR_dataset_images(datapath, test_use_data, use_models, use_data_dict)

    load_transform, train_transform, val_transform = get_GCM_transforms(config)


    train_example = check_example(train_data)
    val_example = check_example(val_data)
    test_example = check_example(test_data)

    def _split_over_label(split: str) -> bool:
        key = f"{split}_over_label"
        if hasattr(config.GCM_loader, key):
            return bool(getattr(config.GCM_loader, key))
        return bool(config.GCM_loader.over_label)

    def _split_over_add(split: str) -> int:
        key = f"{split}_over_add"
        if hasattr(config.GCM_loader, key):
            return int(getattr(config.GCM_loader, key))
        return int(config.GCM_loader.over_add)

    train_dataset = MultiModalityDataset(
        data=train_data,
        over_label=_split_over_label("train"),
        over_add=_split_over_add("train"),
        loadforms=load_transform,
        transforms=train_transform,
        cache_loaded=bool(getattr(config.GCM_loader, "cache_loaded", False)),
    )
    val_dataset = MultiModalityDataset(
        data=val_data,
        over_label=_split_over_label("val"),
        over_add=_split_over_add("val"),
        loadforms=load_transform,
        transforms=val_transform,
        cache_loaded=bool(getattr(config.GCM_loader, "cache_loaded", False)),
    )
    test_dataset = MultiModalityDataset(
        data=test_data,
        over_label=_split_over_label("test"),
        over_add=_split_over_add("test"),
        loadforms=load_transform,
        transforms=val_transform,
        cache_loaded=bool(getattr(config.GCM_loader, "cache_loaded", False)),
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
