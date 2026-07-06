import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader

from util import load_mappings, load_features, load_truths

class EndoDataset(Dataset):
    def __init__(self, features, labels, window=64):
        self.x = torch.from_numpy(features).float()
        self.y = torch.tensor(labels, dtype=torch.long)
        self.w = window

    def __len__(self):
        return len(self.x) - self.w + 1

    def __getitem__(self, idx):
        x = self.x[idx:idx + self.w]
        y = self.y[idx + self.w - 1]
        return x, y


MAPPING_PATH = "data/Endo_Project/mapping0.txt"
FEATURES_DIR = "data/Endo_Project/features"
TRUTHS_DIR = "data/Endo_Project/groundTruth"

video_names = [
    "export1", "export2", "export3", "export4",
    "export5", "export6", "export7-480p", "export8",
    "export9", "export10", "export11", "export12", 
    "export13", "export14", "export15-480P", "export16-480P", 
    "export17", "export18", "export19", "export20",
]

TRAIN_IDX = list(range(0, 16))
TEST_IDX = list(range(16, 20)) # [17]


def build_dataset(indices: list[int], window: int = 64) -> Dataset:
    mappings = load_mappings(MAPPING_PATH)

    features, truths = [], []
    # 暂时复用 MS-TCN2 训练数据
    for name in video_names:
        features.append(load_features(FEATURES_DIR, name))
        truths.append(load_truths(TRUTHS_DIR, name, mappings=mappings))

    return ConcatDataset([
        EndoDataset(features[i], truths[i], window) for i in indices
    ])

def build_dataloader(config: dict, mode: str) -> DataLoader:
    if mode == "train":
        dataset = build_dataset(TRAIN_IDX, config["window"])
    elif mode == "test":
        dataset = build_dataset(TEST_IDX, config["window"])
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=config["shuffle"])


# # Label Studio 数据加载示例
# for id in [50, 51, 52, 53, 54, 55, 56, 58, 59]:
#     print(id)
#     feats, trs = load_data_json("path to LS json", id)
#     features.append(feats)
#     truths.append([mappings[tr] for tr in trs])