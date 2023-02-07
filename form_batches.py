"""module for creating dataloaing"""
import config
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from read_data import DataCollector

class CaptchaDataset(Dataset):
    """
    class for preparing data
    """
    def __init__(self, filenames):
        self.image_filenames = filenames

    def __len__(self) -> int:
        """
        :return: account image's files
        """
        return len(self.image_filenames)

    def __getitem__(self, item):
        """
        get opportunity to choose by index
        """
        filename = self.image_filenames[item]
        file_pth = os.path.join(config.DATA_PATH, filename)
        img = Image.open(file_pth).convert('RGB')
        img = self.transform(img)
        label = filename.split('/')[1].split('.')[0]
        return img, label

    def transform(self, img):
        """
        transform images
        """
        transform_ops = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])
        return transform_ops(img)



if __name__=='__main__':
    reader = DataCollector()
    train_img, test_img = reader.get_files()
    train_dataset = CaptchaDataset(train_img)
    test_dataset = CaptchaDataset(test_img)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE,
                              shuffle=True)
    test_loader = DataLoader(test_dataset,
                              batch_size=config.BATCH_SIZE,
                              shuffle=False)
    print(f'{len(train_loader)} batches in the train_loader')
    print(f'{len(test_loader)} batches in the test_loader')
