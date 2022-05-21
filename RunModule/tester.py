import Options
import os
import glob
import torch
import torchvision.transforms as transforms

from Modeling.Generator import Gen

class test(Options.param):
    def __init__(self):
        super(test, self).__init__()

    def run(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'[device] : {device}')
        print('--------------------------------------------------------------------------------')

        # 1. Model Build
        num_blocks = 6 if self.size <= 256 else 8

        G_A2B = Gen(num_blocks).to(device)
        G_B2A = Gen(num_blocks).to(device)

        # 2. Load CKP
        checkpoint = torch.load(self.OUTPUT_CKP, map_location=device)
        G_A2B.load_state_dict(checkpoint["G_A2B_stae_dict"])
        G_B2A.load_state_dict(checkpoint["G_B2A_stae_dict"])

        G_A2B.eval()
        G_B2A.eval()

        # 3. Load DataSets
        transform_to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

        transforms_to_image = transforms.Compose(
            [
                transforms.Normalize(mean=(-1, -1, -1), std=(2, 2, 2)),
                transforms.ToPILImage(),
            ]
        )

        # 4. Setting Folder
        os.makedirs(f'{self.OUTPUT_TEST}/A2B', exist_ok=True)
        os.makedirs(f'{self.OUTPUT_TEST}/B2A', exist_ok=True)

        os.makedirs(f'{self.OUTPUT_TEST}/A2B_LOSS', exist_ok=True)
        os.makedirs(f'{self.OUTPUT_TEST}/B2A_LOSS', exist_ok=True)

        test_list = [["testA2B", G_A2B], ["testB2A", G_B2A]]

        # LOOP
        for folder_name, model in test_list:
            print(f'[Folder Name] : {folder_name}')
            image_paht_list = sorted(
                glob.glob(os.path.join(f'{self.ROOT}/A/*'))
            )

            # for idx, image_path in enumerate(image_paht_list):
            #