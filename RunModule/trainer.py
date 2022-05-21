import Options
import os
import re

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from Modeling.Generator import Gen
from Modeling.Discrminator import disc
from utils.DataLoader import Loader
from utils.func import ImagePool
from utils.Displayer import LossDisplayer


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class train(Options.param):
    def __init__(self):
        super(train, self).__init__()
        # self.writer = SummaryWriter()

    def init_weight(self, module):
        class_name = module.__class__.__name__

        if class_name.find("Conv") != -1:
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif class_name.find("BatchNorm2d") != -1:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant(module.bias.data, 0.0)

    def sampling(self, output, transform_to_image, name, epoch, type):
        os.makedirs(f"{self.OUTPUT_SAMPLE}/sample_{type}/{epoch}", exist_ok=True)
        output = transform_to_image(output.squeeze())
        output.save(f'{self.OUTPUT_SAMPLE}/sample_{type}/{epoch}/{name}_{type}.jpg')

    def run(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        print(f'[device] : {device}')
        print('--------------------------------------------------------------------------------')
        # writer = SummaryWriter()

        # 1. Model Build
        num_blocks = 6 if self.SIZE <= 256 else 8

        netG_A2B = Gen(num_blocks).to(device)
        netG_B2A = Gen(num_blocks).to(device)

        netD_A = disc().to(device)
        netD_B = disc().to(device)

        os.makedirs(f"{self.OUTPUT_CKP}/PROPOSED_GAN", exist_ok=True)
        os.makedirs(f"{self.OUTPUT_SAMPLE}/sample_A2B", exist_ok=True)
        os.makedirs(f"{self.OUTPUT_SAMPLE}/sample_A2B2A", exist_ok=True)

        # 2. Load CKP
        if self.CKP_LOAD:
            ckp = torch.load(self.OUTPUT_CKP, map_location=device)
            netG_A2B.load_state_dict(ckp["netG_A2B_state_dict"])
            netG_B2A.load_state_dict(ckp["netG_B2A_state_dict"])
            netD_A.load_state_dict(ckp["netD_A_state_dict"])
            netD_B.load_state_dict(ckp["netD_B_state_dict"])
            epoch = ckp["epoch"]
        else:
            netG_A2B.apply(self.init_weight)
            netG_B2A.apply(self.init_weight)
            netD_A.apply(self.init_weight)
            netD_B.apply(self.init_weight)

        netG_A2B.train()
        netG_B2A.train()
        netD_A.train()
        netD_B.train()

        # 3. DataLoader
        transform = transforms.Compose(
            [
                transforms.Resize((self.SIZE, self.SIZE)),
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

        dataset = Loader(self.DATASET_PATH, self.DATA_STYPE, transform)
        dataloader = DataLoader(dataset=dataset, batch_size=self.BATCHSZ, shuffle=True)
        # dataset_name = os.path.basename(self.DATASET_PATH)

        pool_fake_A = ImagePool(self.POOL_SIZE)
        pool_fake_B = ImagePool(self.POOL_SIZE)

        # 4. LOSS
        criterion_cycle = nn.L1Loss()
        criterion_identity = nn.L1Loss()
        critierion_GAN = nn.MSELoss()

        disp = LossDisplayer(["G_GAN", "G_recon", "D"])
        summary = SummaryWriter()

        optim_G = optim.Adam(
            list(netG_A2B.parameters()) + list(netG_B2A.parameters()),
            lr = self.LR,
            betas=(0.5, 0.999)
        )
        optim_D_A = optim.Adam(netD_A.parameters(), lr = self.LR)
        optim_D_B = optim.Adam(netD_B.parameters(), lr=self.LR)

        lr_lambda = lambda epoch: 1-((epoch) // 100) / (self.EPOCH / 100)
        scheduler_G = optim.lr_scheduler.LambdaLR(optimizer=optim_G, lr_lambda=lr_lambda)
        scheduler_D_A = optim.lr_scheduler.LambdaLR(
            optimizer=optim_D_A, lr_lambda=lr_lambda
        )
        scheduler_D_B = optim.lr_scheduler.LambdaLR(
            optimizer=optim_D_B, lr_lambda=lr_lambda
        )

        for epoch in range(self.EPOCH):
            print(f"|| Now Epoch : [{epoch}/{self.EPOCH}] ||")

            for idx, (real_A, real_B, name) in enumerate(dataloader):
                real_A = real_A.to(device)
                real_B = real_B.to(device)

                #Foard Model
                # A -> B
                fake_B = netG_A2B(real_A)
                # B -> A
                fake_A = netG_B2A(real_B)

                # A -> B -> A
                cycle_A = netG_B2A(fake_B)
                # B -> A -> B
                cycle_B = netG_A2B(fake_A)

                # Identity mapping
                same_A = netG_B2A(real_A)
                same_B = netG_A2B(real_B)

                disc_fake_A = netD_A(fake_A)
                disc_fake_B = netD_B(fake_B)

                # Calculate and backward generator
                loss_cycle_A = criterion_cycle(cycle_A, real_A)
                loss_cycle_B = criterion_cycle(cycle_B, real_B)
                loss_GAN_A = critierion_GAN(disc_fake_A, torch.ones_like(disc_fake_A))
                loss_GAN_B = critierion_GAN(disc_fake_B, torch.ones_like(disc_fake_B))

                loss_id_A = criterion_identity(same_A, real_A)
                loss_id_B = criterion_identity(same_B, real_B)

                loss_G = (
                    self.LAMDA_CYCLE * (loss_cycle_A + loss_cycle_B)
                    + (loss_GAN_A + loss_GAN_B)
                    + self.LAMDA_ID * self.LAMDA_CYCLE * (loss_id_A + loss_id_B)
                )

                optim_G.zero_grad()
                loss_G.backward()
                optim_G.step()

                # Calculate and backward discriminator
                disc_real_A = netD_A(real_A)
                disc_fake_A = netD_A(pool_fake_A.query(fake_A))

                loss_D_A = 0.5 * (
                    critierion_GAN(disc_real_A, torch.ones_like(disc_real_A))
                    + critierion_GAN(disc_fake_A, torch.zeros_like(disc_fake_A))
                )

                optim_D_A.zero_grad()
                loss_D_A.backward()
                optim_D_A.step()

                disc_real_B = netD_B(real_B)
                disc_fake_B = netD_B(pool_fake_B.query(fake_B))

                loss_D_B = 0.5 * (
                    critierion_GAN(disc_real_B, torch.ones_like(disc_real_B))
                    + critierion_GAN(disc_fake_B, torch.zeros_like(disc_fake_B))
                )

                optim_D_B.zero_grad()
                loss_D_B.backward()
                optim_D_B.step()

                # Record loss
                loss_G_GAN = loss_GAN_A + loss_GAN_B
                loss_G_cycle = loss_G - loss_G_GAN
                loss_D = loss_D_A + loss_D_B

                print(f'[{idx}/{len(dataloader)}] -> || [Loss Gen GAN : {loss_G_GAN}]  |  [Loss Gen Cycle : {loss_G_cycle}]  |  [Loss Disc : {loss_D}] ||')
                disp.record([loss_G_GAN, loss_G_cycle, loss_D])

                if idx % 100 == 0:
                    name = name[0].split("\\")[-1]
                    name = re.compile(".jpg").sub('', name)

                    self.sampling(fake_B[0], transforms_to_image, name, epoch, "A2B")
                    self.sampling(cycle_A[0], transforms_to_image, name, epoch, 'A2B2A')

            # Step Scheduler
            scheduler_G.step()
            scheduler_D_A.step()
            scheduler_D_B.step()

            # Record and display loss
            avg_losses = disp.get_avg_losses()
            summary.add_scalar("loss_G_GAN", avg_losses[0], epoch)
            summary.add_scalar("loss_G_cycle", avg_losses[1], epoch)
            summary.add_scalar("loss_D", avg_losses[2], epoch)

            torch.save(
                {
                    "netG_A2B_state_dict": netG_A2B.state_dict(),
                    "netG_B2A_state_dict": netG_B2A.state_dict(),
                    "netD_A_state_dict": netD_A.state_dict(),
                    "netD_B_state_dict": netD_B.state_dict(),
                    "epoch": epoch,
                },
                os.path.join(f"{self.OUTPUT_CKP}/PROPOSED_GAN", f"{epoch}.pth"),
            )