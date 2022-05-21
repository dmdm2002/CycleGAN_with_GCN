class param(object):
    def __init__(self):
        # Path
        self.ROOT = '/content/drive/MyDrive/ColabNotebooks/2th'
        self.DATASET_PATH = f'{self.ROOT}/dataset/original'
        self.OUTPUT_CKP = f'{self.ROOT}/train_sequence/ckp'
        self.OUTPUT_SAMPLE = f'{self.ROOT}/train_sequence/sampling'
        self.OUTPUT_TEST = ''
        self.OUTPUT_LOSS = ''
        self.CKP_LOAD = True

        # Data
        self.DATA_STYPE = ['A', 'B']
        self.SIZE = 224
        self.POOL_SIZE = 50

        # Train or Test
        self.EPOCH = 300
        self.LR = 2e-4
        self.LAMDA_CYCLE = 10
        self.LAMDA_ID = 0.5
        self.BATCHSZ = 2

        # Handler
        # run_type 0 : train, 1 : test
        self.run_type = 0
