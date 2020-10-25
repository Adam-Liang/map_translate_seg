class Configuration():
    def __init__(self):
        self.MODEL_NAME = 'deeplabv3plus'
        self.MODEL_BACKBONE = 'xception'
        self.MODEL_OUTPUT_STRIDE = 16
        self.MODEL_ASPP_OUTDIM = 256
        self.MODEL_SHORTCUT_DIM = 48
        self.MODEL_SHORTCUT_KERNEL = 1
        self.MODEL_NUM_CLASSES = 8
        self.MODEL_AUX_OUT = 4
        self.TRAIN_BN_MOM = 0.0003