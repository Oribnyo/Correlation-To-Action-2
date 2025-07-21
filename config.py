# config.py
# Shared configuration for TFT model

import os

class TFTConfig:
    def __init__(self):
        self.INPUT_WINDOW = 96
        self.OUTPUT_HORIZON = 30
        self.HIDDEN_SIZE = 32
        self.LSTM_LAYERS = 2
        self.NUM_ATTENTION_HEADS = 4
        self.DROPOUT = 0.1

        self.BATCH_SIZE = 64          # Reduced for stability
        self.MAX_EPOCHS = 30
        self.LEARNING_RATE = 1e-4     # Reduced for better convergence
        self.WEIGHT_DECAY = 1e-4

        self.TARGET = "Sensor 1"
        self.QUANTILES = [0.1, 0.5, 0.9]
        self.VALIDATION_SPLIT = 0.2

        self.PROC_DIR = "data/processed"
        self.RESULTS_DIR = "results"
        self.LOG_PATH = "log.txt"
        self.MODEL_SAVE_PATH = os.path.join(self.RESULTS_DIR, "tft_model.pkl")
        self.METRICS_FILE = os.path.join(self.RESULTS_DIR, "metrics_tft.csv") 