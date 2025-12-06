from data_loader import get_data_generators
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
data_dir = os.path.join(PROJECT_ROOT, "data", "train")

train_gen, val_gen = get_data_generators(
    data_dir=data_dir,
    img_size=(48, 48),
    batch_size=64,
    val_split=0.2,
    augment=False,
)

print("Train samples:", train_gen.samples)
print("Val samples:", val_gen.samples)
print("Classes:", train_gen.class_indices)
