import torch
from torch.utils.data import Dataset
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, TrainingArguments, Trainer
from PIL import Image
import numpy as np
from glob import glob

print(f"Device: CPU — starting overnight training (~6 hours)")

LOVEDA_TO_UAE = {0: 255, 1: 3, 2: 3, 3: 2, 4: 1, 5: 0, 6: 0}

class SimpleDataset(Dataset):
    def __init__(self, images, masks, processor):
        self.images = images
        self.masks = masks
        self.processor = processor
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        mask = np.array(Image.open(self.masks[idx]))
        mask_remapped = np.full_like(mask, 255)
        for old_id, new_id in LOVEDA_TO_UAE.items():
            mask_remapped[mask == old_id] = new_id
        encoded = self.processor(image, Image.fromarray(mask_remapped), return_tensors="pt")
        return {
            "pixel_values": encoded["pixel_values"].squeeze(),
            "labels": encoded["labels"].squeeze()
        }

processor = SegformerImageProcessor(
    do_resize=True, size={"height": 512, "width": 512},
    do_normalize=True, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225]
)

train_imgs = sorted(glob("Train/Rural/images_png/*.png")) + sorted(glob("Train/Urban/images_png/*.png"))
train_masks = sorted(glob("Train/Rural/masks_png/*.png")) + sorted(glob("Train/Urban/masks_png/*.png"))

print(f"Train: {len(train_imgs)}")

train_dataset = SimpleDataset(train_imgs, train_masks, processor)

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b0", num_labels=4, ignore_mismatched_sizes=True
)

training_args = TrainingArguments(
    output_dir="backend/models/segformer-b0-checkpoints",
    num_train_epochs=5,
    per_device_train_batch_size=1,
    learning_rate=6e-5,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_steps=50,
    save_total_limit=3,
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=0,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=lambda x: {
        "pixel_values": torch.stack([i["pixel_values"] for i in x]),
        "labels": torch.stack([i["labels"] for i in x]),
    },
)

print("Starting CPU training...")
trainer.train()

final_path = "backend/models/segformer-b0-final"
trainer.save_model(final_path)
processor.save_pretrained(final_path)
print(f"✅ Saved to {final_path}")
