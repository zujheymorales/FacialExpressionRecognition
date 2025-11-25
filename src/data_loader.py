# src/data_loader.py

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# consistent class order for labels
EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

def get_data_generators(
    data_dir="../data/train",
    img_size=(48, 48),
    batch_size=64,
    val_split=0.2,
    augment=False
):
    """
    Returns (train_gen, val_gen) ImageDataGenerators.

    data_dir  : path to folder containing subfolders per emotion.
    img_size  : (height, width) of images.
    batch_size: images per batch.
    val_split : fraction of data for validation.
    augment   : whether to use data augmentation on the training set.
    """

    if augment:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            validation_split=val_split,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
        )
    else:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            validation_split=val_split,
        )

    # validation should not be augmented
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=val_split,
    )

    train_gen = train_datagen.flow_from_directory(
        directory=data_dir,
        target_size=img_size,
        color_mode="grayscale",
        classes=EMOTION_CLASSES,  # fix label order
        class_mode="categorical",
        batch_size=batch_size,
        subset="training",
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_directory(
        directory=data_dir,
        target_size=img_size,
        color_mode="grayscale",
        classes=EMOTION_CLASSES,
        class_mode="categorical",
        batch_size=batch_size,
        subset="validation",
        shuffle=False,  # easier for evaluation
    )

    return train_gen, val_gen
