"""Transfer-learning training script for RicePlantDiseaseDetector.

Usage (PowerShell):
  python .\backend\train_transfer.py --train_dir data/train --val_dir data/val --epochs 20

Features:
- MobileNetV2 backbone (ImageNet pretrained)
- Data augmentation via ImageDataGenerator
- Class-weighting based on training class counts
- Train head first, then optional fine-tune of last N layers
- Saves model to `model/retrained_mobilenetv2.h5` and history JSON
"""
import argparse
import json
import os
from math import ceil

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def compute_class_weights(labels):
    # labels: array-like of ints
    classes, counts = np.unique(labels, return_counts=True)
    total = labels.shape[0]
    num_classes = classes.shape[0]
    # simple balanced weights: total / (num_classes * count)
    weights = {int(cls): float(total) / (num_classes * int(cnt)) for cls, cnt in zip(classes, counts)}
    return weights


def build_model(img_size=(224, 224), num_classes=4, dropout=0.3):
    base = MobileNetV2(input_shape=(*img_size, 3), include_top=False, weights='imagenet')
    base.trainable = False
    inputs = layers.Input(shape=(*img_size, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    return model, base


def main(args):
    img_size = (args.img_size, args.img_size)
    batch_size = args.batch_size

    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=(0.7, 1.3),
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator()

    train_flow = train_datagen.flow_from_directory(
        args.train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse'
    )

    val_flow = val_datagen.flow_from_directory(
        args.val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=False
    )

    num_classes = len(train_flow.class_indices)
    print(f"Found classes: {train_flow.class_indices}")

    # gather labels for computing class weights
    train_labels = train_flow.classes
    class_weight = compute_class_weights(train_labels)
    print("Computed class weights:", class_weight)

    model, base = build_model(img_size=img_size, num_classes=num_classes, dropout=args.dropout)
    model.compile(optimizer=optimizers.Adam(learning_rate=args.lr_head),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    cb = [
        callbacks.ModelCheckpoint(args.output_model, save_best_only=True, monitor='val_loss'),
        callbacks.ReduceLROnPlateau(patience=3, factor=0.2, verbose=1),
        callbacks.EarlyStopping(patience=6, restore_best_weights=True)
    ]

    steps_per_epoch = ceil(train_flow.samples / batch_size)
    validation_steps = ceil(val_flow.samples / batch_size)

    print("Training head...")
    history = model.fit(
        train_flow,
        epochs=args.epochs_head,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_flow,
        validation_steps=validation_steps,
        class_weight=class_weight,
        callbacks=cb
    )

    # Save history
    h = history.history
    with open(args.history_path, 'w') as f:
        json.dump(h, f)

    # Optionally fine-tune
    if args.fine_tune and args.unfreeze_layers > 0:
        print(f"Unfreezing last {args.unfreeze_layers} layers and fine-tuning...")
        base.trainable = True
        # freeze earlier layers
        for layer in base.layers[:-args.unfreeze_layers]:
            layer.trainable = False

        model.compile(optimizer=optimizers.Adam(learning_rate=args.lr_finetune),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        ft_history = model.fit(
            train_flow,
            epochs=args.epochs_finetune,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_flow,
            validation_steps=validation_steps,
            class_weight=class_weight,
            callbacks=cb
        )
        # append
        for k, v in ft_history.history.items():
            h.setdefault(k, []).extend(v)
        with open(args.history_path, 'w') as f:
            json.dump(h, f)

    # final save (best was saved by checkpoint)
    if os.path.exists(args.output_model):
        print(f"Best model saved to {args.output_model}")
    else:
        model.save(args.output_model)
        print(f"Model saved to {args.output_model}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train_dir', default='data/train')
    p.add_argument('--val_dir', default='data/val')
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--epochs_head', type=int, default=10)
    p.add_argument('--epochs_finetune', type=int, default=8)
    p.add_argument('--fine_tune', action='store_true')
    p.add_argument('--unfreeze_layers', type=int, default=30)
    p.add_argument('--lr_head', type=float, default=1e-3)
    p.add_argument('--lr_finetune', type=float, default=1e-4)
    p.add_argument('--output_model', default='model/retrained_mobilenetv2.h5')
    p.add_argument('--history_path', default='backend/training_history.json')
    args = p.parse_args()
    main(args)
