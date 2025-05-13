import os
import argparse
import yaml
import mlflow
import mlflow.keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from video_generator import VideoFrameGenerator
from models.models import build_model
import tensorflow as tf
def train_model(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

        img_size = tuple(config['model']['image_size'])  
        train_video_dir = config['model']['train_path']
        test_video_dir = config['model']['test_path']
        num_cls = config['load_data']['num_classes']
        rescale = config['img_augment']['rescale']
        shear_range = config['img_augment']['shear_range']
        zoom_range = config['img_augment']['zoom_range']
        horizontal_flip = config['img_augment']['horizontal_flip']
        vertical_flip = config['img_augment']['vertical_flip']
        class_mode = config['img_augment']['class_mode']
        batch_size = config['img_augment']['batch_size']
        loss = config['model']['loss']
        optimizer = config['model']['optimizer']
        metrics = config['model']['metrics']
        epochs = config['model']['epochs']
        model_path = config['model']['save_dir']

   
    if os.path.exists("mlruns/0") and not os.path.exists("mlruns/0/meta.yaml"):
        import shutil
        shutil.rmtree("mlruns/0")#Deletes corrupted experiment directory
  
    mlflow.set_experiment("violence-detection-exp")

    
    train_generator = VideoFrameGenerator(train_video_dir, batch_size, img_size)
    test_generator = VideoFrameGenerator(test_video_dir, batch_size, img_size)

   
    model = build_model(img_size + (3,))  # Add channels

    
    checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor="val_accuracy", mode="max")
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    
    with mlflow.start_run() as run:

       
        mlflow.log_param("image_size", img_size)
        mlflow.log_param("train_video_dir", train_video_dir)
        mlflow.log_param("test_video_dir", test_video_dir)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("loss_function", loss)
        mlflow.log_param("optimizer", optimizer)
        mlflow.log_param("epochs", epochs)

       
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=test_generator,
            callbacks=[checkpoint, early_stop]
        )

       
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        final_train_acc = history.history.get('accuracy', history.history.get('acc', [None]))[-1]
        final_val_acc = history.history.get('val_accuracy', history.history.get('val_acc', [None]))[-1]

        mlflow.log_metric("final_train_loss", final_train_loss)
        mlflow.log_metric("final_val_loss", final_val_loss)
        mlflow.log_metric("final_train_accuracy", final_train_acc)
        mlflow.log_metric("final_val_accuracy", final_val_acc)

        
        mlflow.keras.log_model(model, "violence_detection_model")

    print("✅ Training complete and model logged in MLflow.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='params.yaml')
    args = parser.parse_args()
    train_model(args.config)



