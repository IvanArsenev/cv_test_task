import os
import cv2
import fire
import logging
import torch
from ultralytics import YOLO
from datetime import datetime


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class YOLOApp:
    """A class to train and demo YOLO models."""

    def metrics(self, data_path, model_path):
        """
        Get the YOLO model metrics
        Args:
            data_path (str): Path to the data configuration file
            model_path (str): Path to the base YOLO model
        """
        model = YOLO(model_path)
        model.val(data=data_path)
        logging.info("Metrics saved!")

    def train(self, data_path, model_path, results_path,
              epochs=150, batch_size=16, img_size=720):
        """
        Train the YOLO model.

        Args:
            data_path (str): Path to the data configuration file
            model_path (str): Path to the base YOLO model
            results_path (str): Directory to save training results
            epochs (int, optional): Number of training epochs. Default is 150
            batch_size (int, optional): Training batch size. Default is 16
            img_size (int, optional): Image size for training. Default is 720
        """
        logging.info("CUDA IS OK: %s", torch.cuda.is_available())
        os.makedirs(results_path, exist_ok=True)
        model = YOLO(model_path)
        model.train(
            data=data_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            project=results_path,
            name='yolo_training',
            exist_ok=True
        )
        logging.info("Training completed. Results are saved in: %s", results_path)

    def fine_tune(self, data_path, pretrained_model_path, results_path,
                  epochs=20, batch_size=16, img_size=640):
        """
        Fine-tune an existing YOLO model on new data

        Args:
            data_path (str): Path to the new data configuration file
            pretrained_model_path (str): Path to the already trained YOLO model
            results_path (str): Directory to save fine-tuning results
            epochs (int, optional): Number of training epochs. Default is 20
            batch_size (int, optional): Batch size. Default is 16
            img_size (int, optional): Image size for training. Default is 640
        """
        os.makedirs(results_path, exist_ok=True)
        model = YOLO(pretrained_model_path)
        model.train(
            data=data_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            project=results_path,
            name='yolo_finetune',
            exist_ok=True
        )
        logging.info("Fine-tuning completed. Results saved in: %s", results_path)

    def demo(self, model_path, video_path):
        """
        Run the demo using the trained YOLO model on a video file

        Args:
            model_path (str): Path to the trained YOLO model
            video_path (str): Path to the input video file
        """
        model = YOLO(model_path)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logging.error("Error opening video stream or file.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            height, width = frame.shape[:2]
            new_width = 1024
            scale = new_width / width
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))

            results = model.predict(source=frame, conf=0.25, stream=False)
            annotated_frame = results[0].plot()
            cv2.imshow('YOLO Detection of dishes', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def live(self, model_path, save_local=False):
        """
        Run real-time object detection using the YOLO model and webcam

        Args:
            model_path (str): Path to the trained YOLO model
            save_local (bool): Saves caption from cam. Default is False
        """
        model = YOLO(model_path)
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            logging.error("Cannot access the webcam.")
            return

        out = None
        if save_local:
            os.makedirs("recordings", exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            filename = datetime.now().strftime("recordings/output_%Y%m%d_%H%M%S.mp4")
            out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
            logging.info(f"Recording started: {filename}")

        logging.info("Starting live detection. Press 'q' to exit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to grab frame from webcam.")
                break

            results = model.predict(source=frame, conf=0.25, stream=False)
            annotated_frame = results[0].plot()

            if save_local and out:
                out.write(annotated_frame)
            cv2.imshow('YOLO Live Detection', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if out:
            out.release()
            logging.info("Recording saved and closed.")
        cv2.destroyAllWindows()


if __name__ == '__main__':
    fire.Fire(YOLOApp)
