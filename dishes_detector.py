import os
import cv2
import fire
import logging
from ultralytics import YOLO


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class YOLOApp:
    """A class to train and demo YOLO models."""

    def train(self, data_path, model_path, results_path,
              epochs=50, batch_size=16, img_size=640):
        """
        Train the YOLO model.

        Args:
            data_path (str): Path to the data configuration file.
            model_path (str): Path to the base YOLO model.
            results_path (str): Directory to save training results.
            epochs (int, optional): Number of training epochs. Default is 50.
            batch_size (int, optional): Training batch size. Default is 16.
            img_size (int, optional): Image size for training. Default is 640.
        """
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

    def demo(self, model_path, video_path):
        """
        Run the demo using the trained YOLO model on a video file.

        Args:
            model_path (str): Path to the trained YOLO model.
            video_path (str): Path to the input video file.
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


if __name__ == '__main__':
    fire.Fire(YOLOApp)
