import cv2
import queue
import threading
import time
import numpy as np
from ultralytics import YOLO

# CONFIG

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT * 3  # bgr24 = 3 bytes/pixel

SMALL_W, SMALL_H = 640, 360  # YOLO input (lebih cepat)

HLS_URL = "https://content.tvkur.com/l/cggk2cokj84dao908mv0/master.m3u8"

# Vehicle class IDs (COCO)
VEHICLE_CLASSES = {2, 3, 5, 7}  
# 2=Car, 3=Motorcycle, 5=Bus, 7=Truck


# FRAME READER THREAD

class FrameReaderThread(threading.Thread):
    def __init__(self, url, frame_queue, max_queue=3):
        super().__init__(daemon=True)
        self.url = url
        self.frame_queue = frame_queue
        self.process = None
        self.max_queue = max_queue
        self.running = True

    def run(self):
        import subprocess

        while self.running:
            if self.process is None:
                print("[FFMPEG] Starting HLS capture...")
                self.process = self._start_ffmpeg(self.url)

            try:
                raw = self.process.stdout.read(FRAME_SIZE)

                if len(raw) != FRAME_SIZE:
                    print("[WARN] Corrupted frame, restarting ffmpeg…")
                    self._restart_ffmpeg()
                    continue

                frame = np.frombuffer(raw, np.uint8).reshape(
                    (FRAME_HEIGHT, FRAME_WIDTH, 3)
                )

                if self.frame_queue.qsize() < self.max_queue:
                    self.frame_queue.put(frame)

            except Exception as e:
                print("[ERROR Reader]", e)
                self._restart_ffmpeg()

    def _start_ffmpeg(self, url):
        import subprocess

        cmd = [
            # "ffmpeg",
            r"C:\Users\hamza\computer vision\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-strict", "experimental",
            "-fflags", "+discardcorrupt",
            "-i", url,
            "-vf", "scale=1280:720,format=yuv420p",
            "-r", "15",
            "-an",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-"
        ]

        return subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )

    def _restart_ffmpeg(self):
        if self.process:
            try:
                self.process.kill()
            except:
                pass
        self.process = None

    def stop(self):
        self.running = False
        if self.process:
            try:
                self.process.kill()
            except:
                pass


# DETECTOR THREAD

class Detector(threading.Thread):
    def __init__(self, frame_queue):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.model = YOLO("yolov8n.pt")
        self.running = True

        # Counter 30 detik
        self.count_30_car = 0
        self.count_30_motor = 0
        self.count_30_bus = 0
        self.count_30_truck = 0
        self.window_start = time.time()

        # Track ID
        self.seen_ids = set()

        # Tracking refresh interval
        self.track_interval = 3
        self.frame_id = 0

        # BYTE TRACK bawaan ultralytics
        self.tracker = "bytetrack.yaml"

        # Scale faktor
        self.scale_x = FRAME_WIDTH / SMALL_W
        self.scale_y = FRAME_HEIGHT / SMALL_H

    def run(self):
        fps_time = time.time()

        while self.running:
            if self.frame_queue.empty():
                time.sleep(0.005)
                continue

            frame = self.frame_queue.get()

            # Resize untuk YOLO
            small = cv2.resize(frame, (SMALL_W, SMALL_H))

            self.frame_id += 1

            # Tracking setiap beberapa frame
            if self.frame_id % self.track_interval == 0:
                results = self.model.track(
                    source=small,
                    stream=False,
                    persist=True,
                    tracker=self.tracker,
                    verbose=False
                )[0]
            else:
                results = self.model.predict(
                    source=small,
                    stream=False,
                    verbose=False
                )[0]

            vehicle_count = 0

            disp = frame.copy()

            if results.boxes is not None:
                for box in results.boxes:
                    cls = int(box.cls[0])
                    if cls not in VEHICLE_CLASSES:
                        continue

                    vehicle_count += 1

                    track_id = int(box.id[0]) if box.id is not None else -1

                    # Ambil bbox kecil
                    x1, y1, x2, y2 = box.xyxy[0]

                    # Scale ke ukuran besar
                    X1 = int(x1 * self.scale_x)
                    Y1 = int(y1 * self.scale_y)
                    X2 = int(x2 * self.scale_x)
                    Y2 = int(y2 * self.scale_y)

                    # Gambar bounding box
                    cv2.rectangle(disp, (X1, Y1), (X2, Y2), (0,255,0), 2)
                    cv2.putText(disp, f"{results.names[cls]} ID:{track_id}",
                                (X1, Y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0,255,0), 2)

                    # Hitung unique ID
                    if track_id not in self.seen_ids:
                        self.seen_ids.add(track_id)

                        if cls == 2:
                            self.count_30_car += 1
                        elif cls == 3:
                            self.count_30_motor += 1
                        elif cls == 5:
                            self.count_30_bus += 1
                        elif cls == 7:
                            self.count_30_truck += 1

            # FPS
            now_time = time.time()
            fps = 1 / (now_time - fps_time)
            fps_time = now_time

            cv2.putText(disp, f"Vehicles: {vehicle_count}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            cv2.putText(disp, f"FPS: {fps:.1f}", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

            cv2.imshow("Vehicle Detector", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Summary 30 detik
            if now_time - self.window_start >= 30:
                print("\n============= SUMMARY 30s =============")
                print(f"Car   : {self.count_30_car}")
                print(f"Motor : {self.count_30_motor}")
                print(f"Bus   : {self.count_30_bus}")
                print(f"Truck : {self.count_30_truck}")
                print("========================================\n")

                # Reset
                self.count_30_car = 0
                self.count_30_motor = 0
                self.count_30_bus = 0
                self.count_30_truck = 0
                self.seen_ids.clear()
                self.window_start = now_time

        cv2.destroyAllWindows()

    def stop(self):
        self.running = False


# MAIN

if __name__ == "__main__":
    frame_queue = queue.Queue(maxsize=3)

    reader = FrameReaderThread(HLS_URL, frame_queue)
    detector = Detector(frame_queue)

    reader.start()
    detector.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
        reader.stop()
        detector.stop()

    reader.join()
    detector.join()
