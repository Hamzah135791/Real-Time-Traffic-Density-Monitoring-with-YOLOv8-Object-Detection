import cv2
import queue
import threading
import time
import numpy as np
import requests
import psycopg2
from ultralytics import YOLO


# DATABASE CONFIG
DB = {
    "host": "localhost",
    "port": 5432,
    "dbname": "trafficdb",
    "user": "admin",
    "password": "admin123"
}


# WEATHER CONFIG
API_KEY = "389ea30e23920cf6ac9c3de42447ceb8"
LAT = 39.9489423
LON = 32.6620792


# YOLO + STREAM CONFIG
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT * 3

SMALL_W, SMALL_H = 640, 360
HLS_URL = "https://content.tvkur.com/l/cggk2cokj84dao908mv0/master.m3u8"

VEHICLE_CLASSES = {2, 3, 5, 7}   # car, motor, bus, truck


# WEATHER FUNCTION
def get_weather():
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
    )
    r = requests.get(url).json()

    return {
        "temperature": r["main"]["temp"],
        "humidity": r["main"]["humidity"],
        "wind_speed": r["wind"]["speed"],
        "visibility": r.get("visibility", None),
        "weather_main": r["weather"][0]["main"],
        "weather_desc": r["weather"][0]["description"],
        "rain_mm": r.get("rain", {}).get("1h", 0),
    }


# INSERT FUNCTION
def insert_record(car, motor, bus, truck):
    weather = get_weather()
    total_vehicle = car + motor + bus + truck

    conn = psycopg2.connect(**DB)
    cur = conn.cursor()

    query = """
        INSERT INTO traffic_weather (
            ts, car, motor, bus, truck, total_vehicle,
            temperature, humidity, wind_speed, visibility,
            weather_main, weather_desc, rain_mm
        )
        VALUES (
            NOW(), %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s
        );
    """

    values = (
        car, motor, bus, truck, total_vehicle,
        weather["temperature"], weather["humidity"],
        weather["wind_speed"], weather["visibility"],
        weather["weather_main"], weather["weather_desc"],
        weather["rain_mm"]
    )

    cur.execute(query, values)
    conn.commit()
    cur.close()
    conn.close()

    print("[DB] Inserted:", values)


# FRAME READER THREAD
class FrameReaderThread(threading.Thread):
    def __init__(self, url, frame_queue):
        super().__init__(daemon=True)
        self.url = url
        self.frame_queue = frame_queue
        self.process = None
        self.running = True

    def run(self):
        import subprocess

        while self.running:
            if self.process is None:
                print("[FFMPEG] Starting stream...")
                self.process = self._start_ffmpeg(self.url)

            raw = self.process.stdout.read(FRAME_SIZE)

            if len(raw) != FRAME_SIZE:
                print("[WARN] Restarting ffmpeg…")
                self._restart_ffmpeg()
                continue

            frame = np.frombuffer(raw, np.uint8).reshape(
                (FRAME_HEIGHT, FRAME_WIDTH, 3)
            )

            if self.frame_queue.qsize() < 3:
                self.frame_queue.put(frame)

    def _start_ffmpeg(self, url):
        import subprocess

        cmd = [
            "ffmpeg", 
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-i", url,
            "-vf", "scale=1280:720",
            "-r", "15",
            "-an",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-"
        ]

        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )

    def _restart_ffmpeg(self):
        try:
            self.process.kill()
        except:
            pass
        self.process = None

    def stop(self):
        self.running = False
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

        self.scale_x = FRAME_WIDTH / SMALL_W
        self.scale_y = FRAME_HEIGHT / SMALL_H

        # hitungan dalam 30s
        self.reset_counts()

    def reset_counts(self):
        self.car = 0
        self.motor = 0
        self.bus = 0
        self.truck = 0
        self.seen_ids = set()
        self.window_start = time.time()

    def run(self):
        while self.running:
            if self.frame_queue.empty():
                time.sleep(0.01)
                continue

            frame = self.frame_queue.get()
            small = cv2.resize(frame, (SMALL_W, SMALL_H))

            results = self.model.track(
                source=small, stream=False, persist=True, tracker="bytetrack.yaml"
            )[0]

            if results.boxes is not None:
                for box in results.boxes:
                    cls = int(box.cls[0])
                    if cls not in VEHICLE_CLASSES:
                        continue

                    track_id = int(box.id[0]) if box.id is not None else -1

                    if track_id not in self.seen_ids:
                        self.seen_ids.add(track_id)

                        if cls == 2:
                            self.car += 1
                        elif cls == 3:
                            self.motor += 1
                        elif cls == 5:
                            self.bus += 1
                        elif cls == 7:
                            self.truck += 1

            now = time.time()
            if now - self.window_start >= 30:
                print("\n=== 30s SUMMARY ===")
                print("Car   :", self.car)
                print("Motor :", self.motor)
                print("Bus   :", self.bus)
                print("Truck :", self.truck)

                insert_record(self.car, self.motor, self.bus, self.truck)
                self.reset_counts()

    def stop(self):
        self.running = False


# MAIN
if __name__ == "__main__":
    frame_queue = queue.Queue(maxsize=3)

    reader = FrameReaderThread(HLS_URL, frame_queue)
    detector = Detector(frame_queue)

    reader.start()
    detector.start()

    print("System running... press CTRL+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping system...")
        reader.stop()
        detector.stop()

    reader.join()
    detector.join()
