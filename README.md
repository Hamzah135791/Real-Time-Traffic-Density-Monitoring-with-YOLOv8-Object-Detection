# 🚦 Real-Time Traffic Density Monitoring with YOLOv8 Object Detection
### 🎓 Computer Vision Final Project

---

## 📌 Overview
This repository implements a **Real-Time Traffic Monitoring System** that ingests live HTTP Live Streaming (HLS) video feeds, detects and tracks vehicles using the **YOLOv8 Nano** model, and aggregates critical traffic metrics into 30-second windows. 

The system provides two operational modes:
1. **Local Visualization Tool:** For real-time inspection, model debugging, and UI rendering.
2. **Full ELT Pipeline:** For long-term data persistence, environmental context enrichment, and backend dashboarding.

---

## ✨ Features
* **High-Resolution Ingestion:** Captures HLS stream frames at $1280 \times 720$ resolution for high-quality visualization.
* **Optimized Inference:** Dynamically resizes frames to $640 \times 360$ during the YOLOv8 processing phase to maintain high FPS.
* **Persistent Tracking:** Utilizes the **ByteTrack** algorithm to maintain unique vehicle identities seamlessly across frames.
* **Data Enrichment:** Integrates real-time environmental context by fetching live weather data from the OpenWeatherMap API during database commits.
* **Multi-threaded Architecture:** Implements an asynchronous producer-consumer pattern backed by a bounded `queue.Queue` to maximize processing efficiency.

---

## 🏗️ System Architecture
The system relies heavily on a multi-threaded **Producer-Consumer Pattern** to handle heavy data streaming and computation concurrently:
* **`FrameReaderThread` (Producer):** Spawns an optimized `ffmpeg` subprocess to decode the live HLS stream smoothly with minimal latency.
* **`Detector Class` (Consumer):** Pulls raw frames asynchronously from a bounded `queue.Queue`, handles preprocessing, inference, tracking, and aggregation.

### 📊 Pipeline Comparison Matrix

| Feature | `Main.py` (Visualization Mode) | `ELT2/traffic_system.py` (ELT Mode) |
| :--- | :--- | :--- |
| **Primary Goal** | Real-time debugging, verification, and UI | Automated long-term data collection & storage |
| **Display Output** | OpenCV UI Window with live bounding boxes | Clean console logs (Headless mode) |
| **Storage Backend** | None (Immediate output to stdout/console) | **TimescaleDB** (Time-series PostgreSQL extension) |
| **Data Enrichment** | None | Real-time **OpenWeatherMap API** integration |
| **Threading Layout** | Reader Thread + Detector Thread + UI Thread | Reader Thread + Detector Thread |

---