#!/usr/bin/env python3
"""
EVM Polling Booth Monitoring — Unified Edition (v5, 3-frame sampling)
================================================================
Now samples 3 frames per camera per poll cycle (spread evenly across
POLL_INTERVAL_SEC / FOLDER_POLL_INTERVAL_SEC). If ANY of the 3 frames
triggers an alert, the alert fires and per-camera cooldown starts.
All other logic remains identical to v4.
"""

# ── env tweaks before any heavy imports ───────────────────────────────────────
import os
os.environ["OPENCV_FFMPEG_LOGLEVEL"]  = "-8"
os.environ["OMP_NUM_THREADS"]         = "1"
os.environ["OPENBLAS_NUM_THREADS"]    = "1"
os.environ["MKL_NUM_THREADS"]         = "1"
os.environ["VECLIB_MAXIMUM_THREADS"]  = "1"
os.environ["NUMEXPR_NUM_THREADS"]     = "1"
os.environ["TOKENIZERS_PARALLELISM"]  = "false"

import resource
# Increase the limit of open files (File Descriptors) to 1,048,576
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (1048576, 1048576))


import argparse
import asyncio
import cv2
import numpy as np
import torch
import math
import time
import sys
import signal
import subprocess
import threading
import queue
from datetime import datetime, timedelta, timezone
import logging
import traceback
import requests
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Optional
from datetime import timedelta

import supervision as sv
from ultralytics import YOLO

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from azure.storage.blob import BlobServiceClient, ContentSettings
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False


# ============================================================
# LOGGING
# ============================================================

class _AlertFilter(logging.Filter):
    def filter(self, record):
        return "[ALERT]" in record.getMessage()

class _CycleFilter(logging.Filter):
    def filter(self, record):
        return "[WATCH]" in record.getMessage() or "[MAIN]" in record.getMessage()

def setup_production_logging():
    # 1. Use a higher base level (INFO instead of DEBUG)
    LOG_LEVEL = logging.INFO 
    
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(LOG_LEVEL)
    root.handlers.clear()

    # Silencing noisy third-party libraries
    logging.getLogger("watchdog").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("azure").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    # Main production log
    fh = logging.FileHandler("pipeline.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Alerts only log
    ah = logging.FileHandler("alerts.log", encoding="utf-8")
    ah.setLevel(logging.INFO)
    ah.setFormatter(fmt)
    ah.addFilter(_AlertFilter())
    root.addHandler(ah)

    # Debug log (only if you REALLY need to investigate crashes)
    dh = logging.FileHandler("debug.log", encoding="utf-8")
    dh.setLevel(logging.DEBUG) 
    dh.setFormatter(fmt)
    root.addHandler(dh)

    # Console Output (What you see on screen)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    # Updated Filter: Show Watchdog cycles, Main start/stop, AND Alerts on screen
    ch.addFilter(lambda record: any(x in record.getMessage() for x in ["[WATCH]", "[MAIN]", "[ALERT]"]))
    root.addHandler(ch)

setup_production_logging()
log = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = "indoor10.csv"

VIDEO_EXTENSIONS         = ('.flv', '.mp4', '.avi', '.mkv', '.mov')
FOLDER_POLL_INTERVAL_SEC = 30

# ─────────────────────────────────────────────────────────────────────────────

POLL_INTERVAL_SEC      = 30
FFMPEG_TIMEOUT_SEC     = 20
FFMPEG_POOL_WORKERS    = 500
UPLOAD_POOL_WORKERS    = 128
CAM_BACKOFF_MAX_CYCLES = 10
CSV_RELOAD_INTERVAL_SEC  = 600

# ── Inference pool ───────────────────────────────────────────────────────────
# 2 GPU workers. Each holds its own (booth, person) model replica and
# pulls from a shared frame queue — so when one worker is busy with NMS
# or Python post-processing, the other can already be running its batch
# on the GPU. Roughly doubles effective GPU utilisation on a single card.
INFERENCE_WORKERS    = 2
INFERENCE_BATCH_SIZE = 256
INFERENCE_QUEUE_SIZE = 8000

BOOTH_MODEL_PATH  = "evm_detection.pt"
PERSON_MODEL_PATH = "yolo11m.pt"

BOOTH_CONFIDENCE  = 0.95
PERSON_CONFIDENCE = 0.20
PERSON_NMS_IOU_THRESH = 0.45
MIN_BOOTH_AREA_PX = 3000

PERSON_BOOTH_IOU_THRESH       = 0.10
PERSON_BOOTH_IOP_THRESH       = 0.10
PERSON_EVM_OVERLAP_IOP_THRESH = 0.15

EVM_ROI_EXPAND_LR_SCALE  = 0.05
EVM_ROI_EXPAND_TOP_SCALE = 0.10
DEAD_ZONE_SCALE          = 0.40
DEAD_ZONE_OVERLAP_THRESH = 0.20 
PROXIMITY_BOOTH_SCALE    = 0.40
PERSON_PERSON_IOP_THRESH = 0.25

# How much of the person must be inside the EVM area to count them
PERSON_BOOTH_IOU_THRESH = 0.35 
PERSON_BOOTH_IOP_THRESH = 0.35

PRIMARY_SETTLE_SECS     = 0.0
DWELL_SECONDS           = 0.0
BOOTH_REDETECT_INTERVAL = 60
COOLDOWN_SECONDS        = 20
ALERT_COOLDOWN_SEC      = 180

ALERT_FOLDER       = "alert"
SAVE_ALERT_IMAGES  = False
INFERENCE_ROOT_DIR = "inference"
ALERT_WEBHOOK_URL  = "https://tn2023demo.vmukti.com/api/analytics"
ANALYTICS_ID            = 103
ANALYTICS_ID_VACANT     = 104
ANALYTICS_ID_MAX_PERSON = 102
MAX_PERSON_THRESHOLD    = 10

AZURE_CONN_STR = (
    "BlobEndpoint=https://nvrdatashinobi.blob.core.windows.net/;"
    "QueueEndpoint=https://nvrdatashinobi.queue.core.windows.net/;"
    "FileEndpoint=https://nvrdatashinobi.file.core.windows.net/;"
    "TableEndpoint=https://nvrdatashinobi.table.core.windows.net/;"
    "SharedAccessSignature=sv=2024-11-04&ss=bfqt&srt=sco&sp=rwdlacupiytfx"
    "&se=2026-05-30T20:16:33Z&st=2026-02-20T12:01:33Z&spr=https,http"
    "&sig=YenJbBQB3iuMwhJqtu724lm7ID%2B6L2GXpbv%2BpmhTwrk%3D"
)
AZURE_CONTAINER_NAME = "nvrdatashinobi"
AZURE_BLOB_PREFIX    = "live-record/frimages"
STATIC_IMAGE_URL     = (
    f"https://nvrdatashinobi.blob.core.windows.net/"
    f"{AZURE_CONTAINER_NAME}/{AZURE_BLOB_PREFIX}"
)


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="EVM Polling Booth Monitor — Unified (v5, 3-frame sampling)."
    )
    p.add_argument("--debug", action="store_true",
        help="Show debug overlay: expanded EVM ROI, dead-zone, proximity circle.")
    return p.parse_args()


# ============================================================
# CYCLE STATS
# ============================================================

class CycleStats:
    def __init__(self):
        self._lock = threading.Lock()
        self._reset()

    def _reset(self):
        self.grabs_ok = 0; self.grabs_fail = 0
        self.grab_sum_s = 0.0; self.grab_max_s = 0.0; self._last_grab_mon = 0.0
        self.infers_ok = 0
        self.infer_sum_s = 0.0; self.infer_max_s = 0.0; self._last_infer_mon = 0.0
        self.alerts_breach = 0; self.alerts_vacant = 0; self.alerts_max_person = 0
        self.alerts_ok = 0; self.alerts_fail = 0
        self.cycle_start_mon = time.monotonic()

    current_cycle: int = 1

    def record_grab(self, elapsed_s: float, success: bool):
        with self._lock:
            if success: self.grabs_ok += 1
            else:        self.grabs_fail += 1
            self.grab_sum_s += elapsed_s
            if elapsed_s > self.grab_max_s: self.grab_max_s = elapsed_s
            t = time.monotonic()
            if t > self._last_grab_mon: self._last_grab_mon = t

    def record_infer(self, elapsed_s: float):
        with self._lock:
            self.infers_ok += 1
            self.infer_sum_s += elapsed_s
            if elapsed_s > self.infer_max_s: self.infer_max_s = elapsed_s
            t = time.monotonic()
            if t > self._last_infer_mon: self._last_infer_mon = t

    def record_alert(self, kind: str):
        with self._lock:
            if kind == "breach":      self.alerts_breach += 1
            elif kind == "vacant":    self.alerts_vacant += 1
            elif kind == "max_person":self.alerts_max_person += 1

    def record_alert_status(self, success: bool):
        with self._lock:
            if success: self.alerts_ok += 1
            else:       self.alerts_fail += 1

    def snapshot_and_reset(self) -> dict:
        with self._lock:
            now = time.monotonic()
            snap = dict(
                grabs_ok=self.grabs_ok, grabs_fail=self.grabs_fail,
                grab_avg_ms=round(1000*self.grab_sum_s/self.grabs_ok, 1) if self.grabs_ok else 0.0,
                grab_max_ms=round(self.grab_max_s*1000, 1),
                last_grab_at_s=round(self._last_grab_mon - self.cycle_start_mon, 2) if self._last_grab_mon else 0.0,
                infers_ok=self.infers_ok,
                infer_avg_ms=round(1000*self.infer_sum_s/self.infers_ok, 1) if self.infers_ok else 0.0,
                infer_max_ms=round(self.infer_max_s*1000, 1),
                last_infer_at_s=round(self._last_infer_mon - self.cycle_start_mon, 2) if self._last_infer_mon else 0.0,
                cycle_wall_s=round(now - self.cycle_start_mon, 2),
                alerts_breach=self.alerts_breach,
                alerts_vacant=self.alerts_vacant,
                alerts_max_person=self.alerts_max_person,
                alerts_ok=self.alerts_ok,
                alerts_fail=self.alerts_fail,
            )
            self._reset()
            return snap


# ============================================================
# HELPERS
# ============================================================

_BOX_THICK = 2
_FONT      = cv2.FONT_HERSHEY_DUPLEX
_RED_DARK  = (0, 0, 180)


def _rounded_rect(img, pt1, pt2, color, thickness, radius):
    x1, y1 = pt1; x2, y2 = pt2
    r = int(max(1, min(radius, (x2-x1)//2, (y2-y1)//2)))
    cv2.line(img, (x1+r,y1), (x2-r,y1), color, thickness)
    cv2.line(img, (x1+r,y2), (x2-r,y2), color, thickness)
    cv2.line(img, (x1,y1+r), (x1,y2-r), color, thickness)
    cv2.line(img, (x2,y1+r), (x2,y2-r), color, thickness)
    cv2.ellipse(img, (x1+r,y1+r), (r,r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2-r,y1+r), (r,r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1+r,y2-r), (r,r),  90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2-r,y2-r), (r,r),   0, 0, 90, color, thickness)


def get_all_video_files(root_dir):
    files = []
    for root, _, filenames in os.walk(root_dir):
        for f in sorted(filenames):
            if f.lower().endswith(VIDEO_EXTENSIONS):
                files.append(os.path.join(root, f))
    return files


def load_camera_urls(csv_path: str):
    if not PANDAS_AVAILABLE:
        log.error("pandas required for CSV mode: pip install pandas"); sys.exit(1)
    df = pd.read_csv(csv_path, header=None, encoding="utf-8-sig")
    urls = [u.strip() for u in df[0].dropna().unique().tolist()
            if str(u).strip().startswith(("rtmp", "rtsp", "http"))]
    if not urls:
        log.error("No valid camera URLs in CSV."); sys.exit(1)
    log.info(f"Loaded {len(urls)} camera URLs from {csv_path}")
    return urls


def make_json_serializable(obj):
    if isinstance(obj, dict):   return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):   return [make_json_serializable(v) for v in obj]
    if isinstance(obj, (np.integer, np.int64)):    return int(obj)
    if isinstance(obj, (np.floating, np.float64)): return float(obj)
    if isinstance(obj, np.ndarray):                return obj.tolist()
    return obj


# ============================================================
# GEOMETRY
# ============================================================

def _box_area(box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2-x1) * max(0.0, y2-y1)

def _intersection_area(a, b) -> float:
    ix1 = max(a[0],b[0]); iy1 = max(a[1],b[1])
    ix2 = min(a[2],b[2]); iy2 = min(a[3],b[3])
    return max(0.0, ix2-ix1) * max(0.0, iy2-iy1)

def box_center(box):
    return ((box[0]+box[2])/2.0, (box[1]+box[3])/2.0)

def pick_best_booth(boxes_xyxy):
    best_idx, best_area = None, -1
    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = box
        area = (x2-x1)*(y2-y1)
        if area < MIN_BOOTH_AREA_PX: continue
        if area > best_area: best_area = area; best_idx = i
    return best_idx

def person_at_booth(person_box, booth_box) -> tuple:
    inter = _intersection_area(person_box, booth_box)
    if inter == 0.0: return False, 0.0, 0.0
    p_area = _box_area(person_box); b_area = _box_area(booth_box)
    union  = p_area + b_area - inter
    iou    = inter / union  if union  > 0 else 0.0
    iop    = inter / p_area if p_area > 0 else 0.0
    at_booth = (iou >= PERSON_BOOTH_IOU_THRESH or
                iop >= PERSON_BOOTH_IOP_THRESH)
    return at_booth, iou, iop

def secondary_near_primary(primary_box, secondary_box, radius_px: float) -> tuple:
    pcx, pcy = box_center(primary_box)
    scx, scy = box_center(secondary_box)
    dist   = math.hypot(scx-pcx, scy-pcy)
    inter  = _intersection_area(primary_box, secondary_box)
    s_area = _box_area(secondary_box); p_area = _box_area(primary_box)
    p2p_iop = inter / s_area if s_area > 0 else 0.0
    near = (dist <= radius_px or
            p2p_iop >= PERSON_PERSON_IOP_THRESH)
    return near, dist, p2p_iop

def person_overlaps_evm(person_box, booth_raw) -> bool:
    if booth_raw is None: return False
    inter = _intersection_area(person_box, booth_raw)
    if inter == 0.0: return False
    iop = inter / _box_area(person_box) if _box_area(person_box) > 0 else 0.0
    return iop >= PERSON_EVM_OVERLAP_IOP_THRESH


# ============================================================
# PERSON DETECTION
# ============================================================

def is_box_inside_or_contained(small_box, large_box, threshold=0.85) -> bool:
    """
    Returns True if small_box is mostly contained inside large_box.
    This catches head-box inside full-body-box duplicates.
    """
    inter = _intersection_area(small_box, large_box)
    small_area = _box_area(small_box)
    
    # What % of the small box is inside the large box?
    containment = inter / small_area if small_area > 0 else 0.0
    
    return containment >= threshold  # 85% of small box is inside large box

def result_to_person_dets(result) -> sv.Detections:
    dets = sv.Detections.from_ultralytics(result)
    dets = dets[dets.class_id == 0]

    if len(dets) < 2:
        return dets

    # ── Step 1: Standard IOU-based NMS (catches side-by-side duplicates) ──
    boxes  = torch.tensor(dets.xyxy, dtype=torch.float32)
    scores = torch.tensor(dets.confidence, dtype=torch.float32)
    keep   = torch.ops.torchvision.nms(boxes, scores, PERSON_NMS_IOU_THRESH)
    dets   = dets[keep.numpy()]

    if len(dets) < 2:
        return dets

    # ── Step 2: Containment check (catches head-box inside body-box) ──────
    # This is what fixes YOUR image — small head box inside large body box
    boxes_list = dets.xyxy
    confs_list = dets.confidence
    remove_indices = set()

    for i in range(len(boxes_list)):
        for j in range(len(boxes_list)):
            if i == j or i in remove_indices or j in remove_indices:
                continue

            inter = _intersection_area(boxes_list[i], boxes_list[j])
            area_i = _box_area(boxes_list[i])

            # How much of box_i is inside box_j?
            containment = inter / area_i if area_i > 0 else 0.0

            if containment >= 0.85:
                # box_i is 85% inside box_j
                # They are the same person — remove the smaller/weaker one
                if confs_list[i] <= confs_list[j]:
                    remove_indices.add(i)  # remove smaller box
                else:
                    remove_indices.add(j)

    keep_indices = [i for i in range(len(dets)) if i not in remove_indices]
    return dets[keep_indices]


# ============================================================
# BREACH CHECK
# ============================================================

def check_breach(person_dets: sv.Detections, booth_raw,
                 frame_width: int, frame_height: int):
    overlap_info = []
    
    # 1. Define Zones
    bx1, by1, bx2, by2 = booth_raw
    booth_mid_y = by1 + (by2 - by1) / 2.0
    evm_w = max(0.0, bx2 - bx1)
    evm_h = max(0.0, by2 - by1)

    # ROI (Pink Box)
    ebx1 = max(0,           bx1 - evm_w * EVM_ROI_EXPAND_LR_SCALE)
    eby1 = max(0,           by1 - evm_h * EVM_ROI_EXPAND_TOP_SCALE)
    ebx2 = min(frame_width, bx2 + evm_w * EVM_ROI_EXPAND_LR_SCALE)
    eby2 = booth_mid_y
    booth_expanded = (ebx1, eby1, ebx2, eby2)

    # Dead zone (Red Box)
    dead_zone_half = min(evm_w, evm_h) * DEAD_ZONE_SCALE
    evm_cx = (bx1 + bx2) / 2.0; evm_cy = (by1 + by2) / 2.0
    dead_zone = (evm_cx - dead_zone_half, evm_cy - dead_zone_half,
                 evm_cx + dead_zone_half, evm_cy + dead_zone_half)

    # 2. Identify the Primary Voter
    primary_idx = None
    best_iop = -1.0
    for i, row in enumerate(person_dets.xyxy):
        at_booth, iou, iop = person_at_booth(row, booth_expanded)
        if at_booth and iop > best_iop:
            best_iop = iop
            primary_idx = i

    if primary_idx is None:
        return (0, [], [], None, [], 0.0, booth_expanded, dead_zone)

    # 3. Process everyone else
    booth_w = ebx2 - ebx1; booth_h = eby2 - eby1
    radius_px = max(booth_w, booth_h) * PROXIMITY_BOOTH_SCALE
    primary_box = person_dets.xyxy[primary_idx]
    
    secondary_info = []
    proximity_indices = []

    for i, row in enumerate(person_dets.xyxy):
        # Calculate Overlap with Dead Zone
        inter_dead = _intersection_area(row, dead_zone)
        dz_area = _box_area(dead_zone)
        overlap_pct = (inter_dead / dz_area) if dz_area > 0 else 0.0
        
        # Determine if this person should be ignored (Safe)
        is_safe_in_dz = overlap_pct >= DEAD_ZONE_OVERLAP_THRESH

        if i == primary_idx:
            # The voter is always "safe" and doesn't trigger alerts on themselves
            overlap_info.append({"iou": 1.0, "iop": 1.0, "dead_zone_skip": True})
            continue

        # LOGIC: If the person meets the overlap threshold in the dead zone, IGNORE them
        if is_safe_in_dz:
            overlap_info.append({"iou": 0, "iop": 0, "dead_zone_skip": True})
            continue 

        # Only check breach rules for people NOT considered safe
        is_near, dist_px, p2p_iop = secondary_near_primary(primary_box, row, radius_px)
        at_booth, iou, iop = person_at_booth(row, booth_expanded)
        
        overlap_info.append({"iou": iou, "iop": iop, "dead_zone_skip": False})

        if is_near or at_booth:
            # This person is an intruder
            secondary_info.append({"idx": i, "dist_px": dist_px, "p2p_iop": p2p_iop})
            proximity_indices.append(i)

    # 4. Final Count and UI setup
    behind_count = len(secondary_info)
    
    if behind_count > 0:
        # If there's an actual intruder, turn the voter's box red as well
        proximity_indices.append(primary_idx)

    return (behind_count, proximity_indices, overlap_info,
            primary_idx, secondary_info, radius_px, booth_expanded, dead_zone)


# ============================================================
# PROXIMITY DWELL TRACKER
# ============================================================

class ProximityTracker:
    def __init__(self):
        self._tracks: dict = {}
        self._next_id: int = 0
        self._primary_since: float | None = None

    def update(self, secondary_info, person_dets, radius_px, now, primary_present):
        if primary_present:
            if self._primary_since is None: self._primary_since = now
        else:
            self._primary_since = None; self._tracks = {}; return []

        primary_ready = (now - self._primary_since) >= PRIMARY_SETTLE_SECS
        current = []
        for info in secondary_info:
            box = person_dets.xyxy[info["idx"]]
            cx, cy = box_center(box)
            current.append({"idx": info["idx"], "cx": cx, "cy": cy})

        match_radius = radius_px * 0.5
        used_tracks = set(); new_tracks = {}

        for cand in current:
            best_tid, best_dist = None, float("inf")
            for tid, trk in self._tracks.items():
                if tid in used_tracks: continue
                d = math.hypot(cand["cx"]-trk["centre"][0], cand["cy"]-trk["centre"][1])
                if d < best_dist and d < match_radius:
                    best_dist = d; best_tid = tid
            if best_tid is not None:
                used_tracks.add(best_tid)
                new_tracks[best_tid] = {
                    "centre": (cand["cx"], cand["cy"]),
                    "start":  self._tracks[best_tid]["start"],
                    "idx":    cand["idx"],
                }
            else:
                new_tracks[self._next_id] = {
                    "centre": (cand["cx"], cand["cy"]),
                    "start":  now if primary_ready else now + 9999,
                    "idx":    cand["idx"],
                }
                self._next_id += 1

        self._tracks = new_tracks
        return [{"idx": t["idx"], "elapsed": max(0.0, now-t["start"]), "track_id": tid}
                for tid, t in self._tracks.items()]


# ============================================================
# SAVE / ALERT
# ============================================================

def save_inference_frame(frame, cam_id, cycle_num):
    try:
        pass
        # subfolder = INFERENCE_ROOT_DIR
        # os.makedirs(subfolder, exist_ok=True)
        # cv2.imwrite(f"{subfolder}/{cam_id}.jpg", frame)
    except Exception as e:
        log.error(f"[ENGINE] save_inference_frame error ({cam_id}, cycle {cycle_num}): {e}")


def save_alert_image(cam_id, frame, timestamp, alert_data,
                     alert_type="booth", an_id=None, cycle_num=None, cycle_stats=None,
                     save_local=False):
    try:
        only_filename = (f"{cam_id}_alert_{alert_type}_"
                         f"{timestamp.strftime('%Y%m%d_%H%M%S')}.png")
        img_url       = f"{STATIC_IMAGE_URL}/{only_filename}"

        if AZURE_AVAILABLE and AZURE_CONN_STR:
            try:
                bc   = BlobServiceClient.from_connection_string(AZURE_CONN_STR)
                blob = bc.get_blob_client(container=AZURE_CONTAINER_NAME,
                                          blob=f"{AZURE_BLOB_PREFIX}/{only_filename}")
                _, buf = cv2.imencode(".png", frame)
                blob.upload_blob(buf.tobytes(), overwrite=True,
                                 content_settings=ContentSettings(content_type="image/png"))
                log.info(f"[{cam_id}] Azure upload: {img_url}")
            except Exception as e:
                log.error(f"[{cam_id}] Azure upload failed: {e}")

        # Local saving (controlled by SAVE_ALERT_IMAGES config)
        # if save_local or SAVE_ALERT_IMAGES:
        #     cycle_folder = f"cycle{cycle_num}" if cycle_num else "cycle0"
        #     local_dir = os.path.join(ALERT_FOLDER, cycle_folder, alert_type)
        #     os.makedirs(local_dir, exist_ok=True)
        #     cv2.imwrite(os.path.join(local_dir, only_filename), frame)

        payload = make_json_serializable({
            "cameradid":    str(cam_id),
            "sendtime": (datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "imgurl":       img_url,
            "an_id":        an_id if an_id is not None else ANALYTICS_ID,
            "totalcount":   alert_data["total"],
            "behind_count": alert_data["behind_count"],
            "alert_type":   alert_type,
            "ImgCount":     4,
        })
        if ALERT_WEBHOOK_URL:
            try:
                requests.post(ALERT_WEBHOOK_URL, json=payload, timeout=10)
                log.info(f"[{cam_id}] Webhook sent ({alert_type})")
                if cycle_stats: cycle_stats.record_alert_status(success=True)
                status_str = "OK"
            except Exception as e:
                log.error(f"[{cam_id}] Webhook failed: {e}")
                if cycle_stats: cycle_stats.record_alert_status(success=False)
                status_str = "FAILED"
            
            # alert.log
            ts_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            try:
                with open("alert.log", "a", encoding="utf-8") as f:
                    f.write(f"[{ts_str}] Cam: {cam_id} | Alert: {alert_type} | Status: {status_str} | URL: {img_url}\n")
            except Exception as e:
                log.error(f"Failed to write to alert.log: {e}")

        log.info(f"[{cam_id}] ALERT [{alert_type}]: processed")
    except Exception as e:
        log.error(f"[{cam_id}] save_alert_image error: {e}")


# ============================================================
# DEBUG OVERLAY
# ============================================================

def _draw_dashed_rect(img, x1, y1, x2, y2, color, gap=8, seg=8):
    for (ax,ay),(bx,by) in [((x1,y1),(x2,y1)),((x2,y1),(x2,y2)),
                              ((x2,y2),(x1,y2)),((x1,y2),(x1,y1))]:
        length = int(math.hypot(bx-ax,by-ay))
        dx = (bx-ax)/max(length,1); dy = (by-ay)/max(length,1)
        pos = 0; draw = True
        while pos < length:
            end = min(pos+(seg if draw else gap), length)
            if draw:
                cv2.line(img, (int(ax+dx*pos),int(ay+dy*pos)),
                         (int(ax+dx*end),int(ay+dy*end)), color, 1, cv2.LINE_AA)
            pos += seg if draw else gap; draw = not draw

def _draw_dashed_circle(img, cx, cy, r, color, gap=10, seg=10):
    c = 2*math.pi*r; n = max(1,int(c/(seg+gap))); step = 2*math.pi/n
    for k in range(n):
        a0 = k*step; a1 = a0+step*seg/(seg+gap)
        pts = [(int(cx+r*math.cos(a)),int(cy+r*math.sin(a)))
               for a in np.linspace(a0,a1,8)]
        for j in range(len(pts)-1):
            cv2.line(img, pts[j], pts[j+1], color, 1, cv2.LINE_AA)

def draw_debug_overlay(ann, booth_raw, booth_expanded, dead_zone,
                       person_dets, primary_idx, secondary_info,
                       overlap_info, radius_px):
    font = cv2.FONT_HERSHEY_SIMPLEX
    if booth_expanded:
        ex1,ey1,ex2,ey2 = [int(v) for v in booth_expanded]
        _draw_dashed_rect(ann,ex1,ey1,ex2,ey2,(255,0,255),gap=8)
        cv2.putText(ann,"EVM ROI",(ex1+2,ey1-4),font,0.42,(255,0,255),1,cv2.LINE_AA)
    if dead_zone:
        dx1,dy1,dx2,dy2 = [int(v) for v in dead_zone]
        _draw_dashed_rect(ann,dx1,dy1,dx2,dy2,(0,0,200),gap=5)
        cv2.putText(ann,"dead zone",(dx1+2,dy2+12),font,0.38,(0,0,200),1,cv2.LINE_AA)
    sec_idxs = {s["idx"] for s in secondary_info}
    if primary_idx is not None and booth_raw is not None:
        px,py = box_center(person_dets.xyxy[primary_idx])
        _draw_dashed_circle(ann,int(px),int(py),int(radius_px),(0,200,0),gap=10)
    for i, row in enumerate(person_dets.xyxy):
        px1,py1,px2,py2 = int(row[0]),int(row[1]),int(row[2]),int(row[3])
        info = overlap_info[i] if i < len(overlap_info) else {}
        if i == primary_idx:
            color=(255,220,0);   label="PRIMARY";   thick=2
        elif i in sec_idxs:
            color=(0,165,255);   label="SECONDARY"; thick=1
        else:
            color=(160,160,160); label="person";    thick=1
        cv2.rectangle(ann,(px1,py1),(px2,py2),color,thick)
        tag = (f"{label} iou={info.get('iou',0):.2f} iop={info.get('iop',0):.2f}"
               + (" [DZ]" if info.get("dead_zone_skip") else ""))
        cv2.putText(ann,tag,(px1,max(py1-5,14)),font,0.38,color,1,cv2.LINE_AA)
    return ann


# ============================================================
# ANNOTATE FRAME
# ============================================================

def annotate_frame(frame, booth_raw, person_dets: sv.Detections,
                   total_persons: int, breach_active: bool,
                   debug: bool = False,
                   booth_expanded=None, dead_zone=None,
                   primary_idx=None, secondary_info=None, overlap_info=None,
                   radius_px: float = 0.0,
                   vacant_active: bool = False, max_person_active: bool = False,
                   proximity_indices=[]):
    ann = frame.copy()
    h, w = ann.shape[:2]

    if debug and person_dets is not None and len(person_dets) > 0:
        ann = draw_debug_overlay(ann, booth_raw, booth_expanded, dead_zone,
                                 person_dets, primary_idx, secondary_info or [],
                                 overlap_info or [], radius_px)

    if person_dets is not None:
        for i, box in enumerate(person_dets.xyxy):
            x1, y1, x2, y2 = map(int, box)
            is_in_breach = i in proximity_indices
            color = (0, 0, 180) if is_in_breach else (0, 200, 0)
            _rounded_rect(ann, (x1, y1), (x2, y2), color, _BOX_THICK, int((y2 - y1) * 0.12))

    if booth_raw is not None:
        bx1, by1, bx2, by2 = [int(v) for v in booth_raw]
        bd = sv.Detections(xyxy=np.array([[bx1, by1, bx2, by2]], dtype=np.float32))
        ann = sv.BoxCornerAnnotator(color=sv.Color.from_hex("#FFFF00"), thickness=3,
                                    color_lookup=sv.ColorLookup.INDEX).annotate(
                                        scene=ann, detections=bd)
        cv2.putText(ann, "", (bx1, max(by1 - 10, 15)), _FONT, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    HUD_WIDTH = 200; font_scale = 0.5
    hx2 = w - 12; hx1 = hx2 - HUD_WIDTH; current_y = 12

    if not vacant_active:
        badge = f"Indoor (People): {total_persons}"
        (bw, bh), _ = cv2.getTextSize(badge, _FONT, font_scale, 1)
        ry1, ry2 = current_y, current_y + bh + 14
        cv2.rectangle(ann, (hx1, ry1), (hx2, ry2), (210, 210, 210), -1)
        cv2.putText(ann, badge, (hx1 + (HUD_WIDTH - bw) // 2, ry1 + bh + 7),
                    _FONT, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
        current_y = ry2

    active_alerts = []
    if vacant_active:
        active_alerts.append("ALERT: VACANT BOOTH")
    if max_person_active:
        active_alerts.append("ALERT: MAX-PERSON")
    if breach_active:
        active_alerts.append("ALERT: PRIVACY BREACH")

    for alert_txt in active_alerts:
        (aw, ah), _ = cv2.getTextSize(alert_txt, _FONT, font_scale, 1)
        ay1, ay2 = current_y, current_y + ah + 14
        cv2.rectangle(ann, (hx1, ay1), (hx2, ay2), _RED_DARK, -1)
        cv2.putText(ann, alert_txt, (hx1 + (HUD_WIDTH - aw) // 2, ay1 + ah + 7),
                    _FONT, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        current_y = ay2

    return ann


# ============================================================
# INFERENCE ENGINE
# ============================================================

class UnifiedInferenceEngine:
    """
    Multi-worker inference pool.

    N worker threads pull from a shared frame queue. Each worker owns its
    own (booth, person) model replica so they don't contend for model
    state. On a single GPU this overlaps Python-side post-processing (NMS
    on CPU, .cpu() transfers, result construction) with the next worker's
    GPU call — the GIL is released around torch calls, so real parallelism
    is possible.
    """

    def __init__(self, first_models,
                 n_workers: int = INFERENCE_WORKERS,
                 batch_size: int = INFERENCE_BATCH_SIZE,
                 queue_size: int = INFERENCE_QUEUE_SIZE):
        self.batch_size     = batch_size
        self.queue          = queue.Queue(maxsize=queue_size)
        self.shutdown_event = threading.Event()

        # Worker 0 reuses the model pair already loaded by the main thread.
        # Workers 1..N-1 each load their own replicas.
        self.model_pairs = [first_models]
        for i in range(1, n_workers):
            extra = load_models(f"W{i}")
            if extra[0] is None:
                raise RuntimeError(f"[ENGINE] Worker {i} model load failed")
            self.model_pairs.append(extra)

        self.workers = []
        for i in range(n_workers):
            t = threading.Thread(target=self._worker, args=(i,),
                                 daemon=True, name=f"infer-W{i}")
            t.start()
            self.workers.append(t)
        log.info(f"[ENGINE] Started {n_workers} GPU workers "
                 f"(batch={batch_size}, queue={queue_size})")

    def _worker(self, widx: int):
        bm, pm = self.model_pairs[widx]
        while not self.shutdown_event.is_set():
            batch = []
            try:
                item = self.queue.get(timeout=0.1)
                batch.append(item)
                while len(batch) < self.batch_size:
                    try: batch.append(self.queue.get_nowait())
                    except queue.Empty: break
            except queue.Empty:
                continue
            if not batch: continue

            frames  = [it[0] for it in batch]
            futures = [it[1] for it in batch]
            try:
                with torch.no_grad():
                    b_res = bm(frames, conf=BOOTH_CONFIDENCE, verbose=False)
                    p_res = pm(
                        frames,
                        imgsz=[384, 640],
                        conf=PERSON_CONFIDENCE,
                        iou=0.35,
                        agnostic_nms=True,
                        classes=[0],
                        verbose=False
                    )
                for i, fut in enumerate(futures):
                    if not fut.done():
                        res = {
                            "booth":  b_res[i].boxes.xyxy.cpu().numpy(),
                            "person": p_res[i],
                        }
                        fut.get_loop().call_soon_threadsafe(fut.set_result, res)
            except Exception as e:
                log.error(f"[ENGINE-W{widx}] Inference worker error: {e}")
                for fut in futures:
                    if not fut.done():
                        fut.get_loop().call_soon_threadsafe(fut.set_exception, e)

    def submit(self, frame):
        loop = asyncio.get_event_loop()
        fut  = loop.create_future()
        try:
            self.queue.put_nowait((frame, fut))
        except queue.Full:
            fut.set_exception(RuntimeError("Inference queue full"))
        return fut

    def stop(self):
        self.shutdown_event.set()
        for t in self.workers:
            if t.is_alive():
                t.join(timeout=1.0)


# ============================================================
# MODEL LOADER
# ============================================================

def load_models(tag=""):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    try:
        bm = YOLO(BOOTH_MODEL_PATH).to(device);  bm.eval()
        pm = YOLO(PERSON_MODEL_PATH).to(device); pm.eval()
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        with torch.no_grad():
            bm(dummy, verbose=False)
            pm(dummy, verbose=False)
        log.info(f"[{tag}] Booth model  (YOLO)    ready on {device}")
        log.info(f"[{tag}] Person model (YOLO11n) ready on {device}")
        return bm, pm
    except Exception as e:
        log.error(f"[{tag}] Model load failed: {e}")
        return None, None


# ============================================================
# CORE FRAME PROCESSOR
# ============================================================

def process_single_frame_v3(
    frame, inference_results,
    booth_cache_ref, last_booth_time_ref,
    frame_time: float, frame_width: int, frame_height: int,
    debug: bool, proximity_tracker,
):
    now = frame_time

    # ── Booth (cached) ────────────────────────────────────────────────────────
    if (booth_cache_ref[0] is None or
            (now - last_booth_time_ref[0]) >= BOOTH_REDETECT_INTERVAL):
        raw = inference_results["booth"]
        if len(raw) > 0:
            idx = pick_best_booth(raw)
            booth_cache_ref[0] = tuple(raw[idx]) if idx is not None else None
        else:
            booth_cache_ref[0] = None
        last_booth_time_ref[0] = now
    booth_raw = booth_cache_ref[0]

    # ── Person detection ──────────────────────────────────────────────────────
    person_dets   = result_to_person_dets(inference_results["person"])
    total_persons = len(person_dets)

    # ── Breach check ──────────────────────────────────────────────────────────
    behind_count = 0; proximity_indices = []; overlap_info = []
    primary_idx = None; secondary_info = []
    radius_px = 0.0; booth_expanded = None; dead_zone = None

    if booth_raw is not None and len(person_dets) > 0:
        (behind_count, proximity_indices, overlap_info,
         primary_idx, secondary_info, radius_px,
         booth_expanded, dead_zone) = check_breach(
            person_dets, booth_raw, frame_width, frame_height)

    # ── Annotate ──────────────────────────────────────────────────────────────
    annotated = annotate_frame(
        frame, booth_raw, person_dets, total_persons,
        breach_active=(behind_count >= 1), debug=debug,
        booth_expanded=booth_expanded, dead_zone=dead_zone,
        primary_idx=primary_idx, secondary_info=secondary_info,
        overlap_info=overlap_info, radius_px=radius_px,
        vacant_active=(total_persons == 0),
        max_person_active=(total_persons > MAX_PERSON_THRESHOLD),
        proximity_indices=proximity_indices
    )

    return {
        "annotated": annotated, 
        "is_breach_frame": (behind_count >= 1), 
        "total": total_persons, 
        "behind": behind_count, 
        "is_vacant_frame": (total_persons == 0),
        "is_max_frame": (total_persons > MAX_PERSON_THRESHOLD)
    }


# ============================================================
# ── NEW HELPER: infer + process N frames, return first alert ─
# ============================================================

async def process_frames_multi(
    frames_with_times, engine, cam_id,
    booth_cache_ref, last_booth_time_ref,
    frame_width, frame_height,
    debug, proximity_tracker,
    last_alert_time_ref, in_cooldown_ref, 
    last_vacant_time_ref, in_vacant_cooldown_ref,
    last_max_person_time_ref, in_max_person_cooldown_ref,
    cooldown_seconds, cycle_stats,
):
    if not frames_with_times: return None

    # 1. Inference
    futs = [engine.submit(f) for f, _ in frames_with_times]
    frame_data = []
    for (frame, vt), fut in zip(frames_with_times, futs):
        t_inf = time.monotonic()
        res = await fut
        cycle_stats.record_infer(time.monotonic() - t_inf)
        
        processed = process_single_frame_v3(
            frame, res, booth_cache_ref, last_booth_time_ref,
            vt, frame_width, frame_height, debug, proximity_tracker
        )
        processed["video_time"] = vt
        frame_data.append(processed)

    # 2. Burst Logic Decision
    now = time.monotonic()
    
    # VACANT: All 3 frames must be empty
    burst_vacant = all(f["is_vacant_frame"] for f in frame_data)
    
    # BREACH: Alert if at least 1 of 3 frames show breach (sensitive mode)
    breach_count = sum(1 for f in frame_data if f["is_breach_frame"])
    burst_breach = (breach_count >= BREACH_FRAMES_REQUIRED)
    
    # MAX PERSON: At least 2 of 3 frames
    max_count = sum(1 for f in frame_data if f["is_max_frame"])
    burst_max = (max_count >= 2)

    # 3. SELECT THE BEST TRIGGER FRAME (Worst Case Selection)
    # Default to the middle frame
    trigger_frame = frame_data[len(frame_data)//2]

    if burst_breach:
        # Loop through burst and pick the frame with the MOST people at the EVM
        max_intruders = -1
        for f in frame_data:
            if f["is_breach_frame"] and f["behind"] > max_intruders:
                max_intruders = f["behind"]
                trigger_frame = f
    elif burst_max:
        # Pick the frame with the absolute highest number of people detected
        max_people = -1
        for f in frame_data:
            if f["total"] > max_people:
                max_people = f["total"]
                trigger_frame = f
    # For Vacant alerts, the default middle frame is fine because all 3 are empty

    # 4. Vacant Alert Logic
    final_vacant = False
    if in_vacant_cooldown_ref[0] and (now - last_vacant_time_ref[0] >= cooldown_seconds):
        in_vacant_cooldown_ref[0] = False
    if burst_vacant and not in_vacant_cooldown_ref[0]:
        final_vacant = True
        in_vacant_cooldown_ref[0] = True
        last_vacant_time_ref[0] = now

    # 5. Breach Alert Logic
    final_breach = burst_breach
    if burst_breach:
        last_alert_time_ref[0] = now

    # 6. Max Person Alert Logic
    final_max = False
    if in_max_person_cooldown_ref[0] and (now - last_max_person_time_ref[0] >= cooldown_seconds):
        in_max_person_cooldown_ref[0] = False
    if burst_max and not in_max_person_cooldown_ref[0]:
        final_max = True
        in_max_person_cooldown_ref[0] = True
        last_max_person_time_ref[0] = now

    # 7. Return the results
    return {
        "frames": frame_data,
        "trigger_frame": trigger_frame, # The "Worst Case" evidence image
        "alerts": {
            "vacant": final_vacant,
            "breach": final_breach,
            "max": final_max
        }
    }
# ============================================================
# MODE A — FOLDER
# ============================================================

def _grab_consecutive_frames(cap, seek_s: float, count: int):
    """Seeks to a timestamp and grabs 'count' frames in a row."""
    cap.set(cv2.CAP_PROP_POS_MSEC, seek_s * 1000.0)
    frames_list = []
    for _ in range(count):
        ret, frame = cap.read()
        if not ret:
            break
        # Get the actual time of the frame for logging/metadata
        v_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        frames_list.append((frame, v_time))
    return frames_list


async def video_coroutine_folder(
    input_path, cam_id, engine, executor, upload_executor, shutdown, debug, cycle_stats
):
    loop = asyncio.get_event_loop()
    cap  = cv2.VideoCapture(input_path)
    if not cap.isOpened(): return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    bcr=[None]; lbtr=[0.0]; latr=[0.0]; icr=[False]
    lvtr=[0.0]; ivcr=[False]; lmptr=[0.0]; impr=[False]
    pt = ProximityTracker()

    cycle_idx = 0

    def _read_frame():
        cap.set(cv2.CAP_PROP_POS_MSEC, cycle_idx * FOLDER_POLL_INTERVAL_SEC * 1000.0)
        ret, frame = cap.read()
        return frame if ret else None

    try:
        while not shutdown.is_set():
            frame = await loop.run_in_executor(executor, _read_frame)
            if frame is None:
                break
            cycle_stats.record_grab(0.1, success=True)

            fut = engine.submit(frame)
            t_inf = time.monotonic()
            res = await fut
            cycle_stats.record_infer(time.monotonic() - t_inf)

            result = process_single_frame_v3(
                frame, res, bcr, lbtr,
                time.monotonic(), w, h, debug, pt
            )

            ts = datetime.now() + timedelta(hours=5, minutes=30)

            if result["is_breach_frame"]:
                cycle_stats.record_alert("breach")
                await loop.run_in_executor(upload_executor,
                    partial(save_alert_image, cam_id,
                    result["annotated"], ts,
                    {"total": result["total"], "behind_count": result["behind"]},
                    "booth", None, cycle_idx + 1, cycle_stats, save_local=True))

            if result["is_vacant_frame"] and not ivcr[0]:
                ivcr[0] = True; lvtr[0] = time.monotonic()
                cycle_stats.record_alert("vacant")
                await loop.run_in_executor(upload_executor,
                    partial(save_alert_image, cam_id,
                    result["annotated"], ts,
                    {"total": result["total"], "behind_count": result["behind"]},
                    "vacant", ANALYTICS_ID_VACANT, cycle_idx + 1, cycle_stats, save_local=True))
            elif not result["is_vacant_frame"]:
                ivcr[0] = False

            if result["is_max_frame"] and not impr[0]:
                impr[0] = True; lmptr[0] = time.monotonic()
                cycle_stats.record_alert("max_person")
                await loop.run_in_executor(upload_executor,
                    partial(save_alert_image, cam_id,
                    result["annotated"], ts,
                    {"total": result["total"], "behind_count": result["behind"]},
                    "max-person", ANALYTICS_ID_MAX_PERSON, cycle_idx + 1, cycle_stats, save_local=True))
            elif not result["is_max_frame"]:
                impr[0] = False

            cycle_idx += 1
    finally:
        cap.release()

async def async_folder_main(folder_path: str, debug: bool):
    video_list = get_all_video_files(folder_path)
    if not video_list:
        log.error(f"No videos found in {folder_path}."); return

    n = len(video_list)
    log.info(f"[MAIN] Folder mode: {n} video(s)" + (" | DEBUG" if debug else ""))
    models = load_models("FOLDER")
    if models[0] is None:
        log.error("[MAIN] Model loading failed — aborting."); return

    grab_pool   = ThreadPoolExecutor(max_workers=min(n,64), thread_name_prefix="grab")
    upload_pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix="upload")
    shutdown    = asyncio.Event(); cycle_stats = CycleStats()
    loop        = asyncio.get_event_loop()

    def _sig(signum, _frame):
        loop.call_soon_threadsafe(shutdown.set)
    signal.signal(signal.SIGINT, _sig); signal.signal(signal.SIGTERM, _sig)

    engine = UnifiedInferenceEngine(models)
    tasks  = []
    for vp in video_list:
        cid = os.path.splitext(os.path.basename(vp))[0]
        tasks.append(asyncio.create_task(
            video_coroutine_folder(vp, cid, engine, grab_pool, upload_pool,
                                   shutdown, debug, cycle_stats), name=f"vid-{cid}"))

    done, pending = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)
    for t in pending:
        t.cancel()
    await asyncio.gather(*pending, return_exceptions=True)

    snap = cycle_stats.snapshot_and_reset()
    log.info(f"[FOLDER] FINAL SUMMARY  wall={snap['cycle_wall_s']:.1f}s  |  "
             f"grabbed={snap['grabs_ok']}  failed={snap['grabs_fail']}  |  "
             f"infers={snap['infers_ok']}  |  breach={snap['alerts_breach']}  "
             f"vacant={snap['alerts_vacant']}  max_person={snap['alerts_max_person']}")
    engine.stop(); grab_pool.shutdown(wait=False); upload_pool.shutdown(wait=False)


def run_folder_mode(folder_path: str, debug: bool):
    asyncio.run(async_folder_main(folder_path, debug))


# ============================================================
# MODE B — CSV
# ============================================================

def grab_single_frame(url: str) -> Optional[np.ndarray]:
    """
    Connects to the stream once, grabs exactly 1 frame, returns BGR numpy array.
    Returns None on any failure — never raises.
    """
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-fflags",           "nobuffer+discardcorrupt",
        "-flags",            "low_delay",
        "-rw_timeout",       str(FFMPEG_TIMEOUT_SEC * 1_000_000),
        "-analyzeduration",  "50000",
        "-probesize",        "100000",
        "-threads",          "1",
        "-i",                url,
        "-frames:v",         "1",
        "-f",                "image2",
        "-vcodec",           "mjpeg",
        "pipe:1",
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            out, err = proc.communicate(timeout=FFMPEG_TIMEOUT_SEC + 2)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.communicate(); return None

        if proc.returncode != 0 or not out:
            return None

        arr = np.frombuffer(out, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception as exc:
        log.debug(f"grab_single_frame exception [{url[-30:]}]: {exc}")
        return None


def _cam_key(url: str) -> str:
    return url.rstrip("/").split("/")[-1].strip().rstrip(",")


class CameraState:
    def __init__(self):
        self.booth_cache_ref            = [None]
        self.last_booth_time_ref        = [0.0]
        self.breach_frame_count_ref     = [0]
        self.last_alert_time_ref        = [0.0]
        self.in_cooldown_ref            = [False]
        self.last_vacant_time_ref       = [0.0]
        self.in_vacant_cooldown_ref     = [False]
        self.last_max_person_time_ref   = [0.0]
        self.in_max_person_cooldown_ref = [False]
        self.proximity_tracker          = ProximityTracker()
        self.consecutive_fails          = 0


async def camera_coroutine_csv(
    url, cam_id, state, engine, executor, upload_executor, shutdown, debug, cycle_stats,
    stagger_offset: float = 0.0,
):
    loop = asyncio.get_event_loop()
    scheduled_at = time.monotonic() + stagger_offset

    while not shutdown.is_set():
        # 1. Wait for 30s interval
        now = time.monotonic(); sleep_for = scheduled_at - now

        # Skip missed cycles if we fell behind — re-align to next boundary
        if sleep_for < 0:
            skipped = int(-sleep_for / POLL_INTERVAL_SEC) + 1
            scheduled_at += skipped * POLL_INTERVAL_SEC
            sleep_for = scheduled_at - time.monotonic()

        if sleep_for > 0:
            try: await asyncio.wait_for(shutdown.wait(), timeout=sleep_for); break
            except asyncio.TimeoutError: pass
        if shutdown.is_set(): break
        scheduled_at += POLL_INTERVAL_SEC

        # 2. Grab single frame
        t_grab_start = time.monotonic()
        frame = await loop.run_in_executor(executor, grab_single_frame, url)
        grab_elapsed = time.monotonic() - t_grab_start

        if frame is None:
            state.consecutive_fails += 1
            cycle_stats.record_grab(grab_elapsed, success=False)
            continue

        state.consecutive_fails = 0
        cycle_stats.record_grab(grab_elapsed, success=True)
        h_px, w_px = frame.shape[:2]

        # 3. Fire-and-forget: inference + processing + upload run in background,
        # so the grab loop keeps ticking on the 30s grid regardless of GPU or Azure.
        async def _process_frame_bg(frame=frame, h_px=h_px, w_px=w_px):
            t_inf = time.monotonic()
            try:
                res = await engine.submit(frame)
            except Exception as e:
                log.warning(f"[{cam_id}] Inference error: {e}")
                return
            cycle_stats.record_infer(time.monotonic() - t_inf)

            result = process_single_frame_v3(
                frame, res,
                state.booth_cache_ref, state.last_booth_time_ref,
                time.monotonic(), w_px, h_px, debug, state.proximity_tracker
            )

            now_bg = time.monotonic()
            ts_bg  = datetime.now() + timedelta(hours=5, minutes=30)

            # Breach alert (no cooldown — fires every cycle like before)
            if result["is_breach_frame"]:
                cycle_stats.record_alert("breach")
                loop.run_in_executor(upload_executor, save_alert_image, cam_id,
                    result["annotated"], ts_bg,
                    {"total": result["total"], "behind_count": result["behind"]},
                    "booth", None, cycle_stats.current_cycle, cycle_stats)

            # Vacant alert (with cooldown)
            if state.in_vacant_cooldown_ref[0] and (now_bg - state.last_vacant_time_ref[0] >= ALERT_COOLDOWN_SEC):
                state.in_vacant_cooldown_ref[0] = False
            if result["is_vacant_frame"] and not state.in_vacant_cooldown_ref[0]:
                state.in_vacant_cooldown_ref[0] = True
                state.last_vacant_time_ref[0] = now_bg
                cycle_stats.record_alert("vacant")
                loop.run_in_executor(upload_executor, save_alert_image, cam_id,
                    result["annotated"], ts_bg,
                    {"total": result["total"], "behind_count": result["behind"]},
                    "vacant", ANALYTICS_ID_VACANT, cycle_stats.current_cycle, cycle_stats)

            # Max person alert (with cooldown)
            if state.in_max_person_cooldown_ref[0] and (now_bg - state.last_max_person_time_ref[0] >= ALERT_COOLDOWN_SEC):
                state.in_max_person_cooldown_ref[0] = False
            if result["is_max_frame"] and not state.in_max_person_cooldown_ref[0]:
                state.in_max_person_cooldown_ref[0] = True
                state.last_max_person_time_ref[0] = now_bg
                cycle_stats.record_alert("max_person")
                loop.run_in_executor(upload_executor, save_alert_image, cam_id,
                    result["annotated"], ts_bg,
                    {"total": result["total"], "behind_count": result["behind"]},
                    "max-person", ANALYTICS_ID_MAX_PERSON, cycle_stats.current_cycle, cycle_stats)

        asyncio.create_task(_process_frame_bg())

async def watchdog_csv(cycle_stats, shutdown, active_urls_ref, interval=POLL_INTERVAL_SEC):
    """
    Monitors the health of the pipeline.
    In Burst Mode, 'grabs_ok' represents the number of cameras successfully connected.
    """
    cycle_num = 1
    while not shutdown.is_set():
        try:
            # Wait for the 30-second interval
            await asyncio.wait_for(shutdown.wait(), timeout=interval)
            break # Exit loop if shutdown is triggered
        except asyncio.TimeoutError:
            pass
        
        try:
            # Get stats and reset counters for the next cycle
            snap = cycle_stats.snapshot_and_reset()
            
            # Update the global cycle number for folder naming
            cycle_stats.current_cycle = cycle_num + 1
            
            wall = snap["cycle_wall_s"]
            status = "OK" if wall <= POLL_INTERVAL_SEC else f"OVERRUN +{wall-POLL_INTERVAL_SEC:.1f}s"
            
            # In Burst Mode, success is based on n_cams (not n_cams * 3)
            # NEW (single-frame, straightforward)
            n_cams    = len(active_urls_ref)
            cams_ok   = snap['grabs_ok']
            cams_fail = n_cams - cams_ok  # all cameras that didn't succeed
            
            # 1. Console Logging (Clean & Actionable)
            log.info(f"[WATCH] ── cycle={cycle_num}  wall={wall:.1f}s  [{status}]")
            log.info(f"[WATCH]   Cameras : OK={cams_ok:<3}  FAIL={cams_fail:<3}  (Total={n_cams})")
            log.info(f"[WATCH]   Latency : Grab_avg={snap['grab_avg_ms']:.0f}ms  Infer_avg={snap['infer_avg_ms']:.0f}ms")
            log.info(f"[WATCH]   Alerts  : Breach={snap['alerts_breach']}  Vacant={snap['alerts_vacant']}  Max={snap['alerts_max_person']}")
            log.info(f"[WATCH]   API     : Success={snap['alerts_ok']}  Failed={snap['alerts_fail']}")

            # 2. Permanent File Logging (cycle.log)
            ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                with open("cycle.log", "a", encoding="utf-8") as f:
                    f.write(
                        f"{ts_str} | Cycle {cycle_num:04d} | {status:<12} | "
                        f"Cams: {cams_ok}/{n_cams} | "
                        f"Grab_Avg: {snap['grab_avg_ms']:>5.0f}ms | "
                        f"Infer_Avg: {snap['infer_avg_ms']:>4.0f}ms | "
                        f"Alerts: OK={snap['alerts_ok']} FAIL={snap['alerts_fail']} (B={snap['alerts_breach']}, V={snap['alerts_vacant']}, M={snap['alerts_max_person']})\n"
                    )
            except Exception as e:
                log.error(f"Failed to write to cycle.log: {e}")

            cycle_num += 1
        except Exception as e:
            log.error(f"[WATCH] Error in watchdog cycle {cycle_num}: {e}")
            log.error(traceback.format_exc())
            cycle_num += 1

class CSVChangeHandler(FileSystemEventHandler):
    """Detects when CSV file is saved and triggers camera reload"""
    
    def __init__(self, csv_path: str, reload_callback):
        self.csv_path = os.path.abspath(csv_path)
        self.reload_callback = reload_callback
        self.last_modified = 0
        self.debounce_seconds = 2  # Wait 2 sec to avoid duplicate triggers
        
    def on_modified(self, event):
        if event.is_directory:
            return
        
        # Only trigger on our specific CSV file
        if os.path.abspath(event.src_path) == self.csv_path:
            now = time.time()
            if now - self.last_modified > self.debounce_seconds:
                self.last_modified = now
                log.info(f"[CSV-WATCH] 📝 CSV saved - reloading cameras now!")
                if self.reload_callback:
                    self.reload_callback()

async def async_csv_main(csv_path: str, debug: bool):
    log.info(f"[MAIN] [CSV] Starting with SYNCHRONIZED CYCLES — {csv_path}")
    
    if not WATCHDOG_AVAILABLE:
        log.warning("[MAIN] pip install watchdog for instant CSV updates")
    
    models = load_models("CSV-MAIN")
    if models[0] is None:
        log.error("[MAIN] Model loading failed"); return
 
    ffmpeg_pool   = ThreadPoolExecutor(max_workers=FFMPEG_POOL_WORKERS, thread_name_prefix="ffmpeg")
    upload_pool   = ThreadPoolExecutor(max_workers=UPLOAD_POOL_WORKERS, thread_name_prefix="upload")
    shutdown      = asyncio.Event()
    cycle_stats   = CycleStats()
    loop          = asyncio.get_event_loop()
    
    running_tasks = {}
    active_urls   = []
 
    def _sig(signum, _frame):
        loop.call_soon_threadsafe(shutdown.set)
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
 
    engine = UnifiedInferenceEngine(models)
    watchdog_task = asyncio.create_task(
        watchdog_csv(cycle_stats, shutdown, active_urls), 
        name="watchdog"
    )
 
    # ═══════════════════════════════════════════════════════════════════
    # NEW: Cycle synchronization
    # ═══════════════════════════════════════════════════════════════════
    reload_requested = asyncio.Event()
    cycle_boundary = asyncio.Event()  # NEW: Signals when cycle completes
    
    def trigger_reload():
        """Called when CSV file is saved"""
        loop.call_soon_threadsafe(reload_requested.set)
    
    # Start file system watcher
    file_observer = None
    if WATCHDOG_AVAILABLE:
        try:
            csv_dir = os.path.dirname(os.path.abspath(csv_path))
            handler = CSVChangeHandler(csv_path, trigger_reload)
            file_observer = Observer()
            file_observer.schedule(handler, csv_dir, recursive=False)
            file_observer.start()
            log.info(f"[MAIN] 👁️  Watching CSV: {os.path.basename(csv_path)}")
        except Exception as e:
            log.error(f"[MAIN] File watcher failed: {e}")
 
    last_reload = -float('inf')
    is_initial_run = True
    
    # ═══════════════════════════════════════════════════════════════════
    # NEW: Track cycle timing for synchronization
    # ═══════════════════════════════════════════════════════════════════
    cycle_start_time = time.monotonic()
    pending_additions = []  # Cameras waiting for next cycle
    pending_removals  = set() # Cameras to stop at next cycle
 
    try:
        while not shutdown.is_set():
            now = time.monotonic()
            
            # ───────────────────────────────────────────────────────────
            # Check if we're at a cycle boundary
            # ───────────────────────────────────────────────────────────
            time_since_cycle_start = now - cycle_start_time
            at_cycle_boundary = time_since_cycle_start >= POLL_INTERVAL_SEC
            
            if at_cycle_boundary:
                cycle_start_time = now
                cycle_boundary.set()  # Signal that new cycle is starting
                
                # 1. STOP pending removals FIRST
                if pending_removals:
                    log.info(f"[MAIN] ❌ Stopping {len(pending_removals)} cameras at cycle boundary")
                    for u in list(pending_removals):
                        if u in running_tasks:
                            #log.info(f"[MAIN]   - {_cam_key(u)}")
                            running_tasks[u].cancel()
                            del running_tasks[u]
                        # Also ensure they aren't in pending_additions
                        pending_additions = [p for p in pending_additions if p['url'] != u]
                    pending_removals.clear()

                # 2. START pending additions
                if pending_additions:
                    log.info(f"[MAIN] 🚀 Starting {len(pending_additions)} cameras at cycle boundary")
                    for task_info in pending_additions:
                        running_tasks[task_info['url']] = task_info['task']
                    pending_additions.clear()
                
                active_urls[:] = list(running_tasks.keys())
                cycle_boundary.clear()
            
            # ───────────────────────────────────────────────────────────
            # Check for CSV reload triggers
            # ───────────────────────────────────────────────────────────
            should_reload = False
            
            if reload_requested.is_set():
                should_reload = True
                reload_requested.clear()
                log.info("[MAIN] 🔄 CSV changed - updating cameras at next cycle...")
            elif now - last_reload >= CSV_RELOAD_INTERVAL_SEC:
                should_reload = True
                log.info("[MAIN] 🔄 Periodic check (10 min)")
            
            if should_reload:
                try:
                    new_urls = load_camera_urls(csv_path)
                    current_set = set(new_urls)
                    running_set = set(running_tasks.keys())
                    pending_set = {p['url'] for p in pending_additions}
                    all_managed = running_set | pending_set
 
                    # ───────────────────────────────────────────────────
                    # QUEUE removals for next cycle boundary
                    # ───────────────────────────────────────────────────
                    to_remove = all_managed - current_set
                    if to_remove:
                        log.info(f"[MAIN] ⏳ Queueing {len(to_remove)} removals for next cycle boundary")
                        pending_removals.update(to_remove)

                    # ───────────────────────────────────────────────────
                    # QUEUE additions for next cycle boundary
                    # ───────────────────────────────────────────────────
                    to_add = current_set - all_managed
                    if to_add:
                        # Clear them from pending removals if they were just added back
                        pending_removals -= to_add
                        
                        next_cycle_wait = POLL_INTERVAL_SEC - time_since_cycle_start
                        log.info(f"[MAIN] ⏳ Queueing {len(to_add)} additions (start in {next_cycle_wait:.1f}s)")
                        
                        # Spread camera starts across one full poll interval
                        # so ffmpeg pool sees a steady drip, never a thundering herd
                        to_add_list = list(to_add)
                        n_adding = max(len(to_add_list), 1)
                        stagger_step = POLL_INTERVAL_SEC / n_adding

                        for i, u in enumerate(to_add_list):
                            cid = _cam_key(u)
                            state = CameraState()
                            stagger = i * stagger_step
                            #log.info(f"[MAIN]   + {cid} [WAITING]")

                            # Create task
                            async def _start_cam_synced(url_in=u, cid_in=cid, st_in=state,
                                                        skip_wait=is_initial_run, stag=stagger):
                                if not skip_wait:
                                    try: await cycle_boundary.wait()
                                    except: return

                                if not shutdown.is_set():
                                    await camera_coroutine_csv(
                                        url_in, cid_in, st_in, engine,
                                        ffmpeg_pool, upload_pool, shutdown,
                                        debug, cycle_stats,
                                        stagger_offset=stag,
                                    )

                            task = asyncio.create_task(_start_cam_synced(), name=f"cam-{cid}")
                            if is_initial_run:
                                running_tasks[u] = task
                            else:
                                pending_additions.append({'url': u, 'task': task, 'cid': cid})
 
                    # Update active list
                    active_urls[:] = list(running_tasks.keys())
                    last_reload = now
                    is_initial_run = False
                    
                    total_managed = len(running_tasks) + len(pending_additions)
                    log.info(f"[MAIN] 📊 Active: {len(running_tasks)} | Pending: {len(pending_additions)} | Total: {total_managed}")
 
                except Exception as e:
                    log.error(f"[MAIN] Reload error: {e}")
 
            # Wait for triggers
            try:
                wait_shutdown = asyncio.ensure_future(shutdown.wait())
                wait_reload   = asyncio.ensure_future(reload_requested.wait())
                done, pending = await asyncio.wait(
                    [wait_shutdown, wait_reload],
                    timeout=1,  # Check more frequently for cycle boundaries
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for p in pending:
                    p.cancel()
                if shutdown.is_set():
                    break
            except asyncio.TimeoutError:
                pass
 
    finally:
        log.info("[MAIN] Shutting down...")
        
        if file_observer:
            try:
                file_observer.stop()
                file_observer.join(timeout=2)
            except RuntimeError:
                pass
        
        watchdog_task.cancel()
        for t in running_tasks.values(): 
            t.cancel()
        for p in pending_additions:
            p['task'].cancel()
        
        all_tasks = list(running_tasks.values()) + [p['task'] for p in pending_additions]
        await asyncio.gather(watchdog_task, *all_tasks, return_exceptions=True)
        
        engine.stop()
        ffmpeg_pool.shutdown(wait=False)
        upload_pool.shutdown(wait=False)
        log.info("[MAIN] Stopped")


def run_csv_mode(csv_path: str, debug: bool):
    try:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        log.info("[MAIN] [CSV] uvloop enabled")
    except ImportError:
        pass
    asyncio.run(async_csv_main(csv_path, debug))


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    args       = parse_args()
    input_path = INPUT_PATH.strip()

    if not os.path.exists(input_path):
        log.error(f"INPUT_PATH does not exist: {input_path}")
        sys.exit(1)

    if os.path.isfile(input_path) and input_path.lower().endswith(".csv"):
        log.info(f"[MAIN] → CSV / stream mode  ({input_path})")
        run_csv_mode(input_path, args.debug)
    elif os.path.isdir(input_path):
        log.info(f"[MAIN] → Folder / local-video mode  ({input_path})")
        run_folder_mode(input_path, args.debug)
    else:
        log.error("INPUT_PATH must be a .csv file or a directory.")
        sys.exit(1)
