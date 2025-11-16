# processing/stream.py
import os, cv2, pathlib

def _try_open(target, api=None):
    cap = cv2.VideoCapture(target) if api is None else cv2.VideoCapture(target, api)
    if cap.isOpened(): return cap
    cap.release(); return None

def _is_file(path: str) -> bool:
    try:
        return pathlib.Path(path).is_file()
    except Exception:
        return False

class VideoSource:
    """
    Prioridad:
      1) CAMERA_URL -> prueba CAP_FFMPEG y genérico (HTTP MJPEG, RTSP, etc.)
      2) Archivo local (si CAMERA_URL apunta a .mp4/.avi/.mjpg)
      3) VIDEO_DEVICE (/dev/videoX o índice)
      4) /dev/video0..5 e índices 0..5
    """
    def __init__(self, camera_url: str | None):
        # 1) URL (o ruta de archivo)
        if camera_url:
            # Si es un archivo local, ábrelo directo
            if _is_file(camera_url):
                cap = _try_open(camera_url)
                if cap:
                    self.cap = cap; self.mode = f"FILE:{camera_url}"; return
                raise RuntimeError(f"No se pudo abrir el archivo de video: {camera_url}")

            # URL remota o local (HTTP/RTSP). Prueba FFMPEG y luego genérico
            cap = _try_open(camera_url, cv2.CAP_FFMPEG) or _try_open(camera_url)
            if cap:
                self.cap = cap; self.mode=f"URL:{camera_url}"; return

            raise RuntimeError(
                "No se pudo abrir CAMERA_URL. Verifica que el endpoint esté activo.\n"
                f"CAMERA_URL={camera_url}\n"
                "Si usas ESP32-CAM/mjpg-streamer, prueba en navegador o con:\n"
                "  curl -v <CAMERA_URL> --max-time 3 --output /dev/null"
            )

        # 2) VIDEO_DEVICE (por ej. '/dev/video0' o '0')
        dev_env = os.getenv("VIDEO_DEVICE")
        if dev_env:
            target = int(dev_env) if dev_env.isdigit() else dev_env
            cap = _try_open(target, cv2.CAP_V4L2) or _try_open(target)
            if cap:
                self.cap = cap; self.mode=f"VIDEO_DEVICE:{dev_env}"; return
            raise RuntimeError(f"No se pudo abrir VIDEO_DEVICE={dev_env} con V4L2.")

        # 3) Barrido típico de dispositivos
        for idx in range(0, 6):
            cap = _try_open(f"/dev/video{idx}", cv2.CAP_V4L2)
            if cap: self.cap=cap; self.mode=f"/dev/video{idx}"; return
        for idx in range(0, 6):
            cap = _try_open(idx, cv2.CAP_V4L2)
            if cap: self.cap=cap; self.mode=f"INDEX:{idx}"; return

        raise RuntimeError(
            "No se encontró ninguna fuente de video.\n"
            "Opciones:\n"
            "  - export CAMERA_URL=http://127.0.0.1:8090/?action=stream  (mjpg-streamer)\n"
            "  - export CAMERA_URL=/ruta/a/video.mp4  (archivo)\n"
            "  - export VIDEO_DEVICE=/dev/video0      (webcam USB)\n"
        )

    def frames(self):
        while True:
            ok, frame = self.cap.read()
            if not ok: break
            yield frame
