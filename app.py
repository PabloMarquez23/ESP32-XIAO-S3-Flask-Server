# app.py
import os
from flask import Flask, render_template, Response, request
from processing.stream import VideoSource
from processing.pipes import ProcessingPipeline, PipelineConfig

app = Flask(__name__)

# =====================================
# CONFIGURACIÓN PRINCIPAL
# =====================================
config = PipelineConfig()
pipeline = ProcessingPipeline(config)

_source = None
_source_err = None


def get_source():
    """
    Inicializa la fuente de video.

    Si existe la variable de entorno CAMERA_URL, la usa.
    Si no, usa por defecto el stream de la ESP32:
        http://192.168.18.10/stream
    """
    global _source, _source_err
    if _source is not None:
        return _source

    # ⚠️ AQUÍ ESTABA EL ERROR:
    # No se debe poner la URL como nombre de variable de entorno.
    # Debe ser algo tipo "CAMERA_URL".
    camera_url = os.getenv("CAMERA_URL", "http://192.168.18.10/stream")

    try:
        _source = VideoSource(camera_url)
        _source_err = None
        print(f"[VideoSource] Usando fuente: {camera_url}")
    except Exception as e:
        _source = None
        _source_err = str(e)
        print(f"[VideoSource][ERROR] {e}")
    return _source


# =====================================
# PÁGINA PRINCIPAL
# =====================================
@app.route("/")
def index():
    if _source is None:
        get_source()

    help_msg = None
    if _source is None and _source_err:
        help_msg = f"""No se pudo abrir la fuente de video.<br><pre>{_source_err}</pre>
<br>Prueba:
- Para webcam USB: <code>export VIDEO_DEVICE=/dev/video0</code><br>
- Para archivo local: <code>export CAMERA_URL=/ruta/video.mp4</code><br>
- Para MJPEG por HTTP (ESP32/mjpg-streamer):<br>
  <code>export CAMERA_URL=http://192.168.18.10/stream</code><br>
Luego reinicia: <code>prime-run python app.py</code>"""
    return render_template("index.html", help_msg=help_msg)


# =====================================
# EVITAR CACHÉ EN EL NAVEGADOR
# =====================================
@app.after_request
def add_no_cache_headers(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


# =====================================
# ENDPOINT DE CONTROLES
# =====================================
@app.route("/controls", methods=["POST"])
def controls():
    data = request.json or {}
    print("Controls =>", data)  # Verás esto en la terminal
    config.update_from_dict(data)
    return {"ok": True, "applied": data, "cfg": config.__dict__}


# =====================================
# ENDPOINT DE DEPURACIÓN DE CONFIG
# =====================================
@app.route("/debug_config")
def debug_config():
    """Permite ver la configuración actual del pipeline."""
    return {
        **config.__dict__,
        "camera_url": os.getenv("CAMERA_URL", "http://192.168.18.10/stream"),
    }


# =====================================
# STREAM DE VIDEO MJPEG
# =====================================
@app.route("/video_feed")
def video_feed():
    src = get_source()
    if src is None:
        # En caso de error, devolver texto
        def err():
            msg = (_source_err or "Sin fuente de video").encode("utf-8")
            yield b"--frame\r\nContent-Type: text/plain\r\n\r\n" + msg + b"\r\n"

        return Response(err(), mimetype="multipart/x-mixed-replace; boundary=frame")

    def gen():
        for frame in src.frames():
            processed = pipeline.process(frame)
            ok, jpeg = pipeline.encode_jpeg(processed)
            if not ok or jpeg is None:
                continue

            payload = jpeg if isinstance(jpeg, (bytes, bytearray)) else jpeg.tobytes()
            header = (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n"
            )
            yield header + payload + b"\r\n"

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


# =====================================
# INICIO DE LA APLICACIÓN
# =====================================
if __name__ == "__main__":
    # Ejecutar el servidor Flask
    app.run(
        host=os.getenv("BIND", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        debug=False,
        threaded=True,
    )
