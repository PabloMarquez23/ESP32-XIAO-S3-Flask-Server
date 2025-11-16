# processing/pipes.py
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# =========================
# Configuración del pipeline
# =========================
@dataclass
class PipelineConfig:
    # Iluminación
    clahe: bool = True
    clipLimit: float = 2.0
    tiles: int = 8  # para CLAHE (tiles x tiles)

    # Ruido
    noise: str = "none"  # "none" | "gauss" | "speckle"
    gauss_std: float = 10.0
    speckle_var: float = 0.05

    # Suavizados
    blur_k: int = 5
    median_k: int = 5
    gauss_k: int = 5

    # Convolución PyTorch
    use_torch_conv: bool = True
    kernel: int = 1            # 0..3
    kernel_strength: float = 1.0  # factor de intensidad del kernel

    # Detectores de borde
    sobel: bool = True
    canny: bool = False

    def update_from_dict(self, d: Dict[str, Any]) -> None:
        # Casters seguros
        def as_bool(v): return bool(v) if not isinstance(v, str) else (v.lower() in ("1", "true", "on", "yes"))
        def as_int(v):  return int(float(v))
        def as_float(v):
            try:
                return float(v)
            except Exception:
                return 0.0

        if "clahe" in d: self.clahe = as_bool(d["clahe"])
        if "clipLimit" in d: self.clipLimit = as_float(d["clipLimit"])
        if "tiles" in d: self.tiles = max(2, as_int(d["tiles"]))

        if "noise" in d: self.noise = str(d["noise"])
        if "gauss_std" in d: self.gauss_std = max(0.0, as_float(d["gauss_std"]))
        if "speckle_var" in d: self.speckle_var = max(0.0, as_float(d["speckle_var"]))

        if "blur_k" in d: self.blur_k = self._odd(as_int(d["blur_k"]))
        if "median_k" in d: self.median_k = self._odd(as_int(d["median_k"]))
        if "gauss_k" in d: self.gauss_k = self._odd(as_int(d["gauss_k"]))

        if "use_torch_conv" in d: self.use_torch_conv = as_bool(d["use_torch_conv"])
        if "kernel" in d: self.kernel = max(0, min(3, as_int(d["kernel"])))
        if "kernel_strength" in d: self.kernel_strength = float(as_float(d["kernel_strength"]))

        if "sobel" in d: self.sobel = as_bool(d["sobel"])
        if "canny" in d: self.canny = as_bool(d["canny"])

    @staticmethod
    def _odd(k: int) -> int:
        # asegura que sea impar y >=1
        if k < 1: k = 1
        return k if k % 2 == 1 else k + 1


# =========================
# Convoluciones en PyTorch
# =========================
class TorchConvolution:
    """
    Varios kernels de ejemplo:
      0: Edge 3x3
      1: Sharpen
      2: Emboss
      3: LoG 5x5 (Laplaciano de Gauss)
    """
    NAMES = {
        0: "Edge3x3",
        1: "Sharpen",
        2: "Emboss",
        3: "LoG5x5",
    }

    def __init__(self) -> None:
        self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self._kernels = {}
        self._build_kernels()

    def _build_kernels(self):
        # 0) realce simple de bordes 3x3
        k0 = torch.tensor(
            [[0, -1,  0],
             [-1, 4, -1],
             [0, -1,  0]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)  # (outC, inC, H, W)

        # 1) sharpen 3x3
        k1 = torch.tensor(
            [[0, -1,  0],
             [-1, 5, -1],
             [0, -1,  0]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)

        # 2) emboss 3x3
        k2 = torch.tensor(
            [[-2, -1, 0],
             [-1,  1, 1],
             [ 0,  1, 2]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)

        # 3) LoG 5x5 (aprox)
        k3 = torch.tensor(
            [[0, 0, -1, 0, 0],
             [0, -1, -2, -1, 0],
             [-1, -2, 16, -2, -1],
             [0, -1, -2, -1, 0],
             [0, 0, -1, 0, 0]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)

        for i, k in enumerate([k0, k1, k2, k3]):
            self._kernels[i] = k.to(self._device)

    def apply(self, gray_u8: np.ndarray, kernel_id: int = 1, strength: float = 1.0) -> np.ndarray:
        k = self._kernels.get(kernel_id, self._kernels[1]) * float(strength)
        t = torch.from_numpy(gray_u8).float().unsqueeze(0).unsqueeze(0).to(self._device)  # (1,1,H,W)
        y = F.conv2d(t, k, padding=k.shape[-1] // 2)
        y = y.squeeze().detach().to("cpu").numpy()
        # Escalamos a 0..255 para visualizar (abs para que se noten los negativos)
        y = np.abs(y)
        y = cv2.normalize(y, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return y


# =========================
# Pipeline principal
# =========================
class ProcessingPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.cfg = config
        self.bg = cv2.createBackgroundSubtractorMOG2()  # para 1A
        self.torch_conv = TorchConvolution()

        self.prev_t = time.time()
        self.fps = 0.0

        # Reusar objetos (evita GC en loop)
        self._clahe = None
        self._clahe_tiles = None
        self._clahe_clip = None

    # ---------- helpers ----------
    def _update_fps(self):
        now = time.time()
        dt = max(1e-6, now - self.prev_t)
        self.fps = 1.0 / dt
        self.prev_t = now

    def _apply_lighting(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Equalize/CLAHE opcional (1A Iluminación). Devuelve BGR."""
        # Trabajamos en YCrCb: ecualizamos Y y reensamblamos
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        y_eq = cv2.equalizeHist(y)  # baseline
        if self.cfg.clahe:
            # recrear si cambian parámetros
            tiles = (max(2, self.cfg.tiles), max(2, self.cfg.tiles))
            if (self._clahe is None or
                self._clahe_clip != self.cfg.clipLimit or
                self._clahe_tiles != tiles):
                self._clahe = cv2.createCLAHE(clipLimit=float(self.cfg.clipLimit),
                                              tileGridSize=tiles)
                self._clahe_clip = float(self.cfg.clipLimit)
                self._clahe_tiles = tiles
            y_eq = self._clahe.apply(y)

        ycrcb_eq = cv2.merge([y_eq, cr, cb])
        return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

    def _add_noise(self, img_bgr: np.ndarray) -> np.ndarray:
        if self.cfg.noise == "none":
            return img_bgr
        h, w, c = img_bgr.shape
        imgf = img_bgr.astype(np.float32)

        if self.cfg.noise == "gauss":
            std = max(0.0, float(self.cfg.gauss_std))
            noise = np.random.normal(0.0, std, (h, w, c)).astype(np.float32)
            out = imgf + noise
        elif self.cfg.noise == "speckle":
            var = max(0.0, float(self.cfg.speckle_var))
            noise = np.random.randn(h, w, c).astype(np.float32)
            out = imgf + imgf * noise * var
        else:
            return img_bgr

        return np.clip(out, 0, 255).astype(np.uint8)

    def _denoise_pack(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        k1 = self.cfg._odd(self.cfg.blur_k)
        k2 = self.cfg._odd(self.cfg.median_k)
        k3 = self.cfg._odd(self.cfg.gauss_k)
        blur = cv2.blur(img_bgr, (k1, k1))
        median = cv2.medianBlur(img_bgr, k2)
        gauss = cv2.GaussianBlur(img_bgr, (k3, k3), 0)
        return blur, median, gauss

    def _edges_pack(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        sob = cv2.Sobel(gray, cv2.CV_16S, 1, 1, ksize=3)
        sob = cv2.convertScaleAbs(sob)
        can = cv2.Canny(gray, 60, 120)
        return sob, can

    # ---------- API principal ----------
    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Devuelve un mosaico 2x2 con: original/primer plano/convTorch/bordes."""
        self._update_fps()

        # 1) Iluminación (equalize/CLAHE)
        lit = self._apply_lighting(frame_bgr)

        # 2) Ruido (si aplica) + conjunto de suavizados para comparar (1B)
        noisy = self._add_noise(lit)
        _, _, gauss = self._denoise_pack(noisy)

        # Para mostrar "primer plano" (1A)
        mask = self.bg.apply(lit)
        fg = cv2.bitwise_and(lit, lit, mask=mask)

        # 3) Convolución PyTorch (1B – filtro distinto al ejemplo)
        gray = cv2.cvtColor(gauss, cv2.COLOR_BGR2GRAY)
        if self.cfg.use_torch_conv:
            conv = self.torch_conv.apply(
                gray,
                kernel_id=int(self.cfg.kernel),
                strength=float(self.cfg.kernel_strength),
            )
        else:
            conv = gray.copy()

        conv_bgr = cv2.cvtColor(conv, cv2.COLOR_GRAY2BGR)

        # 4) Bordes (al menos dos detectores) (1B)
        sob, can = self._edges_pack(gray)
        edge = np.zeros_like(conv_bgr)
        if self.cfg.sobel:
            edge = cv2.max(edge, cv2.cvtColor(sob, cv2.COLOR_GRAY2BGR))
        if self.cfg.canny:
            edge = cv2.max(edge, cv2.cvtColor(can, cv2.COLOR_GRAY2BGR))

        # 5) Overlays de texto
        self._put_hud(lit, f"FPS:{self.fps:.1f} | CLAHE:{self.cfg.clahe} | Noise:{self.cfg.noise}")
        self._put_hud(fg, "Foreground (MOG2)")
        kname = TorchConvolution.NAMES.get(int(self.cfg.kernel), str(self.cfg.kernel))
        self._put_hud(conv_bgr, f"PyTorch(kernel={kname}, s={self.cfg.kernel_strength:.2f})")
        self._put_hud(edge, f"Sobel:{self.cfg.sobel}  Canny:{self.cfg.canny}")

        # 6) Mosaico 2x2
        top = np.hstack(self._safe_same_size([lit, fg]))
        bottom = np.hstack(self._safe_same_size([conv_bgr, edge]))
        out = np.vstack(self._safe_same_size([top, bottom]))
        return out

    def encode_jpeg(self, img_bgr: np.ndarray, quality: int = 85) -> Tuple[bool, Optional[bytes]]:
        ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return False, None
        return True, buf.tobytes()

    # ---------- utilidades ----------
    def _put_hud(self, img: np.ndarray, text: str, color=(0, 255, 0)) -> None:
        cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    def _safe_same_size(self, imgs: list) -> list:
        # redimensiona al tamaño del primero (para mosaico)
        h, w = imgs[0].shape[:2]
        out = []
        for im in imgs:
            if im.shape[:2] != (h, w):
                out.append(cv2.resize(im, (w, h)))
            else:
                out.append(im)
        return out
