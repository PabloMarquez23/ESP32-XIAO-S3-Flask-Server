# morfologia.py
import cv2
import numpy as np
import pydicom
from matplotlib import pyplot as plt
from pathlib import Path

# === CONFIGURACIÓN ===
# Ruta base donde tienes tus imágenes DICOM (.IMA o .dcm)
BASE_DIR = Path("/home/pablomar/vision-interciclo/data/CT_low_dose_reconstruction_dataset/Original Data/Full Dose/3mm Slice Thickness/Sharp Kernel (D45)/L096/full_3mm_sharp")

# Carpeta donde se guardarán los resultados
OUT_DIR = Path("resultados_morfologia")
OUT_DIR.mkdir(exist_ok=True)

# Tamaños de kernel a comparar
KERNEL_SIZES = [5, 15, 37]

# === FUNCIONES ===

def leer_imagen(path: Path):
    """Lee una imagen DICOM (.IMA/.dcm) o PNG/JPG en escala de grises."""
    ext = path.suffix.lower()
    if ext in [".dcm", ".ima"]:
        try:
            ds = pydicom.dcmread(str(path))
            img = ds.pixel_array.astype(np.float32)
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            return img
        except Exception as e:
            print(f"[ERROR] {path.name}: {e}")
            return None
    else:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        return img


def aplicar_morfologia(img: np.ndarray, k: int):
    """Aplica Erosión, Dilatación, TopHat, BlackHat y Combinado."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    erosion = cv2.erode(img, kernel)
    dilatacion = cv2.dilate(img, kernel)
    top_hat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    black_hat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    combinado = cv2.add(top_hat, cv2.subtract(img, black_hat))
    return {
        "Original": img,
        "Erosión": erosion,
        "Dilatación": dilatacion,
        "TopHat": top_hat,
        "BlackHat": black_hat,
        "Combinado": combinado
    }


# === PROCESAMIENTO ===
dicom_files = sorted(BASE_DIR.glob("*.IMA")) or sorted(BASE_DIR.glob("*.dcm"))
print(f"🔍 Se encontraron {len(dicom_files)} archivos en {BASE_DIR}")

if not dicom_files:
    print(" No se encontraron imágenes DICOM. Verifica la ruta.")
    exit()

for i, file in enumerate(dicom_files[:3]):  # Solo procesar 3 imágenes
    img = leer_imagen(file)
    if img is None:
        continue

    print(f" Procesando: {file.name}")

    for k in KERNEL_SIZES:
        resultados = aplicar_morfologia(img, k)

        fig, axes = plt.subplots(2, 3, figsize=(10, 6))
        fig.suptitle(f"{file.name} — kernel {k}x{k}", fontsize=12)

        for ax, (titulo, im) in zip(axes.flatten(), resultados.items()):
            ax.imshow(im, cmap="gray")
            ax.set_title(titulo)
            ax.axis("off")

        plt.tight_layout()
        out_path = OUT_DIR / f"{file.stem}_k{k}.png"
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f" Guardado: {out_path}")
