// static/main.js  (v3)

function send(payload) {
  fetch('/controls', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }).catch(() => {});
}

function readState() {
  const $ = (id) => document.getElementById(id);

  return {
    // Iluminación
    clahe:          !!$('clahe_enable')?.checked,
    clipLimit:      Number($('clahe_clip')?.value || 2),
    tiles:          Number($('clahe_tiles')?.value || 8),

    // Ruido
    noise:          $('noise_type')?.value || 'none',
    gauss_std:      Number($('gauss_std')?.value || 10),
    speckle_var:    Number($('speckle_var')?.value || 0.05),

    // Suavizados
    blur_k:         Number($('blur_ksize')?.value || 5),
    median_k:       Number($('median_ksize')?.value || 5),
    gauss_k:        Number($('gauss_ksize')?.value || 5),

    // PyTorch conv
    use_torch_conv: !!$('torch_enable')?.checked,
    kernel:         Number($('torch_kernel')?.value || 1),

    // Detectores
    sobel:          !!$('sobel_enable')?.checked,
    canny:          !!$('canny_enable')?.checked,
  };
}

function bindControl(id, eventType = 'input') {
  const el = document.getElementById(id);
  if (!el) return;
  el.addEventListener(eventType, () => send(readState()));
}

window.addEventListener('DOMContentLoaded', () => {
  // Vincula sliders y checkboxes
  bindControl('clahe_enable', 'change');
  bindControl('clahe_clip');
  bindControl('clahe_tiles');

  bindControl('noise_type', 'change');
  bindControl('gauss_std');
  bindControl('speckle_var');

  bindControl('blur_ksize');
  bindControl('median_ksize');
  bindControl('gauss_ksize');

  bindControl('torch_enable', 'change');
  bindControl('torch_kernel');

  bindControl('sobel_enable', 'change');
  bindControl('canny_enable', 'change');

  // Enviar estado inicial al cargar
  send(readState());
});
