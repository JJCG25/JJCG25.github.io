// index.js

// 0) Arreglo con los nombres de las clases en el mismo orden en que entrenó tu modelo:
const labels = [
  "Apple_iPhone8Plus",
  "Apple_iPhone12",
  "Apple_iPhone13mini",
  "Apple_iPhoneSE",
  "Apple_iPhoneX",
  "DOOGEE_S96Pro",
  "Google_Pixel3a",
  "Google_Pixel5",
  "Huawei_Mate10Lite",
  "Huawei_Mate10Pro",
  "Huawei_Nova5T",
  "Huawei_P8Lite",
  "Huawei_P9Lite",
  "Huawei_P30Lite",
  "LG_G4c",
  "LG_G7ThinQ",
  "LG_V50ThinQ",
  "Motorola_MotoG",
  "Motorola_MotoG5",
  "Motorola_MotoG5SPlus",
  "Motorola_MotoG9Plus",
  "OnePlus_6T",
  "OnePlus_8T",
  "Samsung_GalaxyA12",
  "Samsung_GalaxyA40",
  "Samsung_GalaxyA52s",
  "Samsung_GalaxyNote8",
  "Samsung_GalaxyS6",
  "Samsung_GalaxyS10",
  "Samsung_GalaxyS10+",
  "Samsung_GalaxyS20+",
  "Samsung_GalaxyS21+",
  "Sony_XperiaM2",
  "Xiaomi_MiA2Lite",
  "Xiaomi_MiMix3",
  "Xiaomi_Redmi5Plus",
  "Xiaomi_RedmiNote8",
  "Xiaomi_RedmiNote8T",
  "Xiaomi_RedmiNote9"
];

const ort = window.ort; // ONNX Runtime Web expone "ort" en window

// ───────────────────────────────────────────────────────────────────
// 1) Softmax: convierte un array de logits a probabilidades
function softmax(logits) {
  const maxLogit = Math.max(...logits);
  const exps = logits.map((x) => Math.exp(x - maxLogit));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sumExps);
}

// 2) postprocess: recibe un ort.Tensor con logits [1, num_classes]
//    Aplica softmax, busca el índice con mayor probabilidad y devuelve { classIndex, score }
function postprocess(resultTensor) {
  // resultTensor.data es Float32Array de longitud = num_classes
  const logits = Array.from(resultTensor.data);
  const probs = softmax(logits);

  let maxIdx = 0;
  let maxProb = probs[0];
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > maxProb) {
      maxProb = probs[i];
      maxIdx = i;
    }
  }

  return { classIndex: maxIdx, score: maxProb };
}

document.addEventListener("DOMContentLoaded", async () => {
  const imageContainer = document.querySelector(".custom-file-upload");
  const status = document.getElementById("status");
  const fileUpload = document.getElementById("upload");
  const resetButton = document.getElementById("reset-image");
  const uploadButton = document.getElementById("upload-btn");
  const container = document.querySelector(".container");

  // 3) Cargar el modelo ONNX float32 (no cuantizado)
  status.textContent = "Loading model…";
  const onnxURL =
    "https://huggingface.co/JuanJCalderonG/efficient-net-quant/resolve/main/model_fp32.onnx";
  const preprocURL =
    "https://huggingface.co/JuanJCalderonG/efficient-net-quant/resolve/main/preprocessor_config.json";

  let session;
  try {
    session = await ort.InferenceSession.create(onnxURL, {
      executionProviders: ["wasm"], // o "webgl" si quieres usar GPU WebGL
    });
    status.textContent = "model loaded";
  } catch (e) {
    console.error("Failed to load ONNX model:", e);
    status.textContent = "Failed to load model";
    return;
  }

  // 4) Descargar el JSON de preprocesamiento
  let preprocConfig;
  try {
    const resp = await fetch(preprocURL);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    preprocConfig = await resp.json();
  } catch (e) {
    console.error("Failed to load preprocessor_config.json:", e);
    status.textContent = "Failed to load preprocessor config";
    return;
  }

  // Extraer parámetros de preprocesamiento
  const { image_size, mean, std, do_center_crop, crop_size } = preprocConfig;

  status.textContent = "Ready";

  // 5) Función de preprocesamiento (canvas → ort.Tensor)
  function preprocessImage(imgElement) {
    const targetSize = crop_size || image_size;

    // Canvas temporal para redimensionar a image_size × image_size
    const tmpCanvas = document.createElement("canvas");
    const tmpCtx = tmpCanvas.getContext("2d");
    tmpCanvas.width = image_size;
    tmpCanvas.height = image_size;
    tmpCtx.drawImage(imgElement, 0, 0, image_size, image_size);

    // Canvas final para recorte (si aplica) o usar directamente
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    if (do_center_crop && crop_size < image_size) {
      const start = Math.floor((image_size - crop_size) / 2);
      canvas.width = crop_size;
      canvas.height = crop_size;
      ctx.drawImage(
        tmpCanvas,
        start,
        start,
        crop_size,
        crop_size,
        0,
        0,
        crop_size,
        crop_size
      );
    } else {
      canvas.width = image_size;
      canvas.height = image_size;
      ctx.drawImage(tmpCanvas, 0, 0);
    }

    // Obtener ImageData (RGBA) del canvas
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const { data, width, height } = imageData; // data es Uint8ClampedArray

    // Convertir a Float32Array normalizado en formato CHW
    const H = height;
    const W = width;
    const C = 3;
    const chwBuffer = new Float32Array(1 * C * H * W);

    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const idx = (y * W + x) * 4;
        const r = (data[idx] / 255 - mean[0]) / std[0];
        const g = (data[idx + 1] / 255 - mean[1]) / std[1];
        const b = (data[idx + 2] / 255 - mean[2]) / std[2];
        const pos = y * W + x;
        chwBuffer[0 * H * W + pos] = r; // canal R
        chwBuffer[1 * H * W + pos] = g; // canal G
        chwBuffer[2 * H * W + pos] = b; // canal B
      }
    }

    return new ort.Tensor("float32", chwBuffer, [1, C, H, W]);
  }

  // 6) Función de inferencia
  async function runInference(img) {
    const inputTensor = preprocessImage(img);
    const inputName = session.inputNames[0];
    const feeds = { [inputName]: inputTensor };
    const results = await session.run(feeds);
    const outputName = session.outputNames[0];
    return results[outputName]; // ort.Tensor shape [1, num_classes]
  }

  // 7) Manejo del evento de carga de archivo
  fileUpload.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e2) => {
      uploadButton.style.display = "none";

      const oldImg = document.getElementById("inputImage");
      if (oldImg) oldImg.remove();

      const image = document.createElement("img");
      image.src = e2.target.result;
      image.style.maxWidth = "100%";
      image.style.height = "100%";
      image.id = "inputImage";
      imageContainer.appendChild(image);

      image.onload = async () => {
        status.textContent = "Analysing…";
        try {
          const resultTensor = await runInference(image);
          const { classIndex, score } = postprocess(resultTensor);
          const classLabel = labels[classIndex] || `Class ${classIndex}`;

          // Mostrar etiqueta y confidence en líneas separadas:
          status.innerHTML = `
            <span class="prediction-text">Predicted: ${classLabel}</span><br/>
            <span class="confidence-text">Confidence: ${(score * 100).toFixed(1)}%</span>
          `;
        } catch (err) {
          console.error("Inference error:", err);
          status.textContent = "Inference failed";
        }
      };
    };
    reader.readAsDataURL(file);
  });

  // 8) Botón de reset
  resetButton.addEventListener("click", () => {
    const img = document.getElementById("inputImage");
    if (img) img.remove();
    fileUpload.value = "";
    status.textContent = "Ready";
    uploadButton.style.display = "flex";
  });

  // 9) Clic en el contenedor abre diálogo de selección
  container.addEventListener("click", () => {
    fileUpload.click();
  });
});
