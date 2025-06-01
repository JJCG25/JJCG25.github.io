import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";
env.allowLocalModels = false;

document.addEventListener("DOMContentLoaded", async () => {
  const imageContainer = document.querySelector(".custom-file-upload");
  const status = document.getElementById("status");
  const fileUpload = document.getElementById("upload");
  const resetButton = document.getElementById("reset-image");
  const uploadButton = document.getElementById("upload-btn");
  const container = document.querySelector(".container");

  // Mostrar que el modelo se está cargando
  status.textContent = "Loading model…";

  // ── URLs EXACTAS en Hugging Face ──

  // ── LLAMADA CORRECTA: pasar UN SOLO OBJETO con 'model' y 'preprocessor' ──
  // Nota: clave "preprocessor" (no "feature_extractor")
    const classifier = await pipeline("image-classification", {
        model: "https://huggingface.co/.../model_quantized.onnx",
        preprocessor: "https://huggingface.co/.../preprocessor_config.json"
    });


  status.textContent = "Ready";

  // Función que hace inferencia sobre la imagen cargada
  async function detect(img) {
    status.textContent = "Analysing…";
    const output = await classifier(img.src);
    status.textContent = "";
    console.log("output", output);
    status.textContent = output[0].label;
  }

  // Evento: usuario selecciona un archivo
  fileUpload.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e2) => {
      // Ocultar el botón de subir para mostrar solo la imagen
      uploadButton.style.display = "none";

      // Si ya existía una imagen previa, la removemos
      const oldImg = document.getElementById("inputImage");
      if (oldImg) oldImg.remove();

      // Creamos el <img> con la imagen seleccionada
      const image = document.createElement("img");
      image.src = e2.target.result;
      image.style.maxWidth = "100%";
      image.style.height = "100%";
      image.id = "inputImage";
      imageContainer.appendChild(image);

      // Ejecutamos la inferencia
      detect(image);
    };
    reader.readAsDataURL(file);
  });

  // Evento: resetear la imagen y el estado
  resetButton.addEventListener("click", () => {
    const img = document.getElementById("inputImage");
    if (img) img.remove();
    fileUpload.value = "";
    status.textContent = "Ready";
    uploadButton.style.display = "flex";
  });

  // Si se hace clic en el contenedor, abrimos el diálogo de selección
  container.addEventListener("click", () => {
    fileUpload.click();
  });
});
