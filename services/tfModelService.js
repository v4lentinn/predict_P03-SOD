// services/tfModelService.js
const path = require("path");
const { pathToFileURL } = require("url");
const tf = require("@tensorflow/tfjs");
const wasmBackend = require("@tensorflow/tfjs-backend-wasm");

const MODEL_VERSION = process.env.MODEL_VERSION || "v1.0";

// Modificacion por prediction = 0
// --- TUS NÚMEROS MÁGICOS (Copiados de Kaggle) ---
const MEANS = [5.88535, 5.959026, 5.975218, 11.489071, 2.9896383, 6.517541, 15.568954];
const VARIANCES = [349.26392, 353.81195, 356.3087, 47.944473, 3.9793808, 11.060944, 75.887474];


let model = null;
let ready = false;
let inputName = null;
let outputName = null;
let inputDim = null;

function getModelInfo() {
  return {
    ready,
    modelVersion: MODEL_VERSION,
    inputName,
    outputName,
    inputDim
  };
}

function wasmFileDirUrl() {
  const distFsPath = path.join(
    __dirname,
    "..",
    "node_modules",
    "@tensorflow",
    "tfjs-backend-wasm",
    "dist"
  );
  return pathToFileURL(distFsPath + path.sep).href;
}

/**
 * Inicializa backend WASM y carga el GraphModel
 * serverUrl: ej. http://localhost:3002
 */
async function initModel(serverUrl) {
  const wasmPath = wasmFileDirUrl();
  wasmBackend.setWasmPaths(wasmPath);

  await tf.setBackend("wasm");
  await tf.ready();
  console.log("[TF] Backend:", tf.getBackend());

  const modelDir = path.resolve(__dirname, "..", "model");
  console.log("[TF] Sirviendo modelo desde:", modelDir);

  const modelUrl = `${serverUrl}/model/model.json`;
  console.log("[TF] Cargando modelo:", modelUrl);

  model = await tf.loadGraphModel(modelUrl);

  inputName = model.inputs?.[0]?.name || null;
  inputDim = model.inputs?.[0]?.shape?.[1] ?? null;
  outputName = model.outputs?.[0]?.name || null;

  console.log("[TF] inputName:", inputName);
  console.log("[TF] outputName:", outputName);
  console.log("[TF] inputDim:", inputDim);

  if (!inputName || !outputName || !inputDim) {
    throw new Error("No se ha podido detectar inputName/outputName/inputDim");
  }

  // Warm-up
  const Xwarm = tf.zeros([1, inputDim], "float32");
  let out;
  if (typeof model.executeAsync === "function") {
    out = await model.executeAsync({ [inputName]: Xwarm });
  } else {
    out = model.execute({ [inputName]: Xwarm });
  }

  if (Array.isArray(out)) out.forEach(t => t?.dispose?.());
  else out?.dispose?.();
  Xwarm.dispose();

  ready = true;
  console.log("[TF] Modelo listo.");
}



// Prediction = 0
/**
 * Aplica la estandarización (Z-Score) manualmente
 * Fórmula: (x - media) / raiz(varianza)
 */
function standardize(features) {
  return features.map((val, i) => {
    const mean = MEANS[i];
    const std = Math.sqrt(VARIANCES[i]);
    return (val - mean) / std;
  });
}



/**
 * Ejecuta el modelo con un vector de features
 * Devuelve un escalar >= 0
 */
async function predict(features) {
  if (!ready || !model) {
    throw new Error("Model not ready");
  }
  if (!Array.isArray(features) || features.length !== inputDim) {
    throw new Error(`features must be an array of ${inputDim} numbers`);
  }

  // 1. NORMALIZACIÓN MANUAL (prediction = 0)
  // Convertimos los "29.5" en "0.8" para que la IA los entienda
  const normalizedFeatures = standardize(features);


  const X = tf.tensor2d([normalizedFeatures], [1, inputDim], "float32");

  let out;
  if (typeof model.executeAsync === "function") {
    out = await model.executeAsync({ [inputName]: X });
  } else {
    out = model.execute({ [inputName]: X });
  }

  const preds2d = Array.isArray(out)
    ? await out[0].array()
    : await out.array();

  const predictionReal = preds2d?.[0]?.[0] ?? 0;
  const prediction = Math.max(predictionReal, 0); // clamp a 0

  if (Array.isArray(out)) out.forEach(t => t?.dispose?.());
  else out?.dispose?.();
  X.dispose();

  return prediction;
}

module.exports = {
  initModel,
  getModelInfo,
  predict
};
