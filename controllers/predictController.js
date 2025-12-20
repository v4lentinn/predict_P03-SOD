const { getModelInfo, predict } = require("../services/tfModelService");
const Prediction = require("../models/Prediction"); 

function health(req, res) {
  res.json({ status: "ok", service: "predict" });
}

function ready(req, res) {
  const info = getModelInfo();
  if (!info.ready) {
    return res.status(503).json({
      ready: false,
      modelVersion: info.modelVersion,
      message: "Model is still loading"
    });
  }
  res.json({ ready: true, modelVersion: info.modelVersion });
}

async function doPredict(req, res) {
  const start = Date.now();

  try {
    const info = getModelInfo();
    if (!info.ready) {
      return res.status(503).json({ error: "Model not ready", ready: false });
    }

    const { features, meta } = req.body;

    // Validaciones
    if (!features) return res.status(400).json({ error: "Missing features" });
    if (!meta || typeof meta !== "object") return res.status(400).json({ error: "Missing meta object" });
    
    const { featureCount } = meta;
    if (featureCount !== info.inputDim) {
      return res.status(400).json({ error: `featureCount must be ${info.inputDim}, received ${featureCount}` });
    }
    if (!Array.isArray(features) || features.length !== info.inputDim) {
      return res.status(400).json({ error: `features must be an array of ${info.inputDim} numbers` });
    }

    // 1. Ejecutar la predicción
    const predictionValue = await predict(features);
    
    const latencyMs = Date.now() - start;
    const timestamp = new Date().toISOString();

   
    const newPrediction = new Prediction({
        timestamp: timestamp,
        prediction: predictionValue,
        features: features
    });

    const savedPrediction = await newPrediction.save();
    console.log(`[DB] Predicción guardada con ID: ${savedPrediction._id}`);

    res.status(201).json({
      predictionId: savedPrediction._id, // El ID que acaba de crear Mongo
      prediction: predictionValue,
      timestamp,
      latencyMs
    });

  } catch (err) {
    console.error("Error en /predict:", err);
    res.status(500).json({ error: "Internal error" });
  }
}

module.exports = { health, ready, doPredict };