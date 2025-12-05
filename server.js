require("dotenv").config();

const express = require("express");
const path = require("path");
const predictRoutes = require("./routes/predictRoutes");
const { initModel } = require("./services/tfModelService");
// ✅ NUEVO: Importar la conexión
const { dbConnection } = require("./config/db.js"); 

const PORT = process.env.PORT || 3002;

const app = express();
app.use(express.json());

// Servir la carpeta del modelo TFJS
const modelDir = path.resolve(__dirname, "model");
app.use("/model", express.static(modelDir));

// Rutas
app.use("/", predictRoutes);

// Arranque del servidor
app.listen(PORT, async () => {
  const serverUrl = `http://localhost:${PORT}`;
  console.log(`[PREDICT] Servicio escuchando en ${serverUrl}`);

  // ✅ NUEVO: Conectar a la base de datos antes o después del modelo
  await dbConnection(); 

  try {
    await initModel(serverUrl);
  } catch (err) {
    console.error("Error al inicializar modelo:", err);
    process.exit(1);
  }
});