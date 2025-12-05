// models/Prediction.js
const { Schema, model } = require('mongoose');

const PredictionSchema = Schema({
    // La fecha y hora de la predicción
    timestamp: {
        type: Date,
        default: Date.now
    },
    // El valor predicho (ej: 3.53)
    prediction: {
        type: Number,
        required: true
    },
    // Los datos de entrada (para saber por qué predijo eso)
    features: {
        type: [Number], // Es un array de números
        required: true
    }
});

// Exportamos el modelo para usarlo en el controlador
module.exports = model('Prediction', PredictionSchema);