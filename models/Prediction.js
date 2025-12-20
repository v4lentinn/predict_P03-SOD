const { Schema, model } = require('mongoose');

const PredictionSchema = Schema({
    // La fecha y hora de la predicción
    timestamp: {
        type: Date,
        default: Date.now
    },
    // El valor predicho
    prediction: {
        type: Number,
        required: true
    },
    // Los datos de entrada
    features: {
        type: [Number],
        required: true
    }
});

module.exports = model('Prediction', PredictionSchema);