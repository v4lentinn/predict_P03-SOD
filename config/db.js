const mongoose = require('mongoose');

const dbConnection = async () => {
    try {
        // TRUCO: 'host.docker.internal' permite a tu contenedor ver lo que hay en tu Mac
        // Si estuvieras en local puro sería 'localhost', pero en Docker Mac es esto:
        await mongoose.connect(process.env.MONGO_URI);
        console.log('Base de datos online') 
        
    } catch (error) {
        console.error(error);
        throw new Error('Error a la hora de iniciar la base de datos');
    }
}

module.exports = {
    dbConnection
}