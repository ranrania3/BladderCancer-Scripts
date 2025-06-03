const axios = require("axios");
require("dotenv").config();

const FLASK_URL = process.env.FLASK_URL;
const FLASK_SECRET_TOKEN = process.env.FLASK_SECRET_TOKEN;

async function getPredictionFromFlask(inputData) {
  try {
    const response = await axios.post(FLASK_URL, inputData, {
      headers: {
        Authorization: `Bearer ${FLASK_SECRET_TOKEN}`,
        "Content-Type": "application/json",
      },
    });

    return response.data;
  } catch (error) {
    console.error("Error:", error.response?.data || error.message);
    throw new Error("Error getting prediction from Flask");
  }
}

module.exports = { getPredictionFromFlask };
