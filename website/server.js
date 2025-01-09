const express = require('express');
const bodyParser = require('body-parser');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(bodyParser.json());
app.use(express.static('public'));

let model = {};

// Load the model from model.json
fs.readFile(path.join(__dirname, 'model.json'), (err, data) => {
    if (err) {
        console.error("Error loading model:", err);
        throw err;
    }
    const rawModel = JSON.parse(data);
    model = {};

    // Parse keys like "(np.int64(0), np.int64(1))"
    for (let key in rawModel) {
        const match = key.match(/np\.int64\((\d+)\), np\.int64\((\d+)\)/);
        if (!match) {
            console.warn(`Skipping invalid key: ${key}`);
            continue;
        }
        const [_, class1, class2] = match.map(Number);

        const classifier = rawModel[key];
        if (!Array.isArray(classifier.weights) || classifier.weights.length !== 20) {
            console.warn(`Skipping ${key}: Invalid weights`);
            continue;
        }
        if (typeof classifier.bias !== 'number') {
            console.warn(`Skipping ${key}: Invalid bias`);
            continue;
        }

        model[`${class1}-${class2}`] = classifier;
    }

    console.log("Model loaded successfully with classifiers:", Object.keys(model));
});

// Standardization settings
const means = [1286.18, 0.49, 1.79, 0.51, 5.89, 0.52, 37.74, 0.63, 140.25, 4.31, 10.79, 645.64, 1251.51, 2124.21, 12.32, 5.49, 9.99, 0.74, 0.71, 0.51];
const stds = [501.97, 0.50, 0.80, 0.50, 4.78, 0.50, 18.55, 0.28, 41.08, 2.11, 6.55, 442.44, 432.19, 1084.75, 4.51, 3.99, 5.02, 0.44, 0.45, 0.50];

function standardizeFeatures(features) {
    return features.map((feature, index) => (feature - means[index]) / stds[index]);
}

app.post('/predict', (req, res) => {
    try {
        let features = req.body.features;
        if (!Array.isArray(features) || features.length !== means.length) {
            return res.status(400).json({ error: "Invalid input features." });
        }

        console.log('Raw input features:', features);
        features = standardizeFeatures(features);
        console.log('Standardized features:', features);

        let votes = [0, 0, 0, 0];
        let decisions = {};

        for (let key in model) {
            const [class1, class2] = key.split('-').map(Number);
            const classifier = model[key];

            let decision = classifier.bias;
            for (let i = 0; i < features.length; i++) {
                decision += features[i] * classifier.weights[i];
            }

            decisions[key] = decision;

            if (decision > 0) {
                votes[class1]++;
            } else {
                votes[class2]++;
            }
        }

        const prediction = votes.indexOf(Math.max(...votes));
        console.log('Prediction details:', { votes, decisions, finalPrediction: prediction });

        res.json({
            prediction: prediction,
            priceRange: getPriceRangeDescription(prediction),
            confidence: calculateConfidence(votes),
            details: { votes, decisions }
        });
    } catch (error) {
        console.error('Prediction error:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

function getPriceRangeDescription(prediction) {
    const ranges = {
        0: "Low Cost (Budget Friendly)",
        1: "Medium Cost (Mid-Range)",
        2: "High Cost (Premium)",
        3: "Very High Cost (Flagship)"
    };
    return ranges[prediction];
}

function calculateConfidence(votes) {
    const total = votes.reduce((a, b) => a + b, 0);
    const maxVotes = Math.max(...votes);
    return ((maxVotes / total) * 100).toFixed(2);
}

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
