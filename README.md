---
title: Student Performance API
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: docker
app_file: main.py
pinned: false
---
# Student Performance API

FastAPI backend for predicting student exam scores and recommendations.

## API Endpoints

- **POST** `/analyze` - Predict score and get recommendations
- **GET** `/docs` - Interactive API documentation
- **GET** `/` - Health check

## Usage

```bash
curl -X POST "https://[YOUR-USERNAME]-student-performance-api.hf.space/analyze" \
  -H "Content-Type: application/json" \
  -d '{...}'
```

## Model Details

- Trained on student performance dataset
- 19 input features (numeric + categorical)
- Predicts exam scores 0-100