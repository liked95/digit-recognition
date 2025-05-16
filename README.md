# Digit Recognition Web Application

This is a web application that uses a machine learning model to recognize handwritten digits (0 or 1) from images. The application provides a REST API endpoint that accepts image uploads and returns predictions with confidence scores.

## Features

- REST API endpoint for digit recognition
- Supports image upload and processing
- Returns prediction (0 or 1) with confidence score
- Cross-Origin Resource Sharing (CORS) enabled
- Built with Flask and TensorFlow
- Modern React frontend with Material-UI

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- Node.js 16 or higher
- npm (Node package manager)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd digit-recognition
```

2. Set up the backend:
```bash
# Create and activate a virtual environment (recommended)
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

3. Set up the frontend:
```bash
cd frontend
npm install
```

## Running the Application

1. Start the backend server:
```bash
# From the root directory
python app.py
```
The backend server will start running on `http://localhost:5000`

2. Start the frontend development server:
```bash
# From the frontend directory
cd frontend
npm run dev
```
The frontend will start running on `http://localhost:5173`

## API Usage

### Predict Endpoint

**URL**: `/predict`
**Method**: `POST`
**Content-Type**: `multipart/form-data`

**Request Body**:
- `image`: Image file containing a handwritten digit (0 or 1)

**Response**:
```json
{
    "prediction": 1,  // 0 or 1
    "confidence": 0.95  // Confidence score between 0 and 1
}
```

**Example using curl**:
```bash
curl -X POST -F "image=@path/to/your/image.jpg" http://localhost:5000/predict
```

## Error Handling

The API returns appropriate error messages in the following cases:
- No image provided in the request
- Invalid image format
- Processing errors

## Dependencies

### Backend
- Flask 2.0.1
- Werkzeug 2.0.3
- Flask-CORS 3.0.10
- NumPy >= 1.23
- Pillow >= 9.0.0
- TensorFlow >= 2.12.0

### Frontend
- React 19.1.0
- Material-UI 7.1.0
- Axios 1.9.0
- Vite 6.3.5

## License

[Add your license information here] 