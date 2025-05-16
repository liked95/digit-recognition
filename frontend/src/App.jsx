import { useState } from 'react'
import { 
  Container, 
  Box, 
  Typography, 
  Button, 
  Paper,
  CircularProgress
} from '@mui/material'
import axios from 'axios'

function App() {
  const [selectedImage, setSelectedImage] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleImageUpload = (event) => {
    const file = event.target.files[0]
    if (file) {
      setSelectedImage(file)
      setPreviewUrl(URL.createObjectURL(file))
      setPrediction(null)
      setError(null)
    }
  }

  const handlePredict = async () => {
    if (!selectedImage) return

    setLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append('image', selectedImage)

    try {
      const response = await axios.post('http://localhost:5000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })
      setPrediction(response.data)
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred while predicting')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Container maxWidth="sm">
      <Box sx={{ my: 4, textAlign: 'center' }}>
        <Typography variant="h4" component="h1" gutterBottom color="primary">
          Handwritten Digit Recognition
        </Typography>
        <Typography variant="subtitle1" color="text.secondary" gutterBottom>
          Upload an image of a handwritten digit (0 or 1)
        </Typography>

        <Paper 
          elevation={3} 
          sx={{ 
            p: 3, 
            mt: 3, 
            display: 'flex', 
            flexDirection: 'column', 
            alignItems: 'center',
            gap: 2
          }}
        >
          <input
            accept="image/*"
            style={{ display: 'none' }}
            id="image-upload"
            type="file"
            onChange={handleImageUpload}
          />
          <label htmlFor="image-upload">
            <Button variant="contained" component="span">
              Upload Image
            </Button>
          </label>

          {previewUrl && (
            <Box sx={{ mt: 2 }}>
              <img 
                src={previewUrl} 
                alt="Preview" 
                style={{ 
                  maxWidth: '100%', 
                  maxHeight: '200px',
                  border: '1px solid #ccc'
                }} 
              />
            </Box>
          )}

          {selectedImage && (
            <Button 
              variant="contained" 
              color="primary" 
              onClick={handlePredict}
              disabled={loading}
            >
              {loading ? <CircularProgress size={24} /> : 'Predict'}
            </Button>
          )}

          {error && (
            <Typography color="error" sx={{ mt: 2 }}>
              {error}
            </Typography>
          )}

          {prediction && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="h6">
                Prediction: {prediction.prediction}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Confidence: {(prediction.confidence * 100).toFixed(2)}%
              </Typography>
            </Box>
          )}
        </Paper>
      </Box>
    </Container>
  )
}

export default App
