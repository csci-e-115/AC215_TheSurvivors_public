# Use images within path ./data

# Sample query with curl
curl -X 'POST' \
  "http://localhost:8181/gradcam_post/"  \
  -H "accept: application/json" \
  -F "image=@./data/ISIC_0026421.jpg"  \
  -F "age=25" \
  -F "location=head/neck" \
  -F "sex=male"

  # Sample query with FastAPI Swagger UI
  # Note Use GET method
  
  # http://localhost:8181/docs