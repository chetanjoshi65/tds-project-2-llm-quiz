# Use official Microsoft Playwright image with Python
FROM mcr.microsoft.com/playwright/python:v1.40.0-jammy

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Expose port 8000
EXPOSE 8000

# Start the application
# Render sets PORT environment variable, we'll use it
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
