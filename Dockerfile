# Use official Python image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy project files into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables (modify as needed)
ENV PYTHONUNBUFFERED=1

# Command to run the bot (modify if needed)
CMD ["python", "main.py"]
