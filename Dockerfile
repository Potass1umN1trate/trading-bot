# Use official Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install dependencies
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Command to run the bot (modify if needed)
CMD ["python", "main.py"]
