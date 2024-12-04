# Inherit from the Python Docker image
FROM python:3.7-slim
# Copy the requirements file to the container
COPY requirements.txt /app/requirements.txt
# Install dependencies from the requirements file
RUN pip install --no-cache-dir -r /app/requirements.txt
# Copy the source code to app folder
COPY ./app.py /app/
# Change the working directory
WORKDIR /app/
# Set "python" as the entry point
ENTRYPOINT ["python"]
# Set the command as the script name
CMD ["app.py"]