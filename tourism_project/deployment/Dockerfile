# Use a minimal base image with Python 3.12 installed
FROM python:3.12-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Copy all files from the current directory on the host to the container's /app directory
COPY . .

# Install Python dependencies listed in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Add a non-root user for security
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory for the user
WORKDIR $HOME/app

# Copy project files to user home directory with ownership
COPY --chown=user . $HOME/app

# Command to run the Streamlit app
CMD ["streamlit", "run", "serve_model.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
