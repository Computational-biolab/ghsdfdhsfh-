# Use a base image that includes Miniconda
FROM continuumio/miniconda3

# Set working directory inside the container
WORKDIR /app

# Copy the conda environment file into a temp location
COPY environment.yml /tmp/environment.yml

# Install all dependencies into the *base* conda environment
# This avoids the need for `conda activate` completely.
RUN conda env update -n base -f /tmp/environment.yml && conda clean -afy

# Copy all project files into the image
COPY . /app

# Render sets $PORT automatically; fall back to 10000 if not set
ENV PORT=10000
EXPOSE 10000

# Start the Streamlit app (no conda activate needed)
CMD ["bash", "-c", "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"]
