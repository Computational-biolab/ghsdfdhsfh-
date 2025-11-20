# Use a base image that includes Miniconda
FROM continuumio/miniconda3

# Set working directory inside the container
WORKDIR /app

# Copy the conda environment file into the image
COPY environment.yml /app/environment.yml

# Create the conda environment defined in environment.yml
RUN conda env create -f environment.yml

# Use bash as default shell
SHELL ["bash", "-c"]

# Put the rnalig environment on PATH so we don't need `conda activate`
ENV PATH /opt/conda/envs/rnalig/bin:$PATH

# Copy all project files into the image
COPY . /app

# Render sets $PORT automatically; fall back to 10000 if not set
ENV PORT=10000
EXPOSE 10000

# Start the Streamlit app (NO `conda activate` here)
CMD ["bash", "-c", "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"]
