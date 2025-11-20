# Base miniconda image
FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Copy environment file
COPY environment.yml /app/environment.yml

# Create conda environment
RUN conda env create -f environment.yml

# Activate environment by default
SHELL ["bash", "-c"]
RUN echo "conda activate rnalig" >> ~/.bashrc
ENV PATH /opt/conda/envs/rnalig/bin:$PATH

# Copy all project files into container
COPY . /app

# Render sets $PORT automatically — we must respect it
ENV PORT 10000

# Expose port (optional)
EXPOSE 10000

# Start Streamlit inside the conda environment
CMD ["bash", "-c", "conda activate rnalig && streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"]
