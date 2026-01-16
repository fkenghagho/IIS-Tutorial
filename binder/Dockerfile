# 1. Use the official Jupyter 3.10 image
FROM jupyter/base-notebook:python-3.10

USER root

# 2. Install system dependencies (Logic + Graphics)
RUN apt-get update && apt-get install -y \
    swi-prolog \
    libgl1-mesa-dev \
    python3-opengl \
    xvfb \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

USER ${NB_UID}

# 3. Handle Python Requirements
# Copy the requirements file from your computer into the Docker image
COPY --chown=${NB_UID}:${NB_GID} ../requirements.txt /tmp/requirements.txt

# Install the libraries listed in that file
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 4. Set the Working Directory 
WORKDIR /home/jovyan/work

# 5. Copy the rest of your project (Notebooks, Images, URDFs)
COPY --chown=${NB_UID}:${NB_GID} . /home/jovyan/work
