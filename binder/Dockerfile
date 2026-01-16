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


# 3. Set the Working Directory 
WORKDIR /home/jovyan/work

# 4. Copy the rest of your project (Notebooks, Images, URDFs)
COPY --chown=${NB_UID}:${NB_GID} . /home/jovyan/work
