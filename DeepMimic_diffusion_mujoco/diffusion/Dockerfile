# Start from the miniconda base image
FROM continuumio/miniconda3

##### Install requirements
RUN apt update && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
                   apt install -y --allow-unauthenticated \
                                  wget git vim build-essential \
                                  libosmesa6-dev libglew-dev \
                                  glibc-source unzip \
                                  mpich python3-dev python3-pip patchelf

RUN python3 -m pip install pip --upgrade pip

# Set the working directory
ENV HOME=/diffusion-mimic
WORKDIR $HOME

# Copy all files over
COPY . .
ADD mjkey.txt $HOME/mjkey.txt

#### single script creating the docker image  
RUN <<EOF_Diffmimic
pip3 install -U "Cython<3"

cd $HOME
mkdir $HOME/.mujoco
cd $HOME

wget -c https://www.roboti.us/download/mujoco200_linux.zip -O mujoco200.zip
unzip mujoco200.zip -d $HOME/.mujoco/
rm mujoco200.zip
mv $HOME/.mujoco/mujoco200_linux $HOME/.mujoco/mujoco200

mv $HOME/mjkey.txt $HOME/.mujoco/mjkey.txt

LD_LIBRARY_PATH=/DiffRL/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}

cd $HOME

conda env create -f environment.yml

EOF_Diffmimic
### end of script

ENV LD_LIBRARY_PATH=/DiffRL/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}
ENV MUJOCO_PY_MUJOCO_PATH=/DiffRL/.mujoco/mujoco200

# Activate the conda environment
SHELL ["conda", "run", "-n", "diffmimic", "/bin/bash", "-c"]

