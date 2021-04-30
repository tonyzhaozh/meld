# MELD: Meta-RL with Latent Dynamics
Paper: https://arxiv.org/abs/2010.13957

Project Website: https://sites.google.com/view/meld-lsm

by Tony Z. Zhao*, Anusha Nagabandi*, Kate Rakelly*, Chelsea Finn, and Sergey Levine (UC Berkeley, Stanford)

> Meta-reinforcement learning algorithms can enable autonomous agents, such as robots, to quickly acquire new behaviors by leveraging a set of related training tasks to learn a new skill from a small amount of experience.
However, meta-RL has proven challenging to apply to robots in the real world, largely due to onerous data requirements during meta-training compounded with the challenge of learning from high-dimensional sensory inputs such as images.
Latent state models, which learn compact state representations from a sequence of observations, can accelerate representation learning from visual inputs.
In this paper, we leverage the perspective of meta-learning as task inference to show that latent state models can also perform meta-learning given an appropriately defined observation space.
Building on this insight, we develop meta-RL with latent dynamics (MELD), an algorithm for meta-RL from images that performs inference in a latent state model to quickly acquire new skills given observations and rewards.
MELD outperforms prior meta-RL methods, and enables a real WidowX robotic arm to insert an ethernet cable into new locations given sparse task completion rewards after only 4 hours of meta-training in the real world.
To our knowledge, MELD is the first meta-RL algorithm trained in a real-world robotic control setting with image observations.

## Getting started ##
### Prerequisites
- Linux or macOS
- CPU or NVIDIA GPU + CUDA (10) + CuDNN (7.6)

### Installation

#### Anaconda

- Install mujoco, put files and key into ~/.mujoco.
Make sure that `LD_LIBRARY_PATH` is set like this:
```bash
export LD_LIBRARY_PATH=~/.mujoco/mujoco200/bin
```
- Install cuda 10 and cudnn 7.6, and set `LD_LIBRARY_PATH` to point to cuda and nvidia drivers. Example:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/lib:/usr/lib64:/usr/lib/nvidia-418
```

- Install all requirements for this repo into a conda env
```bash
cd <path_to_meld>/docker
conda env create -f environment.yml
source activate meld
```

- Clone this repo and then clone the agents repo which is a modified version of tf-agents
```
cd <path_to_meld>
git clone https://github.com/tonyzhaozh/agents.git
cd agents
git fetch origin
git checkout r0.3.0
pip install -e .
```
- Make sure path is set correctly for each of newly opened shell:
```bash
export PYTHONPATH=<path_to_meld>
export LD_PRELOAD=''
```

#### Docker
- Follow the directions above to clone this repo and the `agents `repo but no need to pip install it. Move the `agents` repo into `<path_to_meld>/docker`
- Place your `mjkey.txt` into `<path_to_meld>/docker`
- From `<path_to_meld>/docker` directory, run this command to build the Docker image from the Dockerfile:
```docker build . -t meld```
- Run this command to start the Docker container from the image and run bash:
```docker run --rm --runtime=nvidia -it -v <path_to_meld>:/root/meld meld /bin/bash```
From here you can run any experiment using the commands listed below.


### Examples usage

Notice: even though all training will be done in a single GPU, Mujoco requires two available GPUs for rendering the train and eval env. It will segfault if only one GPU is installed in the machine.

```bash

CUDA_VISIBLE_DEVICES=0 python meld/scripts/run_meld.py \
  --root_dir save_dir \
  --experiment_name dense_cheetah \
  --gin_file meld/configs/0_meld_dense.gin \
  --gin_file meld/configs/1_cheetahVel.gin \
  --gin_file meld/configs/2_cheetahVel_1episode.gin
 
```
For commands to reproduce experiments in the paper, please refer to examples.txt

To view training and evaluation information (e.g. learning curves, GIFs of rollouts and predictions), run
```bash
tensorboard --logdir save_dir
```
and open http://localhost:6006 in your browser.

--------------------------------------
#### Communication

If you spot a bug or have a problem running the code, please open an issue.

Please direct other correspondence to Tony Zhao: tonyzhao0824@berkeley.edu
