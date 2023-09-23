# `cupy_som`

Self-organizing maps using [Cupy](https://cupy.dev/)

## Installation

### Install cuda (on Ubuntu)

```sh
sudo apt install nvidia-cuda-toolkit # 7 GiB!
# verify
poetry install -E cuda11x
python -c "import cupy as cp; print(cp.cuda.get_cuda_path()); x_gpu = cp.array([1, 2, 3])"
```

## Install the package

```sh
poetry install -E cuda11x # with cuda installed
poetry install -E cuda11x --with jupyter # when you want jupyter
poetry install -E cuda11x --with jupyter,doc # when you want to build the documentation
```

or add it to another project as a dependency (assuming you have your github configured
adequately)

```sh
poetry add git++ssh://github.com:StefanUlbrich/cupy_som.git
```

## Build documentation

```sh
poetry install --with doc
cd docs
sphinx-apidoc ../src/cupy_som -o api # if the code changed
make html
```

## Monitor GPU usage

```sh
watch -n 1 nvidia-smi
```
