# `cupy_som`

Self-organizing maps on CPU/GPU using [Cupy](https://cupy.dev/). The content of this
package is to be considered experimental and maybe educational. Please do not use in production
at your own risk only.

With it one can

* define arbitrary neural topologies–extending the concept of
    [Self-Organizing Maps](https://en.wikipedia.org/wiki/Self-organizing_map)
    (see my [blog post on SOM](https://www.lemonfold.io/posts/2022/aiml_essentials/part1/aiml-essentials-part1/)).
    This is achieved by allowing (or rather, requiring) users to explicitly specifying the neurons
    coordinates in the neural/latent space. To compute distance in the neural space you can choose
    between the traditional Euclidean norm or the cosine distance (e.g., together with the ico-sphere topology).
* Train with sequential (life-long) or fast, parallelized batch learning (stochastic Expectation maximization) with
    (see [Chapter 7](http://docs.unigrafia.fi/publications/kohonen_teuvo/))
* Effortlessly choose between the CPU or GPU thanks to [Cupy](https://cupy.dev/)

This package is based on [this recent blog post](https://www.lemonfold.io/posts/2023/citrate/cerebral/cerebral_part1_motivation/#first-version-of-the-algorithm) about an
implementation of self-organization in Rust (and Python), and years of research into
its [applications in Robotics](https://www.lemonfold.io/publications/)

I used this package in some personal research projects and it does not contain
any external contributions prior to version 0.2.

## Installation

## Poetry and PoeThePoet

[Poetry](https://python-poetry.org/) is a packaging and environment management took.
[PoeThePoet](https://poethepoet.natn.io/index.html) is a task runner

```sh
curl -sSL https://install.python-poetry.org | python3 -
poetry self add 'poethepoet[poetry_plugin]'
```

Available tasks:

```sh
poetry lint # runs ruff to fix all auto-fixable issues
poetry types # runs the mypy type checker
poetry format # runs ruff to format the source files
poetry test # runs test and coverage
poetry all # runs all of the above
poetry install-kernel # makes the environment available in jupyter notebooks
poetry uninstall-kernel # removes the associated jupyter kernel
```

### Install cuda (on Ubuntu)

```sh
sudo apt install nvidia-cuda-toolkit # 7 GiB!
# verify
poetry install -E cuda11x
python -c "import cupy as cp; print(cp.cuda.get_cuda_path()); x_gpu = cp.array([1, 2, 3])"
```

## Install the package

```sh
# optional. when using pyenv, select your version
pyenv shell 3.11 && poetry env use python
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

## License

The code is licensed under the GNU Affero General Public License.
