# Deep Fluids Pytorch Implementation

Pytorch implementation of [Deep Fluids: A Generative Network for Parameterized Fluid Simulations](http://www.byungsoo.me/project/deep-fluids).

![teaser](./asset/teaser.png)

## Requirements

* Pytorch
* [Mantaflow](http://mantaflow.com/)

To install `mantaflow`, run:

    $ git clone https://bitbucket.org/mantaflow/manta.git
    $ git checkout 15eaf4
    
and follow the [instruction](http://mantaflow.com/install.html). Note that `numpy` cmake option should be set to enable support for numpy arrays. (i.e., `-DNUMPY='ON'`)

## Usage

To generate dataset:

```bash
manta.exe ./scene/smoke_pos_size.py
manta.exe ./scene/liquid_pos_size.py
```

To train:

```bash    
python main.py --name=smoke
python main.py --name=liquid --no-curl
```

To test:

```bash
python main.py --test --name=smoke
python main.py --test --name=liquid --no-curl
```