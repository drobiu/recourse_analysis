# Recourse Analysis

Recourse Analysis is a framework for measuring shifts in domains and models induced by employing counterfactual explanations in the process of algorithmic recourse. 


## How to use

### Installation

1. Clone the project
2. Create a `python3.7` environment
3. Install dependencies using `pip install -r requirements.txt`

### Running experiments

In order to run an experiment you have to specify the model, the dataset and the recourse generators.

The framework is built on top of [CARLA](https://github.com/carla-recourse/CARLA/blob/main/docs/source/index.rst), an algorithmic recourse framework. The models and generators are thus specified using CARLA classes.

#### Specifying recourse generators
Here is an example of an object specifying two generators: CLUE and Wachter:

```python
generators = {
            "CLUE": {
        "class": "Clue",
        "hyperparameters": {
            ...
        }
    },
    "Wachter": {
        "class": "Wachter",
        "hyperparameters": {
            ...
        }
    }
}
```

#### Specifying models
Here is an example of an object specifying a two-layer ANN model:

```python
model = {
        'model_type': 'ann',
        'hyperparameters': {...}
    }
```

#### Executing the experiment
In order to run the experiment, we first need to set up the experiment by creating an experiment object.

```python
experiment = Experiment(
    title='test_experiment',
    generators=generators,
    model=model
)
```
A title, the generators and the models objects need to be supplied to the constructor


Next, a dataset must be loaded. This snippet loads one of the synthetic datasets by specifying its path and its features.
```python
experiment.load_dataset(
    path='../datasets/linearly_separable.csv',
    continuous=['feature1', 'feature2'], categorical=[],
    immutables=[], target='target'
)
```

In order to execute the experiment, we call `run_experiment(...)` and specify the number of iterations (rounds) of recourse as well as the number of CEs per round. In order to save the results, `save_results()` must be called.
```python
experiment.run_experiment(iterations=10, samples=10)
experiment.save_results()
```

Additionally, we can generate animations using `save_gifs(...)`.


## Background

### Algorithmic recourse

Algorithmic recourse is the process of providing actionable alternatives to decisions made by a black-box machine learning model in the form of **counterfactual explanations** (CEs) that bring a positive outcome to the person receiving recourse. Counterfactual explanations can take a form of a sentence similar to "*You have received a negative credit score assessment because your monthly pay is 10,000€. If you had a monthly pay of 15,000€, you would have received a
positive credit score.*" 

<img src=recourse_analysis/notebooks/images/clue_wachter_25cf_caption_1.png style="display: block;
  margin-left: auto;
  margin-right: auto;
  width: 50%;">

As counterfactual explanations are employed, the corresponding data points shift and if later the model is retrained its decision boundary shifts too. After multiple iterations of this, we can expect to see shifts in both the domain and the classifier model. We employ a number of metrics that measure these shifts.

<img src=recourse_analysis/gifs/020005_1_CLUE_0%20latent%20dim%2012%20epochs%203%201_gif.gif style="display: block;
  margin-left: auto;
  margin-right: auto;
  width: 80%;">

### Metrics

T.B.D.