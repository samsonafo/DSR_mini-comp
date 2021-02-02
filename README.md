# Rossman Mini-Competition Solution

A model solution to the [Rossman Mini-Competition](https://github.com/ADGEfficiency/minicomp-rossman).

This project is built using a [cookie-cutter-data-science](https://github.com/ADGEfficiency/cookie-cutter-data-science) template.


## Setup

Due to the wide variety in options for managing virtual environments, we leave it up to the user to create and activate your virtual environment.

```bash
$ make requirements
$ make dotenv
$ make init
```

We then need to get the data for the competition:

```bash
$ make rossman
```

## Use

This project is organized into sprints - full iterations through the data pyramid.

Currently we have one sprint complete - you can view this sprint in [notebooks/sprint0.ipynb](). You can also run this sprint using (it will train a model from scratch):

```bash
$ python3 src/sprint0.py 
```
