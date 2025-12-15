# Master's thesis - [Roots of creativity]

This is the repository for my master's thesis in Cognitive Affective Neuroscience at the Chair of Psychological Methods and Cognitive Modeling. The goal is to develop a minimal model for internal languag prediction and it's role in social interaction. This involves two language prediction networks (Elman networks) in combination with a resolver network (Parallel Constraint Satisfaction networks).

## Structure
The repository contains:

|directory | contents |
|---|---|
| code/ | Python code for Elman and PCS networks, home of log files |
| data/ | home of training data and analysis results |
| figures/ | home of figures |


## Getting started
Python version: 3.13.0

used packages and requirements:

    - numpy
    - pandas
    - matplotlib
    - seaborn
    - tabulate
    - pyarrow

## Getting results
After installing the required packages, the main script main.py runs and produces all data, log files, and figures I used for my thesis. Caution: The script will take several hours to run completely. Every called function in main.py can be called separately if all preceding functions did run and hence all required data was created.

The script will:

    - create training data
    - traing the Elman networks
    - validate their performance
    - show the functionality of the PCSN
    - run the main simulation
    - explore valid parameters
    - make sensitivity analyses
    - find parameter sets for novices and improv players