# Resevoir Computing and modelling Echo State Networks

## Installation
### Requirements
- Python **3.9 or later**
- 'git'
- 'pip'
- 'numpy'
- 'jupyterlab'

### Clone repository
Open your terminal and go to the folder were you want to install this code. Once there, you would run the following code:
'''
git clone https://github.com/ivanMC-A/CompPhysProject.git
cd CompPhysProject
'''
The latter line is to move to the code's folder.

### Verify Installation

You can verify that everything is set up correctly by running:

'''
jupyter lab rnnExamples.ipynb
'''

## Project Description
As AI and machine learning become more prevelant in society, finding device architecture to match the software needs becomes increasingly important. Resevoir computing is a way to use optical integration into existing silicon infrastructure to realize these needs. A resevoir computer as in Brunner, _et al_., creates a system similar to an echo state network in the time domain, where the weighting matrix is confined by the components and material characteristics. Here we will create a toy model by designing an echo state network that fits and predicts a function of Mackey Glass type. 

## Project Structure
As given in the example, we will first begin with an attempt to create a main python file that contains our echo state network class in a subdirectory which is run by a top-level demo file. This demo file will run our simulation and create sample graphics that demonstrate the results.

## Project Process
1. Impliment the portion of the file responsible for loading in data files and storing it in an appropriate data structure.
2. Create an echo state network solver based on the Mackey-Glass family of delayed differential equations.
3. Design functions to allow data and prediction capabilities to be shown in a visually appealing manner.
4. If time allows, attempt to adjust parameters based on true material capabilities and archetechtures to model a true system. 

## Delegation
- Pablo will work on graphic implimentation and work on the presentation and background.
- Jason will work on class definitions and ensuring the model and presentation are structured to work towards the goal of modeling optical networks such as in Brunner, _et al_.
- Ivan will work on graphic implimentation and help with the implimentation of auxillary class functions.

## Background
The systems modeled in this problem are simple toy models relevent to networks as reported in 
[1] D. Brunner, _et al_., Journal of Applied Physics 124, (2018).
