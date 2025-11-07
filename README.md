# Resevoir Computing and modelling Echo State Networks

## Project Description
As AI and machine learning become more prevelant in society finding device archetechture to match the software needs becomes increasingly important. Resevoir computing is a way to use optical integration into existing silicon infrastructure to realize these needs. A resevoir computer as in Brunner et al creates a system similar to an echo state network in the time domain, where the weighting matrix is confined by the components and material charachteristics. Here we will create a toy model by designing an echo state network that fits and predicts a function of Mackey Glass type. 

## Project Structure
As given in the example we will first begin with an attempt to create a main python file that contain our echo state network class in a subdirectory which is run by a top-level demo file. This demo file will run our simulation and create interesting graphics that demonstrate the results.

## Project Process
1. We will impliment the portion of the file responsible for loading in data files and storing it in an appropriate data structure.
2. We then create a echo state network solver based on the MAckey-Glass family of delayed differntial equations equations including (initialization, fitting, and prediction)
3. Finally we will design fucntions allowing the data and prediction capabilities to be shown in a visually appealing manner.
4. If time we may attempt to then attempt to adjust parameters based on true material capabilities and archetechtures to model a true system. (Feel free to remove this if you want *note to self*)

## Delegation
- Pablo 
- Jason will work on class definitions and ensuring the model and presentation are structured to work towards the goal of modeling optical networks such as in Brunner et al.
- Ivan

possible roles graphics implimentation, echo state class implimentation, and presentation and background based design (each of us should take 2 roles so we overlap)

## Background
The systems modeled in this problem are simple toy models relevent to networks as reported in 
[1] D. Brunner et al., Journal of Applied Physics 124, (2018).
