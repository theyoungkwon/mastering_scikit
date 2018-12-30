## COMP 5212: Machine Learing Project 1

#### Description
This is the README.md file which explains the way to run 'code.py' code.
'code.py' code assume that it is located in a folder named "code" and all the datasets is in the folder named "dataset".


If you run program, it will train and give you results of four different classifiers on five datasets.
(Classifiers : Logistic Regression, Linear Support Vector Machines, Radial Base Function Support Vector Machine, Neural Networks
Datasets : Breast-cancer, diabetes, digits, iris, wine)

Results are displayed on the screen.
(Accuracy, Loss, AUC, Training Time, Precision, Recall, F1-score)

Plots are saved on the the current directory in a '.png' format.

#### How to run the code
- go to a folder where "code.py" is located
- $python code.py

- In order to run the program on each dataset
- $python [0,1,2,3,4]

- in order to run Logistic Regression over time
- $python [0,1,2,3,4,] 1