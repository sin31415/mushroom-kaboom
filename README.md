# project-2

# Mushroom Classification Project

The mushroom dataset consists of 8,124 instances characterized by 22 categorical features, including cap shape, color, gill characteristics, and odor. The target variable indicates whether a mushroom is edible (e) or poisonous (p).

### Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Loading the Dataset](#loading-the-dataset)
- [Analysis](#analysis)
  - [Predictive Features](#predictive-features)
- [Model Performance](#model-performance)
- [Key Findings](#key-findings)
- [Challenges](#challenges)
- [Installation](#installation)
- [Machine Learning Pipeline](#machine-learning-pipeline)

## Project Overview

This project analyzes a dataset of mushrooms to classify them as edible or poisonous using machine learning algorithms. The dataset includes descriptions of hypothetical samples representing 23 species of gilled mushrooms from the Agaricus and Lepiota families, both of which are found in the North American region.

## Features

The dataset used in this project contains the following columns:
1. `Cap Shape`: bell (b), conical (c), convex (x), flat (f), knobbed (k), sunken (s)
2. `Cap Surface`: fibrous (f), grooves (g), scaly (y), smooth (s)
3. `Cap Color`: brown (n), buff (b), cinnamon (c), gray (g), green (r), pink (p), purple (u), red (e), white (w), yellow (y)
4. `Bruises`: bruises (t), no (f)
5. `Odor`: almond (a), anise (l), creosote (c), fishy (y), foul (f), musty (m), none (n), pungent (p), spicy (s)
6. `Gill Attachment`: attached (a), descending (d), free (f), notched (n)
7. `Gill Spacing`: close (c), crowded (w), distant (d)
8. `Gill Size`: broad (b), narrow (n)
9. `Gill Color`: black (k), brown (n), buff (b), chocolate (h), gray (g), green (r), orange (o), pink (p), purple (u), red (e), white (w), yellow (y)
10. `Stalk Shape`: enlarging (e), tapering (t)
11. `Stalk Root`: bulbous (b), club (c), cup (u), equal (e), rhizomorphs (z), rooted (r), missing (?)
12. `Stalk Surface Above Ring`: fibrous (f), scaly (y), silky (k), smooth (s)
13. `Stalk Surface Below Ring`: fibrous (f), scaly (y), silky (k), smooth (s)
14. `Stalk Color Above Ring`: brown (n), buff (b), cinnamon (c), gray (g), orange (o), pink (p), red (e), white (w), yellow (y)
15. `Stalk Color Below Ring`: brown (n), buff (b), cinnamon (c), gray (g), orange (o), pink (p), red (e), white (w), yellow (y)
16. `Veil Type`: partial (p), universal (u)
17. `Veil Color`: brown (n), orange (o), white (w), yellow (y)
18. `Ring Number`: none (n), one (o), two (t)
19. `Ring Type`: cobwebby (c), evanescent (e), flaring (f), large (l), none (n), pendant (p), sheathing (s), zone (z)
20. `Spore Print Color`: black (k), brown (n), buff (b), chocolate (h), green (r), orange (o), purple (u), white (w), yellow (y)
21. `Population`: abundant (a), clustered (c), numerous (n), scattered (s), several (v), solitary (y)
22. `Habitat`: grasses (g), leaves (l), meadows (m), paths (p), urban (u), waste (w), woods (d)

## Loading the Dataset

To load the dataset, use the following code:

   ```python
   from ucimlrepo import fetch_ucirepo
   ``` 

   # Fetch dataset 
   mushroom = fetch_ucirepo(id=73) 

   # Data (as pandas DataFrames) 
   X = mushroom.data.features 
   y = mushroom.data.targets 
   df = mushroom.data.original

   # Combine features and target into a single DataFrame
   df = pd.concat([X, y], axis=1)

## Analysis
### Predictive Features
1. **Odor**: A strong predictive feature, providing clear signals about edibility. Strong odors are often associated with poisonous mushrooms, making simpler models like Logistic Regression highly effective in this context.

2. **Habitat & Population**: Environmental factors play a crucial role. Features like habitat and population density offer insights into the conditions influencing mushroom growth and can aid in classification.

## Model Performance
We trained and tested various classification algorithms on the mushroom dataset. Below are the training and testing accuracies for each model:

| Model                  | Train Accuracy | Test Accuracy |
|------------------------|----------------|---------------|
| Logistic Regression     | 0.9995         | 0.9975        |
| SVM                     | 1.0            | 1.0           |
| KNN                     | 1.0            | 1.0           |
| Decision Tree           | 1.0            | 1.0           |
| Random Forest           | 1.0            | 1.0           |
| Extra Trees             | 1.0            | 1.0           |
| Gradient Boosting       | 1.0            | 1.0           |
| AdaBoost                | 1.0            | 1.0           |

## Key Findings
1. **High Model Performance**: Most models achieved near-perfect accuracy on both training and test sets, indicating strong predictive power in the dataset's features.

2. **Strong Odor Signals**: Odor remains one of the most significant indicators, allowing models to achieve high accuracy with simpler algorithms.
3. **Environmental Influence**: Habitat and population features provide additional context, enhancing model performance.

4. **Species Similarity**: The close resemblance between some edible and poisonous species poses challenges, emphasizing the importance of multiple features for accurate classification.

## Challenges
1. **Overfitting Risk**: Since many models achieved 100% accuracy, there's a potential risk of overfitting, meaning the model might perform less well on unseen data that differs significantly from the training set.

2. **Feature Interdependencies**: Some features may be correlated, requiring careful preprocessing to avoid multicollinearity.

## Installation

To run this project locally:

1. `Clone the Repository:`
   ```bash
   git clone https://github.com/sin31415/mushroom-kaboom.git
   cd mushroom-kaboom
   
2. `Run in Jupyter Notebook:`

   jupyter notebook mushroom.ipynb
   
4. `Necessary Installs`
    ```python
   import pandas as pd
   import numpy as np
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import OneHotEncoder, StandardScaler
   from sklearn.compose import ColumnTransformer
   from sklearn.pipeline import Pipeline
   from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
   from sklearn.linear_model import LogisticRegression
   from sklearn.svm import SVC
   from sklearn.neighbors import KNeighborsClassifier
   from sklearn.tree import DecisionTreeClassifier, plot_tree
   from sklearn.ensemble import (
      RandomForestClassifier,
      ExtraTreesClassifier,
      GradientBoostingClassifier,
      AdaBoostClassifier,
   )
   import matplotlib.pyplot as plt
   import seaborn as sns
   
## Machine Learning Pipeline
The project utilizes the **plumber** library to create a machine learning pipeline for training models on the dataset. Below is the code used for the pipeline:

```python
from plumber import mlpipe, mlht
import pandas as pd

# Load dataset
df_read = pd.read_csv('mushrooms.csv')

# Fetch dataset from UCI repository
mushroom = fetch_ucirepo(id=73) 

# Prepare features and targets
X = mushroom.data.features 
y = mushroom.data.targets 

# Combine features and target into a single DataFrame
df = pd.concat([X, y], axis=1)

# Create and run the machine learning pipeline
pipe = mlpipe(df=df, target_column='poisonous', test_size=0.2, random_state=1, display_analytics=True)
pipe.run_pipeline(drop_max_na_col_in=True, drop_threshold_in=0.25)
pipe.get_feature_importance()
my_model = pipe.user_model_return()
pipe.get_feature_importance()
pipe.visualize_decision_tree()
