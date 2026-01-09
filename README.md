# Pokémon Data Analysis – Exploratory Data Science Project

## Overview

This project explores a comprehensive Pokémon dataset from **TidyTuesday (April 1, 2025)**. Here, I investigate relationships between physical attributes, battle statistics, and typing combinations. The analysis focuses on identifying correlations, testing intuitive hypotheses, and highlighting statistical outliers using visualization and regression techniques.

The goal of this project is to practice applying simple visualization, regression and learning to draw conclusions about the data. I enjoy video games as a personal hobby, hence doing a project on something I enjoy was good practice.

---

## Questions Explored

1. Is there a relationship between Pokémon height and weight?
2. Does physical attack correlate with special attack?
3. Do heavier Pokémon tend to have higher defense?
4. Are heavier Pokémon generally slower?
5. What are the most common Pokémon types and dual-type combinations?
6. Which Pokémon stand out as statistical outliers in specific attributes?

---

## Data

* **Source:** TidyTuesday (2025-04-01)
* **Observations:** Pokémon across multiple generations
* **Key Features:**

  * Physical attributes: height, weight
  * Battle stats: attack, defense, special attack, special defense, speed
  * Typing information: primary and secondary types

---

## Methods & Tools

* **Python**
* **Pandas & NumPy** for data manipulation
* **Plotly Express** for interactive visualization
* **Matplotlib** for supplemental plotting
* **Linear & Polynomial Regression** using NumPy and scikit-learn

---

## Key Findings

### Height vs Weight

* Height and weight exhibit a **weak linear relationship**.
* Most Pokémon cluster at lower heights and weights, with several extreme outliers.
* Height alone is not a reliable predictor of Pokémon weight.

---

### Attack vs Special Attack

* Physical attack and special attack show **little linear correlation**.
* This supports the idea that Pokémon are designed to specialize in either physical or special combat roles.

---

### Weight vs Defense

* A **moderate positive relationship** exists between weight and defense.
* Heavier Pokémon tend to have higher defense values, though variability remains high.
* This trend likely reflects design choices favoring bulkier Pokémon as defensive units.

---

### Speed vs Weight

* Linear and quadratic regression models both perform poorly.
* Pokémon speed and weight appear largely **independent**, with no strong functional relationship.

---

### Common Typings and Dual Types

* Water is the most common Pokémon type overall.
* Among dual-typed Water Pokémon, **Ground** is the most frequent secondary type.
* These results reflect common design patterns rather than statistical performance advantages.

---

### Notable Outliers

Several Pokémon stand out due to extreme stat distributions:

* **Mega Gardevoir:** Exceptionally high special attack
* **Shuckle:** Extremely high defense at the cost of speed
* **Ninjask:** One of the fastest Pokémon with low defensive stats

These examples highlight intentional stat specialization within Pokémon design.

---

## Limitations

* Correlation coefficients were not explicitly calculated for every comparison.
* Regression models were used for visualization rather than prediction.
* Results are sensitive to outliers and design-driven data distributions.


