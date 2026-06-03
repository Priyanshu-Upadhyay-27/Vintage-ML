# scratch_lab

A personal lab for implementing Machine Learning algorithms from scratch using Python and OOP. Built as preparation for developing a full ML module.

---

## Purpose

- Understand the math and internals of each algorithm deeply
- Practice OOP design patterns — inheritance, mixins, abstraction, encapsulation
- Build a clean base that can be refactored into a full module later

---

## Structure

```
scratch_lab/
│
├── base.py                      ← BaseEstimator, RegressorMixin, ClassifierMixin
│
├── linear_model/
│   ├── linear_regression.py     ← MSE loss, gradient descent
│   └── logistic_regression.py   ← cross entropy, sigmoid
│
├── tree/
│   └── decision_tree.py         ← Gini impurity, recursive splitting
│
├── ensemble/
│   └── random_forest.py         ← bagging, aggregation
│
├── neighbors/
│   └── knn.py                   ← euclidean distance, voting
│
├── naive_bayes/
│   └── gaussian_nb.py           ← Bayes theorem, conditional probability
│
└── tests/
    ├── test_linear_regression.py
    ├── test_logistic.py
    └── test_decision_tree.py
```

---

## OOP Design

Every algorithm inherits from `BaseEstimator` in `base.py`.

```
BaseEstimator (abstract)
│   fit()           ← abstract, must implement
│   predict()       ← abstract, must implement
│   get_params()    ← free
│   set_params()    ← free
│   save()          ← free
│   load()          ← free
│   __repr__()      ← free
│
├── RegressorMixin
│   score()         ← R² score
│   evaluate()      ← R², MSE, RMSE, MAE
│
└── ClassifierMixin
    score()         ← accuracy
    evaluate()      ← accuracy, precision, recall, F1
```

