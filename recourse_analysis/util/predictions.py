from sklearn.metrics import f1_score, accuracy_score
import numpy as np


def predict(model, data):
    pred = model.predict(data.df)
    return np.where(pred > 0.5, 1, 0)


def print_f1_score(model, data):
    score = f1_score(np.array(data.df[data.target]), predict(model, data))
    print(f"F1 score: {score}")


def print_accuracy(model, data):
    score = accuracy_score(np.array(data.df[data.target]), predict(model, data))
    print(f"Accuracy score: {score}")


def print_scores(model_pre, model_post, data):
    print("Before recourse:")
    print_f1_score(model_pre, data)
    print_accuracy(model_pre, data)
    print("\nAfter recourse:")
    print_f1_score(model_post, data)
    print_accuracy(model_post, data)