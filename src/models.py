"""Baselines, candidate Pipelines and hyper-parameter search spaces.

All learned preprocessing (scaling, resampling) lives inside the Pipeline so
that it is re-fit on each training fold only."""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config


class MomentumRule(BaseEstimator, ClassifierMixin):
    """B3: Buy if ret_20 > 0 and Close > SMA200; Sell if ret_20 < 0 and Close < SMA200."""

    def fit(self, X, y):
        self.classes_ = np.array(config.CLASSES)
        return self

    def predict(self, X):
        up = (X["ret_20"] > 0) & (X["close_sma200"] > 0)
        down = (X["ret_20"] < 0) & (X["close_sma200"] < 0)
        return np.where(up, 1, np.where(down, -1, 0))


class AlwaysBuy(BaseEstimator, ClassifierMixin):
    """B2: always predict Buy (equivalent to staying invested)."""

    def fit(self, X, y):
        self.classes_ = np.array(config.CLASSES)
        return self

    def predict(self, X):
        return np.ones(len(X), dtype=int)


def baselines() -> dict:
    return {
        "B0_majority": DummyClassifier(strategy="most_frequent"),
        "B1_stratified": DummyClassifier(strategy="stratified", random_state=config.SEED),
        "B2_always_buy": AlwaysBuy(),
        "B3_momentum_rule": MomentumRule(),
    }


def make_pipeline(name: str, class_weight="balanced", smote: bool = False):
    """Candidate Pipelines M1-M3. Scaling only for the linear model."""
    if name == "LR":
        steps = [("scale", StandardScaler()),
                 ("clf", LogisticRegression(max_iter=5000, class_weight=class_weight))]
    elif name == "RF":
        steps = [("clf", RandomForestClassifier(n_estimators=300, class_weight=class_weight,
                                                n_jobs=-1, random_state=config.SEED))]
    elif name == "HGB":
        steps = [("clf", HistGradientBoostingClassifier(class_weight=class_weight,
                                                        random_state=config.SEED))]
    else:
        raise ValueError(name)
    if smote:
        steps.insert(len(steps) - 1, ("smote", SMOTE(random_state=config.SEED)))
        return ImbPipeline(steps)
    return Pipeline(steps)


SEARCH_SPACES = {
    "LR": {"clf__C": loguniform(1e-3, 10)},
    "RF": {
        "clf__max_depth": [3, 4, 6, 8, None],
        "clf__min_samples_leaf": randint(20, 200),
        "clf__max_features": ["sqrt", 0.3, 0.5],
    },
    "HGB": {
        "clf__learning_rate": loguniform(0.01, 0.2),
        "clf__max_iter": randint(50, 400),
        "clf__max_depth": [2, 3, 4, None],
        "clf__min_samples_leaf": randint(20, 200),
        "clf__l2_regularization": uniform(0, 1.0),
    },
}


def tune(name: str, X: pd.DataFrame, y: pd.Series, cv, n_iter: int = 25, **pipe_kwargs):
    """RandomizedSearchCV over the whole Pipeline with time-series CV, scored by macro-F1."""
    search = RandomizedSearchCV(
        make_pipeline(name, **pipe_kwargs), SEARCH_SPACES[name], n_iter=n_iter,
        scoring="f1_macro", cv=cv, random_state=config.SEED, n_jobs=-1, refit=True,
        return_train_score=True,
    )
    search.fit(X, y)
    return search
