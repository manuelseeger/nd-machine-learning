from sklearn.metrics import roc_auc_score, fbeta_score, accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from keras.callbacks import Callback
import pandas as pd
from keras import backend as K
import tensorflow as tf
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt


class ClassifierMetricsCallback(Callback):
    ''' Keras callback to calculate and print metrics including the f-score during training'''
    def __init__(self, beta=1, pos_label=1):
        super().__init__()
        self.pos_label = pos_label
        self.beta = beta

    def on_epoch_end(self, batch, logs={}):
        logs = logs or {}
        predict_results = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]

        roc_auc = roc_auc_score(val_targ, predict_results)
        logs["roc_auc"] = roc_auc

        fbeta = fbeta_score(val_targ, predict_results, beta=self.beta, pos_label=self.pos_label)
        logs["fbeta"] = fbeta

        accuracy = accuracy_score(val_targ, predict_results)
        logs["accuracy"] = accuracy

        print(' - val_f[{}]: {} - val_acc: {}'.format(self.beta, fbeta, accuracy))


class MetricReport:
    ''' Custom report collector. Provides some convenience methods to calculated and collect
    classification statistics, per classifier.
    It also runs a small cost simulation for a fictitious retention campaign based on the
    classifiers' predictions.'''

    # cost of the retention campaign (if successful)
    retention_campaign = 50
    # cost of re-acquiring a lost customer
    acquisition_campaign = 500
    # cost of reaching out to a potential churner
    cost_of_campaign = 2
    # assumed success rates of retention campaign
    success_rate = [0.1, 0.3, 0.5]

    def __init__(self, beta=2):
        self.report = self.make_df()
        self.beta = beta

    def make_df(self):
        columns = ['Classifier', 'Fbeta', 'Accuracy', 'TN', 'FP', 'FN', 'TP', 'Beta',
                   'Precision', 'Recall', 'F1', 'Support']
        for s in self.success_rate:
            columns.append('Cost[{}]'.format(s))

        return pd.DataFrame(columns=columns)

    def predict_add(self, clf, X_test, y_true):
        result = self.predict_report(clf, X_test, y_true)

        self.report = self.report.append(result, ignore_index=True)

    def predict_report(self, clf, X_test, y_true):

        y_pred = clf.predict(X_test)
        precision, recall, f1, support = precision_recall_fscore_support(y_pred, y_true, average='binary')
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        result = {
            'Classifier': clf.__class__.__name__,
            'Fbeta': fbeta_score(y_true, y_pred, self.beta),
            'Accuracy': accuracy_score(y_true, y_pred),
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'TP': tp,
            'Beta': self.beta,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
        }

        for s in self.success_rate:
            result['Cost[{}]'.format(s)] = self.calculate_cost(tn, fp, fn, tp, s)
        return result

    def add_from_dict(self, clf, d):
        result = {
            **d,
            'Classifier': clf.__class__.__name__,
        }
        for s in self.success_rate:
            result['Cost[{}]'.format(s)] = self.calculate_cost(result['TN'], result['FP'], result['FN'], result['TP'], s)
        self.report = self.report.append(result, ignore_index=True)
        return result

    def predict_single(self, clf, X_test, y_true):
        df = self.make_df()
        result = self.predict_report(clf, X_test, y_true)
        return df.append(result, ignore_index=True)

    def calculate_cost(self, tn, fp, fn, tp, success_rate):
        # Total cost of reaching out to predicted churners
        cost = (fp + tp) * self.cost_of_campaign
        # Cost of offering retention benefit to loyal customers
        cost += fp * self.retention_campaign
        # Cost of successfully retained churners
        cost += tp * success_rate * self.retention_campaign
        # Cost of customers lost who the campaign couldn't convince to stay
        cost += tp * (1 - success_rate) * self.acquisition_campaign
        # Cost of customers leaving who we didn't target at all
        cost += fn * self.acquisition_campaign
        return cost

    def reset(self, beta=None):
        self.report = self.make_df()
        if beta is not None:
            self.beta = beta

def fbeta_loss(y_true, y_pred, beta=2):
    ''' Custom loss function directly based on F-score to be used as a loss in Keras.
    Ultimately not used as it made the classifier label everyone either as a churner
    or as loyal.'''
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = (1 + beta ** 2) * p * r / ((beta ** 2) * p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

# It is good to randomize the data before drawing Learning Curves
def randomize(X, Y):
    permutation = np.random.permutation(Y.shape[0])
    X2 = X[permutation,:]
    Y2 = Y[permutation]
    return X2, Y2


def draw_learning_curves(estimator, X, y, n_jobs, random_state, scoring):
    X, y = randomize(X,y)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=n_jobs,
        train_sizes=np.linspace(.2, 1.0, 20),
        scoring=scoring, verbose=0, random_state=random_state)

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    fig = plt.figure(figsize=(10, 8))

    plt.title("Learning Curves")
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    plt.plot(train_scores_mean, 'o-', color="g", label="Training score")
    plt.plot(test_scores_mean, 'o-', color="y", label="Cross-validation score")
    plt.legend(loc="best")
    plt.show()

def drop_from(categorical_columns, binary_columns, continuous_columns, outlier_candidates, drop):
    '''' Drop a column name from the groups of columns. This can be used when columns
    are to be dropped as a result of analysis further down the road.'''
    categorical_columns = list(set(categorical_columns) - set(drop))
    binary_columns = list(set(binary_columns) - set(drop))
    continuous_columns = list(set(continuous_columns) - set(drop))
    outlier_candidates = list(set(outlier_candidates) - set(drop))
    return categorical_columns, binary_columns, continuous_columns, outlier_candidates
