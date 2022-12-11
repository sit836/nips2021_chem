import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GridSearchCV
import multiprocessing
from functools import partial
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import cvxpy as cp

from config import in_path, out_path


def create_id(data):
    if data.shape[0] == 1:
        return data.str.split('_', expand=True)[1].astype(int)[0]
    else:
        return data.str.split('_', expand=True)[1].astype(int)


def get_xy(dim_u):
    columns = ['system_id', 'target', 'Y'] + [f'X{i}' for i in range(1, 15)] + [f'U{i}' for i in range(1, dim_u + 1)]
    df = []

    for i in Path(in_path + "CHEM_trainingdata/").iterdir():
        df_i = pd.read_csv(i)
        row = []
        target = df_i['Y'][df_i['t'] >= 40].mean()
        sys_id = create_id(df_i.loc[0, ["System"]])
        row.append(sys_id)
        row.append(target)
        row.append(df_i['Y'].iloc[0])
        row.extend(
            df_i.iloc[0][[f'X{i}' for i in range(1, 15)] + [f'U{i}' for i in range(1, dim_u + 1)]].values.tolist())
        df.append(row)

    data = pd.DataFrame(df, columns=columns)
    features = [x for x in list(data) if x != 'target']

    return data[features], data['target']


def search_opt_model(X, y, model, param_grid):
    regressor = GridSearchCV(model, param_grid, cv=10)
    regressor.fit(X, y)
    print(regressor.best_estimator_)
    return regressor.best_estimator_


def fit_predict(model, X, diff):
    model.fit(X, diff)
    return model.predict(X)


def my_function(i, row, model, scaler):
    dim_u = 8
    target, features = row[['target']], row[
        ['system_id', 'Y'] + [f'X{i}' for i in range(1, 15)] + [f'U{i}' for i in range(1, dim_u + 1)]]
    features_normalized = scaler.transform(features.values.reshape(1, -1))
    features_without_u = features_normalized[0][:-dim_u]
    theta = model.coef_
    theta_0 = model.intercept_

    C_1 = np.hstack((np.identity(dim_u), np.zeros((dim_u, 2))))
    c_2 = np.hstack((np.zeros(dim_u), np.ones(2)))
    c_3 = np.hstack((theta[-dim_u:] / scaler.scale_[-dim_u:], -1, 1))
    C_4 = np.hstack((np.zeros((2, dim_u)), np.identity(2)))
    b = -theta_0 - theta[:-dim_u] @ features_without_u + target + (scaler.mean_[-dim_u:] / scaler.scale_[-dim_u:]) @ theta[-dim_u:]
    z = cp.Variable(dim_u + 2)

    objective = cp.Minimize((cp.norm(C_1 @ z) / (20 * np.sqrt(dim_u)) + c_2 @ z) / 600)
    constraints = [0 <= C_4 @ z, c_3 @ z == b]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    u = z.value[:-2]

    return {"i": i, "u": u, "train_err": prob.value}


if __name__ == "__main__":
    dim_u = 8

    X, y = get_xy(dim_u)
    df_submit = pd.read_csv(in_path + 'CHEM_starter_kit/submission_template.csv')
    df_submit["system_id"] = create_id(df_submit["System"])

    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    model = Ridge(alpha=100.0, random_state=123)
    # opt_model = search_opt_model(X_normalized, y, model, param_grid={'alpha': [1e-1, 1, 10, 1e2, 1e3, 1e4]})

    pred = fit_predict(model, X_normalized, y)
    r2 = model.score(X_normalized, y)
    print("R^2: ", r2)
    print("X.columns: ", X.columns)

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    result = pool.starmap(partial(my_function, model=model, scaler=scaler),
                          [(i, row) for (i, row) in df_submit.iterrows()])

    train_err = []
    for i, r_i in enumerate(result):
        if i != r_i["i"]:
            raise Exception("Index does not match.")

        df_submit.loc[i, [f'U{i}' for i in range(1, dim_u + 1)]] = r_i["u"]
        train_err.append(r_i["train_err"])

    plt.plot(list(range(df_submit.shape[0])), train_err)
    plt.show()

    print("train_err: ", np.sum(train_err))
    file_name = 'ridge_cvx_alter_submission.csv'
    # train_err: 0.09222022523313707

    df_submit.to_csv(out_path + file_name, index=False)
