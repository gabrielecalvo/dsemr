import numpy as np
import scipy.optimize as so
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.mod(90 - np.degrees(np.arctan2(y, x)), 360)
    return rho, phi

def base_model(ws):
    v_cutin = 3
    v_rated = 10
    v_cutout = 25
    P_rated = 1
    half_cp_rho_A = 1e-3
    
    pred_p = np.zeros(ws.shape)
    is_ramping = (ws >= v_cutin) & (ws < v_rated)
    is_rated = (ws >= v_rated) & (ws < v_cutout) 
    pred_p[is_ramping] = half_cp_rho_A * ws[is_ramping]**3
    pred_p[is_rated] = P_rated
    return pred_p
    
def eval_metrics(y_pred, y_actual, X, plot=True):
    MAE = mean_absolute_error(y_actual, y_pred)
    RMSE = mean_squared_error(y_actual, y_pred)**0.5
    BIAS = np.mean(y_actual-y_pred)
    R2 = r2_score(y_actual, y_pred)
    print(f'Evaluation Metrics:\n- MAE: {MAE:.4f}\n- RMSE: {RMSE:.4f}\n- BIAS: {BIAS:.4f}\n- R2: {R2:.4f}')
    
    if plot:
        df_plt = pd.DataFrame({'y_pred':y_pred, 'X': X, 'y_actual': y_actual})
        
        fig, axes = plt.subplots(1,2)
        fig.set_size_inches(20,4)
        df_plt.plot.scatter('X', 'y_pred', s=1, alpha=0.4, ax=axes[0])
        df_plt.plot.scatter('X', 'y_actual', c='r', s=1, alpha=0.4, ax=axes[0])
        df_plt.plot.scatter('y_pred', 'y_actual', c='m', s=1, ax=axes[1])
        
        fig, ax = plt.subplots()
        fig.set_size_inches(20,4)
        df_plt['y_pred'].iloc[:1000].plot(style='b', ax=ax, label='forecast')
        df_plt['y_actual'].iloc[:1000].plot(style='k', ax=ax, label='actual')
        ax.grid()
        ax.legend()

    return MAE, RMSE, BIAS

class BaseSolarModel:
    def fit(self, X, y):
        self.m = so.fmin(lambda b, x, y: ((b*x-y)**2).sum(), x0=0.1, args=(X, y))
        return self

    def predict(self, X):
        return self.m*X
