# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 20:25:04 2020

@author: Wang
"""

import flask
import pandas as pd
import tensorflow as tf
from tensorflow import keras

model_P = tf.keras.models.load_model('Model_P.h5')
model_eta = tf.keras.models.load_model('Model_eta.h5')

mean_raw = pd.read_csv('mean.csv')
std_raw = pd.read_csv('std.csv')
mean = pd.Series(mean_raw['mean'].tolist(), index = mean_raw['Unnamed: 0'].tolist())
std = pd.Series(std_raw['std'].tolist(), index = std_raw['Unnamed: 0'].tolist())

app = flask.Flask(__name__, template_folder='templates')
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    
    if flask.request.method == 'POST':
        Tc = flask.request.form['Tc']
        Th = flask.request.form['Th']
        kn1 = flask.request.form['kn1']
        rhon1 = flask.request.form['rhon1']
        sn1 = flask.request.form['sn1']
        taun1 = flask.request.form['taun1']
        rhon2 = flask.request.form['rhon2']
        taun2 = flask.request.form['taun2']
        kp1 = flask.request.form['kp1']
        rhop1 = flask.request.form['rhop1']
        sp1 = flask.request.form['sp1']
        taup1 = flask.request.form['taup1']
        rhop2 = flask.request.form['rhop2']
        taup2 = flask.request.form['taup2']
        snh = flask.request.form['snh']
        snc = flask.request.form['snc']
        sph = flask.request.form['sph']
        spc = flask.request.form['spc']
        betan = flask.request.form['betan']
        betap = flask.request.form['betap']
        RL = flask.request.form['RL']
        thetacn = flask.request.form['thetacn']
        thetacp = flask.request.form['thetacp']
        Rcn = flask.request.form['Rcn']
        Rcp = flask.request.form['Rcp']
        N = flask.request.form['N']
        input_variables = pd.DataFrame([[Tc, Th, 
                                         kn1, rhon1, sn1, taun1, rhon2, taun2,
                                         kp1, rhop1, sp1, taup1, rhop2, taup2, 
                                         snh, snc, sph, spc, 
                                         betan,  betap, RL, N,
                                         thetacn, thetacp, Rcn, Rcp, 
                                         ]],
                                       columns=['T_c', 'T_h', 
                                                'k_n1', 'rho_n1', 's_n1', 'tau_n1', 'rho_n2', 'tau_n2',
                                                'k_p1', 'rho_p1', 's_p1', 'tau_p1', 'rho_p2', 'tau_p2', 
                                                's_nh', 's_nc', 's_ph', 's_pc', 
                                                'beta_n',  'beta_p', 'R_L', 'N',
                                                'theta_cn', 'theta_cp', 'R_cn', 'R_cp'], 
                                       dtype=float)
        data = input_variables.copy()
        data['R_L'] =  data['R_L']/data['N']
        N = data['N'].values[0]
        data.pop('N')
        
        def Norm(x):
            return  (x - mean)/std
        
        Normed_data = Norm(data)
        Power = model_P.predict(Normed_data)[0][0]*N
        Efficiency = model_eta.predict(Normed_data)[0][0]
    
        return flask.render_template('main.html',
                                     eta = round(Efficiency,3),
                                     P = round(Power,3),
                                     )

if __name__ == '__main__':
    app.run()
