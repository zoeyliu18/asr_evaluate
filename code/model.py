import pandas as pd
from sklearn import preprocessing
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import MinMaxScaler

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


data = pd.read_csv('data/wolof/wolof_regression.txt', sep = '\t')
X = data[['Duration_ratio', 'Pitch_ratio', 'Intensity_ratio', 'PPL_ratio', 'Num_word_ratio', 'Word_type_ratio', 'OOV_ratio']]
#X = normalize(X)
X = MinMaxScaler().fit_transform(X)
X = sm.add_constant(X)
new_data = X
new_data['WER'] = data['WER']
new_data['Speaker'] = data['Speaker']
new_data['Evaluation'] = data['Evaluation']

#sub = data.loc[data['Evaluation'] == 'heldout_speaker']

lm=smf.mixedlm(formula='WER~ Duration_ratio+Pitch_ratio+Intensity_ratio+PPL_ratio+Num_word_ratio+Word_type_ratio+OOV_ratio', data=new_data, groups=new_data['Speaker'])
fit = lm.fit()
variables = fit.model.exog
vif = [variance_inflation_factor(variables, i) for i in range(variables.shape[1] - 1)]

lm=smf.mixedlm(formula='WER~ Duration_ratio+Pitch_ratio+Intensity_ratio+PPL_ratio+Num_word_ratio+Word_type_ratio+OOV_ratio', data=new_data, re_formula="Duration_ratio+Pitch_ratio+Intensity_ratio", groups=new_data['Speaker'])
fit = lm.fit()
variables = fit.model.exog
vif = [variance_inflation_factor(variables, i) for i in range(variables.shape[1] - 1)]

lm=smf.mixedlm(formula='WER~ Duration_ratio+Pitch_ratio+Intensity_ratio+PPL_ratio+Num_word_ratio+Word_type_ratio+OOV_ratio', data=new_data, re_formula="Duration_ratio+Pitch_ratio+Intensity_ratio+PPL_ratio+Num_word_ratio+Word_type_ratio+OOV_ratio", groups=new_data['Speaker'])
fit = lm.fit()

lm=smf.mixedlm(formula='WER~ Duration_ratio+Pitch_ratio+Intensity_ratio+PPL_ratio+OOV_ratio+Evaluation', data=new_data, re_formula="Duration_ratio+Pitch_ratio+Intensity_ratio+PPL_ratio+OOV_ratio", groups=new_data['Speaker'])
fit = lm.fit()

variables = fit.model.exog
vif = [variance_inflation_factor(variables, i) for i in range(variables.shape[1])]



lm=smf.ols(formula='WER~ Duration_ratio+Pitch_ratio+Intensity_ratio+PPL_ratio+Num_word_ratio+Word_type_ratio+OOV_ratio+Evaluation', data=new_data)
fit = lm.fit()
variables = fit.model.exog
vif = [variance_inflation_factor(variables, i) for i in range(variables.shape[1])]


#lm.fit_regularized(alpha=2., L1_wt=0,refit=False)

lm=smf.ols(formula='WER~ Duration_ratio + Pitch_ratio',data=new_data)
lm=smf.ols(formula='WER~ Duration_ratio + Intensity_ratio',data=new_data)
lm=smf.ols(formula='WER~ Duration_ratio + PPL_ratio',data=new_data)
lm=smf.ols(formula='WER~ Duration_ratio + Num_word_ratio',data=new_data)
lm=smf.ols(formula='WER~ Duration_ratio + Word_type_ratio',data=new_data)
lm=smf.ols(formula='WER~ Duration_ratio + OOV_ratio',data=new_data)


lm=smf.ols(formula='WER~ Pitch_ratio+Intensity_ratio+PPL_ratio+Evaluation',data=new_data)

variables = lm.fit().model.exog
vif = [variance_inflation_factor(variables, i) for i in range(variables.shape[1] - 1)]
vif

fongbe: pitch, intensity

Pitch_ratio                        0.0039      0.000     37.203      0.000       0.004       0.004
Intensity_ratio                    0.4851      0.096      5.063      0.000       0.297       0.673

iban: pitch, intensity

Pitch_ratio                        0.0005   1.63e-05     29.965      0.000       0.000       0.001
Intensity_ratio                   -0.4630      0.045    -10.258      0.000      -0.552      -0.375

hupa top tier: pitch, intensity

Pitch_ratio                -0.0002      0.000     -1.389      0.165      -0.001    9.87e-05
Intensity_ratio            -3.6656      0.189    -19.381      0.000      -4.036      -3.295

ratio was computed as test / train
therefore a negative coefficient indicates the
a negative coefficient indicates that the larger the average intensity ratio is between the training and the test set, the lower WER

backup['Evaluation'].replace({'len':'len_different'}, inplace=True)
