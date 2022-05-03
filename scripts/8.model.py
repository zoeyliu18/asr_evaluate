import pandas as pd
from sklearn import preprocessing
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import MinMaxScaler
import sys

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

data = ''
if 'hupa' in sys.argv[1]:
	data = pd.read_csv(sys.argv[1], sep = ',')
else:
	data = pd.read_csv(sys.argv[1], sep = '\t')
#data = pd.read_csv('data/hupa/hupa_top_tier_full_regression.txt', sep = ',')
#data = pd.read_csv('data/hupa/hupa_top_tier_regression.txt', sep = '\t')
X = data[['Duration_ratio', 'Pitch_ratio', 'Intensity_ratio', 'PPL_ratio', 'Num_word_ratio', 'Word_type_ratio', 'OOV_ratio', 'Train_Duration']]
#X = normalize(X)
X = MinMaxScaler().fit_transform(X)
X = pd.DataFrame(X, columns = ['Duration_ratio', 'Pitch_ratio', 'Intensity_ratio', 'PPL_ratio', 'Num_word_ratio', 'Word_type_ratio', 'OOV_ratio', 'Train_Duration'])
X = sm.add_constant(X)
new_data = X
new_data['WER'] = data['WER']
new_data['Speaker'] = data['Speaker']
new_data['Evaluation'] = data['Evaluation']
new_data['Group'] = data['Speaker'].astype('str') + data['File'].astype('str')

#sub = data.loc[data['Evaluation'] == 'heldout_speaker']

lm = smf.mixedlm(formula = 'WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio + Num_word_ratio + Word_type_ratio + OOV_ratio', data = new_data, groups = new_data['Speaker'])
fit = lm.fit()
print(fit.summary())

## For Hupa and Swahili
lm = smf.ols(formula = 'WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio + Num_word_ratio + Word_type_ratio + OOV_ratio', data = new_data)
fit = lm.fit()
print(fit.summary())

#lm = smf.mixedlm(formula = 'WER ~ Evaluation * (Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio + Num_word_ratio + Word_type_ratio + OOV_ratio)', data = new_data, groups = new_data['Speaker'])
#fit = lm.fit()


mdata=[]
fdata=[]
with open('iban_eval.txt') as f:
	for line in f:
		if 'ibm' in line:
			toks = line.split('\t')
			mdata.append(float(toks[-2]))
		if 'ibf' in line:
			toks = line.split('\t')
			fdata.append(float(toks[-2]))

#variables = fit.model.exog
#vif = [variance_inflation_factor(variables, i) for i in range(variables.shape[1] - 1)]

#lm = smf.mixedlm(formula = 'WER ~ Duration_ratio * Pitch_ratio * Intensity_ratio * PPL_ratio * Num_word_ratio * Word_type_ratio * OOV_ratio * Evaluation', data = new_data, groups = new_data['Speaker'], re_formula = "Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio + Num_word_ratio + Word_type_ratio + OOV_ratio + Evaluation + Duration_ratio : Pitch_ratio : Intensity_ratio : PPL_ratio : Num_word_ratio : Word_type_ratio : OOV_ratio : Evaluation")


#lm.fit_regularized(alpha=2., L1_wt=0,refit=False)
