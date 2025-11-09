# Resumo do Modelo OLS (statsmodels)

```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:              log_price   R-squared:                       0.745
Model:                            OLS   Adj. R-squared:                  0.744
Method:                 Least Squares   F-statistic:                     528.5
Date:                Sun, 09 Nov 2025   Prob (F-statistic):               0.00
Time:                        13:42:29   Log-Likelihood:                -1844.7
No. Observations:                3093   AIC:                             3725.
Df Residuals:                    3075   BIC:                             3834.
Df Model:                          17                                         
Covariance Type:            nonrobust                                         
=======================================================================================
                          coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------------
const                   9.4152      0.008   1187.677      0.000       9.400       9.431
Mileage                -0.4487      0.009    -49.803      0.000      -0.466      -0.431
EngineV                 0.2090      0.010     20.982      0.000       0.190       0.229
Brand_BMW               0.0142      0.012      1.207      0.228      -0.009       0.037
Brand_Mercedes-Benz     0.0129      0.013      1.023      0.306      -0.012       0.038
Brand_Mitsubishi       -0.1406      0.011    -13.299      0.000      -0.161      -0.120
Brand_Renault          -0.1799      0.012    -15.519      0.000      -0.203      -0.157
Brand_Toyota           -0.0605      0.012     -5.237      0.000      -0.083      -0.038
Brand_Volkswagen       -0.0899      0.013     -6.885      0.000      -0.116      -0.064
Body_hatch             -0.1455      0.010    -14.532      0.000      -0.165      -0.126
Body_other             -0.1014      0.010    -10.403      0.000      -0.121      -0.082
Body_sedan             -0.2006      0.012    -16.608      0.000      -0.224      -0.177
Body_vagon             -0.1299      0.010    -12.402      0.000      -0.150      -0.109
Body_van               -0.1686      0.012    -14.179      0.000      -0.192      -0.145
Engine Type_Gas        -0.1215      0.009    -12.821      0.000      -0.140      -0.103
Engine Type_Other      -0.0334      0.008     -4.340      0.000      -0.048      -0.018
Engine Type_Petrol     -0.1469      0.010    -14.581      0.000      -0.167      -0.127
Registration_yes        0.3205      0.009     36.783      0.000       0.303       0.338
==============================================================================
Omnibus:                      683.222   Durbin-Watson:                   1.970
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             2701.464
Skew:                          -1.038   Prob(JB):                         0.00
Kurtosis:                       7.081   Cond. No.                         4.39
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```
