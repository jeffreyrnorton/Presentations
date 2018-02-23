---
title: "Regression"
author: "Jeffrey Norton"
date: "February 2018"
output:
  html_document:
    keep_md: yes
  pdf_document: default
  word_document: default
---



# Regression

An overview of regression.

[Return to Musings On AI Resources](http://musingsonai.net/resources/).

## Linear regression model

The linear regression model describes how the dependent variable is related to the independent
variable(s) and the error term:

$y = \beta_0 + \beta_1 x_1 + u$

or in matrix form

$y = \mathbf{\bar{x}} \overrightarrow{\beta} +u$

where

* $y$ is the dependent variable.  Always just one dependent variable in linear regression.  This variable is the *predicted* or *response* variable.
* $x$ is the independent variable.  Simple linear regression has one independent model.  Multiple linear regression has two or more independent variables (matrix form).  These variables are the *control* variables or *regressors*.
* $\beta$ are unknown parameters to be estimated,
  * $\beta_0$ is the intercept
  * $\beta_1$ is the slope
* $u$ is the error term (disturbance).  Sometimes denoted with $e$.

## Ordinary Least Squares (OLS)

Minimize the sum of the squared residuals $u$.

$min \sum u^2 = min \sum (y-\hat{y})^2 = min \sum (y - \beta_0 - \beta_1 x)^2$

In simple linear regression, we calculate $\beta_1$ as

$\beta_1 = \frac{cov(x,y)}{var(x)}$

In probability theory and statistics, **covariance** is a measure of the joint variability of two random variables.  If the greater values of one variable mainly correspond with the greater values of the other variable, and the same holds for the lesser values, i.e., the variables tend to show similar behavior, the covariance is positive.

In terms the fit and the original data,

$cov(x,y) = \sum_i^N \frac{(x_i - \hat{x})(y_i - \hat{y})}{N}$

and

$var(x) = cov(x,x) = \sum_i^N \frac{(x_i - \hat{x})^2}{N}$

$\beta_0 = y - \beta_1 x$

Note that the *standard deviation* is the square root of the variance (as given above):

$\sigma = \sqrt{\sum_i^N \frac{(x_i - \hat{x})^2}{N}}$

Assumptions of the OLS estimator:

* Exogeneity of regressors - without exogeneity of regressors, we have bias in the coefficients.
* Homoscedasticity - constant variance.
* Uncorrelated observations - There is no autocorrelation.

There is another [way](https://leanpub.com/regmods/read) to look at the relationship between linear regression and variance and covariance.  This starts with Galton's data on the relationship between a parent's and a child's height.

$\beta_1 = \frac{cor(y,x) \sigma(y)}{sigma(x)}$

and

$\beta_0 = \bar{y} - \beta_1 \bar{x}$


```r
library(UsingR)
```


```r
data(galton)

y <- galton$child
x <- galton$parent
# beta1 is the slope
beta1 <- cor(y, x) *  sd(y) / sd(x)
# beta0 is the intercept
beta0 <- mean(y) - beta1 * mean(x)
rbind(c(beta0, beta1), coef(lm(y ~ x)))
```

```
##      (Intercept)         x
## [1,]    23.94153 0.6462906
## [2,]    23.94153 0.6462906
```

```r
plot(lm(y ~ x))
```

![](Regression2_files/figure-html/unnamed-chunk-1-1.png)<!-- -->![](Regression2_files/figure-html/unnamed-chunk-1-2.png)<!-- -->![](Regression2_files/figure-html/unnamed-chunk-1-3.png)<!-- -->![](Regression2_files/figure-html/unnamed-chunk-1-4.png)<!-- -->

## R-Squared

The coefficient of determination $R^2$ provides a measure of the goodness of fit.

$R^2 = \frac{SSR}{SST}$ where SSR is the sum of squares due to regression, SSE is the sum of squares due to error and SST = SSR + SSE.  $R^2 \rightarrow 1$ indicates a perfect fit (that is, no variation due to error) and $R^2 \rightarrow 0$ indicates a poor fit.

It is also written as
$R^2 = \frac{\sum{\hat{y}_i - \bar{y}}}{\sum{y_i - \bar{y}}}$

Here we have two examples that have perfect correlation and one that does not:


```r
x <- c(1,2,3,4,5)
y <- c(2,4,6,8,10)
rsq <- function (x, y) cor(x, y) ^ 2
print(rsq(x,y))
```

```
## [1] 1
```

```r
v <- c(1,2,3,4,5)
w <- c(5,4,3,2,1)
print(rsq(v,w))
```

```
## [1] 1
```

```r
w <- c(5,3,4,7,5)
print(rsq(v,w))
```

```
## [1] 0.1818182
```

$R^2$ always increases when a new variable is added.  SST is still the same, but SSE declines and SSSR increases.  For this, we should use adjusted $R^2$ to correct for the number of independent variables:

$R^2_a = 1 - (1-R^2)\frac{n-1}{n-p-1}$

where p is the number of independent variables and n is the number of observations.

## Null Hypothesis

From [wikipedia](https://en.wikipedia.org/wiki/Null_hypothesis):

In a statistical test, the hypothesis that there is no significant difference between specified populations, any observed difference being due to sampling or experimental error.

A null hypothesis is rejected if the observed data are significantly unlikely to have occurred if the null hypothesis were true. In this case the null hypothesis is rejected and an alternative hypothesis is accepted in its place. If the data are consistent with the null hypothesis, then the null hypothesis is not rejected. In neither case is the null hypothesis or its alternative proven; the null hypothesis is tested with data and a decision is made based on how likely or unlikely the data are. This is analogous to the legal principle of presumption of innocence, in which a suspect or defendant is assumed to be innocent (null is not rejected) until proven guilty (null is rejected) beyond a reasonable doubt (to a statistically significant degree).

* A small p-value (typically $\le$ 0.05) indicates strong evidence against the null hypothesis, so you reject the null hypothesis.

* A large p-value ($gt$ 0.05) indicates weak evidence against the null hypothesis, so you fail to reject the null hypothesis.

* p-values very close to the cutoff (0.05) are considered to be marginal (could go either way). Always report the p-value so your readers can draw their own conclusions.

When the null hypothesis is rejected, we opt to use the alternative hypothesis instead.

## t-test for significance of one coefficient

Determines whether the relationship between $y$ and $x_j$ is significant.

The null hypothesis is that the coefficient is not significantly different than zero, that is, $H_0: \beta_j = 0$.

The alternative hypothesis is that the coefficient is significantly different from zero, that is, $H_a: \beta_j \ne 0$.

We use the t-distribution:

* The test statistic t = coefficient /standard error

* The critical values are from the t distribution

* The test is a two-tailed test.

  ![alt](http://trendingsideways.com/wp-content/uploads/2013/05/ztest4.png)

Reject the null hypothesis and conclude that coefficient is significantly different from zero (in the red zone) if:

* The test statistic t is in the critical rejection zone
* (equivalently) The p-value is less than 0.05

The goal is to find coefficients that are significant.

From [wikipedia](https://en.wikipedia.org/wiki/Student%27s_t-test):

Most $t$-test statistics have the form $\frac{Z}{s}$, where $Z$ and $s$ are functions of the data. Typically, $Z$ is designed to be sensitive to the alternative hypothesis (i.e., its magnitude tends to be larger when the alternative hypothesis is true), whereas $s$ is a scaling parameter that allows the distribution of $t$ to be determined.

As an example, in the one-sample $t$-test

$t=\frac{Z}{x} = \frac{(\bar{X}-\mu) \bigg{/}  \frac{\sigma}{\sqrt{n}}}{s}$
where $\bar{X}$ is the sample mean from a sample $X_1, X_2, ., X_n$, of size $n$, $s$ is the ratio of sample standard deviation over population standard deviation, $\sigma$ is the population standard deviation of the data, and $\mu$ is the population mean.

## Formula of paired samples t-test

t-test statistisc value can be calculated using the following formula for some sample $d$:

$t = \frac{m}{s \big{/} \sqrt{n}}$

where,

* $m$ is the mean differences
* $n$ is the sample size (i.e., size of $d$).
* $s$ is the standard deviation of $d$

We can compute the p-value corresponding to the absolute value of the t-test statistics ($|t|$) for the degrees of freedom (df): $df=n - 1$.

If the p-value is inferior or equal to 0.05, we can conclude that the difference between the two paired samples are significantly different.


```r
set.seed(19921)
x <- rnorm(10)
y <- rnorm(10)
print(x)
```

```
##  [1]  1.0419794  0.9141689 -1.2393180  1.1846819 -0.3274486  1.8384269
##  [7] -0.8368079  1.0307982  2.7355513  0.5514608
```

```r
print(y)
```

```
##  [1] -0.9945195  1.0565387  0.2118059  2.0095213 -0.9217237 -0.6227325
##  [7]  0.7006537  0.7714340 -0.1881614  0.7103663
```

```r
ttest <- t.test(x,y)
print(ttest)
```

```
## 
## 	Welch Two Sample t-test
## 
## data:  x and y
## t = 0.85217, df = 17.128, p-value = 0.4059
## alternative hypothesis: true difference in means is not equal to 0
## 95 percent confidence interval:
##  -0.6134016  1.4454636
## sample estimates:
## mean of x mean of y 
## 0.6893493 0.2733183
```

## F-test for overall significance of all coefficients

* Testing whether the relationship between y and all x variables is significant.
* The null hypothesis is that the coefficients are not jointly significantly different from zero, that is, $H_0: \beta_1 = \beta_2 = ... = \beta_p = 0$.
* The alternative hypothesis is that the coefficients are jointly significantly different from zero, that is, $H_a : \beta_1 \ne 0$ or $\beta_2 \ne 0 $ or ... or $\beta_p \ne 0$
* Use the F-distribution
  * The test statistic F = MSR/MSE where MSR is the regression mean square and MSE is the mean squared error.  For degrees of freedom = 1,
    * MSE $= \frac{\sum (y_i - \hat{y}_i)^2}{n-2} = \frac{SSE}{n-2}$
    * MSR $= \sum (\hat{y}_i - \bar{y}_i)^2 = SSR$
    * WHERE
      - $\hat{y}$ is the predicted value
      - $\bar{y}$ is the average (mean) value
      - $y$ is the actual value
  * The critical values are from the F distribution
  * The F-test is an upper one-tail test
  
  ## ANOVA Table

ANOVA - ANalysis Of Variance.

| Source         | Sum of Squares                   | Degrees of Freedom                        | Mean Square       | F-Statistic |
| -------------- | -------------------------------- | ----------------------------------------- | ----------------- | :---------: |
| **Regression** | SSR $=\sum(\hat{y} - \bar{y})^2$ | p = number of independent variables       | MSR = SSR/p       |  F=MSR/MSE  |
| **Error**      | SSF $=\sum(y - \hat{y})^2$       | n-p-1                                     | MSE = SSE/(n-p-1) |             |
| **Total**      | SST $=\sum(y - \bar{y})^2$       | n-1 where n is the number of observations |                   |             |

* Find critical values in the F table (significance level =0.05)
  * degrees of freedom in the numerator = number of independent variables = p
  * degrees of freedom in the denominator = n-p-1
* Reject the null hypothesis if the F-test statistic is greater than the F-critical value.
* Reject the null hypothesis if the p-value is less than 0.05.
* The goal is to find a regression model with coefficients that are jointly significant. 

[Examples](https://docs.google.com/file/d/0BwogTI8d6EEiTS1SWjRjX1RWeGc/edit)

R - [lm - Fitting Linear Models](https://www.rdocumentation.org/packages/stats/versions/3.4.3/topics/lm)

R - [t.test - Student T-Test](https://www.rdocumentation.org/packages/stats/versions/3.4.3/topics/t.test)

R - [var.test - F Test for comparing two variances](https://www.rdocumentation.org/packages/stats/versions/3.4.3/topics/var.test)

Python - [scipy.stats.linregress](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.linregress.html)

Python - [scipy.stats.ttest_ind](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html)

Python - [scipy.stats.f_oneway](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html)



```r
# Copyright 2013 by Ani Katchova
# The file is at https://docs.google.com/file/d/0BwogTI8d6EEiZnBzTk9Ec3d3Qjg/edit.
#
mydata<- read.csv("regression_auto.csv")
attach(mydata)
```

```r
# Define variables
Y <- cbind(mpg)
X1 <- cbind(weight1)
X <- cbind(weight1, price, foreign)
# Descriptive statistics
summary(Y)
```

```
##       mpg       
##  Min.   :14.00  
##  1st Qu.:17.25  
##  Median :21.00  
##  Mean   :20.92  
##  3rd Qu.:23.00  
##  Max.   :35.00
```

```r
summary(X)
```

```
##     weight1          price          foreign      
##  Min.   :2.020   Min.   : 3299   Min.   :0.0000  
##  1st Qu.:2.643   1st Qu.: 4466   1st Qu.:0.0000  
##  Median :3.200   Median : 5146   Median :0.0000  
##  Mean   :3.099   Mean   : 6652   Mean   :0.2692  
##  3rd Qu.:3.610   3rd Qu.: 8054   3rd Qu.:0.7500  
##  Max.   :4.330   Max.   :15906   Max.   :1.0000
```

```r
# Correlation among variables
cor(Y, X)
```

```
##        weight1      price   foreign
## mpg -0.8081609 -0.4384618 0.4003376
```

```r
# Plotting data on a scatter diagram
plot(Y ~ X1, data = mydata)
# Simple linear regression
olsreg1 <- lm(Y ~ X1)
summary(olsreg1)
```

```
## 
## Call:
## lm(formula = Y ~ X1)
## 
## Residuals:
##     Min      1Q  Median      3Q     Max 
## -5.4123 -1.6073 -0.1043  0.9261  8.1072 
## 
## Coefficients:
##             Estimate Std. Error t value Pr(>|t|)    
## (Intercept)  38.0665     2.6112  14.578 2.02e-13 ***
## X1           -5.5315     0.8229  -6.722 5.93e-07 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 2.86 on 24 degrees of freedom
## Multiple R-squared:  0.6531,	Adjusted R-squared:  0.6387 
## F-statistic: 45.19 on 1 and 24 DF,  p-value: 5.935e-07
```

```r
confint(olsreg1, level=0.95)
```

```
##                 2.5 %    97.5 %
## (Intercept) 32.677256 43.455664
## X1          -7.229797 -3.833196
```

```r
anova(olsreg1)
```

```
## Analysis of Variance Table
## 
## Response: Y
##           Df Sum Sq Mean Sq F value    Pr(>F)    
## X1         1 369.57  369.57  45.189 5.935e-07 ***
## Residuals 24 196.28    8.18                      
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

```r
# Plotting regression line
abline(olsreg1)
```

![](Regression2_files/figure-html/unnamed-chunk-4-1.png)<!-- -->

```r
# Predicted values for dependent variable
Y1hat <- fitted(olsreg1)
summary(Y1hat)
```

```
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
##   14.12   18.10   20.37   20.92   23.45   26.89
```

```r
plot(Y1hat ~ X1)
```

![](Regression2_files/figure-html/unnamed-chunk-4-2.png)<!-- -->

```r
# Regression residuals
e1hat <- resid(olsreg1)
summary(e1hat)
```

```
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
## -5.4120 -1.6070 -0.1043  0.0000  0.9261  8.1070
```

```r
plot(e1hat ~ X1)
```

![](Regression2_files/figure-html/unnamed-chunk-4-3.png)<!-- -->

```r
# Multiple linear regression
olsreg2 <- lm(Y ~ X)
summary(olsreg2)
```

```
## 
## Call:
## lm(formula = Y ~ X)
## 
## Residuals:
##     Min      1Q  Median      3Q     Max 
## -4.6942 -1.1857 -0.0452  0.6433  8.6895 
## 
## Coefficients:
##               Estimate Std. Error t value Pr(>|t|)    
## (Intercept) 42.1661962  4.2647533   9.887 1.48e-09 ***
## Xweight1    -7.1211114  1.6046735  -4.438 0.000207 ***
## Xprice       0.0002258  0.0002654   0.851 0.404002    
## Xforeign    -2.5071265  2.0565685  -1.219 0.235723    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 2.89 on 22 degrees of freedom
## Multiple R-squared:  0.6752,	Adjusted R-squared:  0.6309 
## F-statistic: 15.25 on 3 and 22 DF,  p-value: 1.374e-05
```

```r
confint(olsreg2, level=0.95)
```

```
##                     2.5 %        97.5 %
## (Intercept)  3.332164e+01 51.0107531780
## Xweight1    -1.044900e+01 -3.7932221856
## Xprice      -3.245229e-04  0.0007760878
## Xforeign    -6.772188e+00  1.7579354345
```

```r
anova(olsreg2)
```

```
## Analysis of Variance Table
## 
## Response: Y
##           Df Sum Sq Mean Sq F value    Pr(>F)    
## X          3 382.08 127.360  15.247 1.374e-05 ***
## Residuals 22 183.77   8.353                      
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

```r
# Predicted values for dependent variable
Yhat <- fitted(olsreg2)
summary(Yhat)
```

```
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
##   13.90   17.91   20.46   20.92   23.99   27.89
```

```r
# Regression residuals
ehat <- resid(olsreg2)
summary(ehat)
```

```
##     Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
## -4.69400 -1.18600 -0.04524  0.00000  0.64330  8.68900
```

The following is python code for doing linear regression.  It won't work in this notebook, but I provide the code for completeness.


```python
from sklearn.linear_model import LinearRegression

X_train = [[1,2],[2,4],[6,7]]
y_train = [1.2, 4.5, 6.7]
X_test = [[1,3],[2,5]]    

# create a Linear Regressor   
lin_regressor = LinearRegression()

# convert to be used further to linear regression
X_transform = lin_regressor.fit_transform(X_train)

# fit this to Linear Regressor
lin_regressor.fit(X_transform,y_train) 

# get the predictions
y_preds = lin_regressor.predict(X_test)
```

# Other Regression Models

I suggest taking a good look at [Penn State's online course Stat 501](https://onlinecourses.science.psu.edu/stat501/) for more models.  Here are a few additional regression models.

## Polynomial Regression

Sometimes, a plot of the residuals versus a predictor may suggest there is a nonlinear relationship.

![alt](https://onlinecourses.science.psu.edu/stat501/sites/onlinecourses.science.psu.edu.stat501/files/ch17/17.1_plot_01.png)

One way to try to account for such a relationship is through a polynomial regression model. Such a model for a single predictor, $X$, is:

$Y = \beta_0 + \beta_1 X + \beta_2 X^2 + ... + \beta_h X^h + \epsilon$

where $h$ is called the degree of the polynomial. For lower degrees, the relationship has a specific name (i.e., $h = 2$ is called quadratic, $h = 3$ is called cubic, $h = 4$ is called quartic, and so on). Although this model allows for a nonlinear relationship between $Y$ and $X$, polynomial regression is still considered linear regression since it is linear in the regression coefficients, $\beta_1$, $\beta_2$, ..., $\beta_h$.

[R bloggers](https://www.r-bloggers.com/fitting-polynomial-regression-in-r/) shows how to do polynomial regression.  [Scikit](http://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions) shows how to use polynomial regressors to fit functions in Python.  Consider the [following problem](https://stackoverflow.com/questions/3822535/fitting-polynomial-model-to-data-in-r):


```r
x <- c(32,64,96,118,126,144,152.5,158)  
y <- c(99.5,104.8,108.5,100,86,64,35.3,15)

# Get 3rd degree polynomial 
fit <- lm(y ~ poly(x, 3, raw=TRUE))

# Let's let AIC give us the best degree fit up to 4 because beyond that lies insanity
bestpolyfit <- function(x, y) {
    polyfit <- function(i) x <- AIC(lm(y ~ poly(x, i, raw=TRUE)))
    degree <- as.integer(optimize(polyfit,interval=c(1, min(4, length(x)-1)))$minimum)
    fit <- lm(y ~ poly(x, degree, raw=TRUE))
    return(list(fit=fit, degree=degree, AIC=AIC(fit)))
}

polyregression <- bestpolyfit(x, y)
print(sprintf("A polynomial fit of degree %d has AIC=%f", polyregression$degree, polyregression$AIC))
```

```
## [1] "A polynomial fit of degree 3 has AIC=49.569365"
```

```r
# What about a 3rd degree spline?
library(splines)
spline_fit <- lm(y ~ ns(x, 3))
print( sprintf("Use 3rd degree spline if AIC(spline) < AIC(polyfit) : %f < %f.", AIC(spline_fit), polyregression$AIC) )
```

```
## [1] "Use 3rd degree spline if AIC(spline) < AIC(polyfit) : 51.064180 < 49.569365."
```

```r
plot(x, y, col="red") 
lines(x, predict(polyregression$fit), col="blue")
```

![](Regression2_files/figure-html/unnamed-chunk-6-1.png)<!-- -->

Polynomial regression in Python is very similar to linear regression.  Here is the code for completeness (but it is not executed in this notebook).


```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

X_train = [[1,2],[2,4],[6,7]]
y_train = [1.2, 4.5, 6.7]
X_test = [[1,3],[2,5]]    

# create a Linear Regressor   
lin_regressor = LinearRegression()

# pass the order of your polynomial here  
poly = PolynomialFeatures(2)

# convert to be used further to linear regression
X_transform = poly.fit_transform(X_train)

# fit this to Linear Regressor
lin_regressor.fit(X_transform,y_train) 

# get the predictions
y_preds = lin_regressor.predict(X_test)
```

## Nonlinear Regression

Nonlinear regression refers to the fact that the coefficients are not in a linear form as they are even in a polynomial equation.  [Penn State's](https://onlinecourses.science.psu.edu/stat501/node/370) page is pretty good about walking through this.  Take a look at the [nls](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/nls.html) documentation as well as the [R-blogger's page](https://www.r-bloggers.com/first-steps-with-non-linear-regression-in-r/).

First [example](https://datascienceplus.com/first-steps-with-non-linear-regression-in-r/) using the [Michaelis-Menten equation](https://en.wikipedia.org/wiki/Michaelis%E2%80%93Menten_kinetics):


```r
#simulate some data
set.seed(20160227)
x<-seq(0,50,1)
y<-((runif(1,10,20)*x)/(runif(1,0,10)+x))+rnorm(51,0,1)
#for simple models nls find good starting values for the parameters even if it throw a warning
m<-nls(y~a*x/(b+x), start=list(a=1,b=1))
#get some estimation of goodness of fit
print(sprintf("Correlation=%f", cor(y,predict(m))))
```

```
## [1] "Correlation=0.949660"
```

```r
print(m)
```

```
## Nonlinear regression model
##   model: y ~ a * x/(b + x)
##    data: parent.frame()
##      a      b 
## 11.848  4.278 
##  residual sum-of-squares: 30.01
## 
## Number of iterations to convergence: 6 
## Achieved convergence tolerance: 1.57e-06
```

```r
#plot
plot(x,y)
lines(x,predict(m),lty=2,col="red",lwd=3)
```

![](Regression2_files/figure-html/unnamed-chunk-8-1.png)<!-- -->

Selecting the starting values is important, because without doing so well, the solution may not [converge](https://stats.stackexchange.com/questions/183653/getting-the-right-starting-values-for-an-nls-model-in-r).


```r
modeldf <- data.frame(rev=c(17906.4,5303.72,2700.58,1696.77,947.53,362.03), weeks=c(1,2,3,4,5,6))
newMod <- nls(rev ~ a*weeks^b, data=modeldf, start = list(a=1,b=1))
```

```
## Error in nls(rev ~ a * weeks^b, data = modeldf, start = list(a = 1, b = 1)): singular gradient
```

```r
predict(newMod, newdata = data.frame(weeks=c(1,2,3,4,5,6,7,8,9,10)))
```

```
## Error in predict(newMod, newdata = data.frame(weeks = c(1, 2, 3, 4, 5, : object 'newMod' not found
```

Lets plot the data and see what we actually have.


```r
plot(modeldf$weeks, modeldf$rev, xlab="weeks", ylab="rev")
```

![](Regression2_files/figure-html/unnamed-chunk-10-1.png)<!-- -->

If we start with a=1 and b=1 and fill into the equation, we get at week 1 the following value for rev = 1 and that value is not close to the first value of rev at all (1.79064\times 10^{4}).

One suggestion which could work is to take the natural log of both sides and do a linear fit.  I think this is still tricky and may or may not work as experienced below.  Scaling by taking the log is a reasonable approach on data for many problems, but not failsafe.


```r
logdf <- log(modeldf)
lm(logdf$rev ~ logdf$weeks)
```

```
## 
## Call:
## lm(formula = logdf$rev ~ logdf$weeks)
## 
## Coefficients:
## (Intercept)  logdf$weeks  
##       9.947       -2.011
```

```r
newMod <- nls(rev ~ a*weeks^b, data=modeldf, start = list(a=9.947, b=-2.011))
```

```
## Error in numericDeriv(form[[3L]], names(ind), env): Missing value or an infinity produced when evaluating the model
```

```r
predict(newMod, newdata = data.frame(weeks=c(1,2,3,4,5,6,7,8,9,10)))
```

```
## Error in predict(newMod, newdata = data.frame(weeks = c(1, 2, 3, 4, 5, : object 'newMod' not found
```

This is a problem where we can probably guess that we need a large $a$ - say around $18000$ and substantial negative coefficient for $b$ - perhaps $-1$, but not all problems are guessable like this.

My preferred solution is to use the [Levenberg-Marquardt](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm) for solving nonlinear least squares problems.  This works brilliantly and probably works in most cases.


```r
require(minpack.lm)
```

```
## Loading required package: minpack.lm
```

```r
require(broom)
```

```
## Loading required package: broom
```

```r
fit <- nlsLM(rev ~ a*weeks^b, data=modeldf, start = list(a=1,b=1))

fit_data <- augment(fit)

m <-predict(fit, newdata = data.frame(weeks=c(1,2,3,4,5,6)))

print(fit)
```

```
## Nonlinear regression model
##   model: rev ~ a * weeks^b
##    data: modeldf
##         a         b 
## 17919.213    -1.763 
##  residual sum-of-squares: 204141
## 
## Number of iterations to convergence: 20 
## Achieved convergence tolerance: 1.49e-08
```

```r
plot(modeldf$weeks,modeldf$rev, xlab="weeks", ylab="rev")
lines(modeldf$weeks,m,lty=2,col="red",lwd=3)
```

![](Regression2_files/figure-html/unnamed-chunk-12-1.png)<!-- -->

Nonlinear fitting for python:

```python
import numpy as np
from sklearn import metrics
from scipy.optimize import curve_fit #we could import more, but this is what we need
###defining your fitfunction
def func(x, a, b, c):
    return a - b* np.exp(c * x)

# Data to fit
x = np.array([475, 108, 2, 38, 320])
y = np.array([95.5, 57.7, 1.4, 21.9, 88.8])

# guess some start values
initialGuess=[100, 100,-.01]
guessedFactors=[func(x,*initialGuess ) for x in x]

# Fit the curve
popt, pcov = curve_fit(func, x, y, initialGuess)

# popt are the calculated coefficients
print("Calculated coefficients = {}".format(popt))
# pcov is the coveriance matrix
print("Covariance Matrix")
print(pcov)

# How closs is the fit?
fittedData=[func(x, *popt) for x in x]
print("y_true")
print(y)
print("y_fitted")
print(fittedData)
print("R2 = {}".format(metrics.r2_score(y, fittedData)))
```
Calculated coefficients = [  9.73367195e+01,   9.86853716e+01,  -7.99827891e-03]  
Covariance Matrix  
[[  7.28014130e+00,   5.55658116e+00,   1.59027635e-03]  
  [  5.55658116e+00,   9.87606726e+00,   5.14102604e-04]  
  [  1.59027635e-03,   5.14102604e-04,   6.21028849e-07]]

y_true  
[ 95.5,  57.7,   1.4,  21.9,  88.8]

y_fitted  
[95.127245990786321, 55.735786167693526, 0.2174148868403023, 24.515883251665116, 89.703669701635533]

R2 = 0.9980629714658874



[Return to Musings On AI Resources](http://musingsonai.net/resources/).