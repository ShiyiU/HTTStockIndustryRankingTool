# Testing for stationarity & cointegration

### 1. What component (recycled gold, net production, or mine production) is mostly influencing the price movement?
In order to be able to answer such a question, one should factor in the **mutual dependence** formed between each of these components and gold price. In other words, it is knowing whether the gold price is influencing, for instance, the mine production as much as understanding to what extent the mine production is influencing the gold price.

A simple **regression-based solution** will not be of great help since the data is:
- suffering of autocorrelation (e.g., the price we saw yesterday is linked to the one we see today);
- non-stationarity (e.g., the data doesn't hover around a constant mean; it trends and wanders over time). Running basic correlations on non-stationary data often results in a "spurious correlation" (mathematically significant, but practically meaningless).

A **regression-based solution** will come into effect only if the following equation holds:
$$y_t - \beta x_t = I(0)
\tag 1$$
- where, $y_t$ and $x_t$ both represeting the targeted time series (e.g., gold price and one of the supply component);
- $\beta$ is a hyperparameter that can only take one unique value for which the above equation holds;

If the aforementioned equation (1) doesn't hold then taking a different allegedly stationary representation (e.g., **its $I(1)$/first-order integration**) and using that instead for testing might ultimately help us rule out the data.

### Resources:
[1. Cointegration - an introduction](https://www.youtube.com/watch?v=vvTKjm94Ars)

[2. Integration, Cointegration, and Stationarity](https://www.youtube.com/watch?v=Pn_RiDbK82M)