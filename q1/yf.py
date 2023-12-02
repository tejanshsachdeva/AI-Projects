import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.stdout.write("\r")
sys.stdout.flush()

# Download S&P 500 data from Yahoo Finance
ticker_symbol = "^GSPC"
start_date = "2000-01-01"
end_date = "2023-11-19"

sp500_data = yf.download(ticker_symbol, start=start_date, end=end_date, interval="1mo")
sp500_data = sp500_data['Adj Close'].dropna()

# Calculate monthly returns
returns = sp500_data.pct_change().dropna().values

# Function to perform EM algorithm for Markov regime switching
def em_algorithm(observations, n_states, n_iterations=100):
    num_obs = len(observations)
    regime_probs = np.zeros((num_obs, n_states))
    
    # Initialize parameters randomly
    initial_probs = np.random.rand(n_states)
    initial_probs /= initial_probs.sum()
    
    transition_matrix = np.random.rand(n_states, n_states)
    transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)
    
    regime_means = np.random.randn(n_states)
    regime_stds = np.random.rand(n_states)
    
    for _ in range(n_iterations):
        # E-step: Compute probabilities of being in each regime at each time point
        for i in range(num_obs):
            if i == 0:   
                regime_probs[i] = initial_probs
            else:   
                regime_probs[i] = transition_matrix @ regime_probs[i - 1]

            # M-step: Update regime parameters
            for j in range(n_states):
                likelihood = (
                    1 / (np.sqrt(2 * np.pi) * regime_stds[j]) *
                    np.exp(-0.5 * ((observations[i] - regime_means[j]) / regime_stds[j]) ** 2)
                )
                regime_probs[i, j] *= likelihood

            regime_probs[i] /= regime_probs[i].sum()

        # M-step: Update parameters
        initial_probs = regime_probs[0]
        transition_matrix = regime_probs[:-1].T @ regime_probs[1:] / regime_probs[:-1].sum(axis=0, keepdims=True)
        
        regime_means = np.zeros(regime_probs.shape[1])   
        for i in range(regime_probs.shape[1]):
            regime_means[i] = np.sum(regime_probs[:, i] * observations) / np.sum(regime_probs[:, i])

        regime_stds = np.zeros(regime_probs.shape[1])
        for i in range(regime_probs.shape[1]):
           regime_stds[i] = np.sqrt(np.sum(regime_probs[:, i] * (observations - regime_means[i]) ** 2) / np.sum(regime_probs[:, i]))

    return initial_probs, transition_matrix, regime_means, regime_stds

# Run EM algorithm
n_states = 2  # Number of regimes
initial_probs, transition_matrix, regime_means, regime_stds = em_algorithm(returns, n_states)

# Function to predict future regimes
def predict_future_regimes(initial_probs, transition_matrix, num_steps):
    future_regimes = np.zeros(num_steps, dtype=int)
    regime_probs = initial_probs.copy()
    
    step = 0
    while step < num_steps:
        regime_probs = transition_matrix.T @ regime_probs
        future_regimes[step] = regime_probs.argmax()
        step += 1

    return future_regimes

# Predict future regimes for 5 time steps
future_regimes = predict_future_regimes(initial_probs, transition_matrix, 5)

# Create a color array for the scatter plot
scatter_colors = [-1 for _ in range(len(returns) - 5)]
scatter_colors.extend(future_regimes)
scatter_colors = np.array(scatter_colors)

# Plot the regimes on the time series
plt.figure(figsize=(12, 8))
plt.plot(returns, label='S&P 500 Returns', alpha=0.7)
plt.scatter(range(len(returns)), returns, c=scatter_colors, cmap='coolwarm', label='Regimes', marker='x')
plt.title('Markov Regime-Switching Model on S&P 500 Index Returns')
plt.xlabel('Time')
plt.ylabel('Returns')
plt.legend()
plt.show()

# Plot the S&P 500 data
plt.figure(figsize=(12, 8))
plt.plot(sp500_data.index, sp500_data, label='S&P 500')
plt.title('S&P 500 Series')
plt.xlabel('Time')
plt.ylabel('Adjusted Close Price')
plt.legend()
plt.show()


# Print estimated parameters
print("Estimated Initial Probabilities:")
print(initial_probs)

print("\nEstimated Transition Matrix:")
print(transition_matrix)

print("\nEstimated Regime Means:")
print(regime_means)

print("\nEstimated Regime Standard Deviations:")
print(regime_stds)

# Print predicted future regimes
print("\nPredicted Future Regimes for 5 Time Steps:", future_regimes)