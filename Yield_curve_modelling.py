#so here I am going to model the yield curve using the Nelson-Siegel model, which is a popular method for fitting yield curves
#the Nelson-Siegel model is a parametric model that describes the yield curve as a function of three parameters: level, slope, and curvature
#the model is given by the following equation:

#y(t) = β0 + β1 * (1 - exp(-t/τ)) / (t/τ) + β2 * ((1 - exp(-t/τ)) / (t/τ) - exp(-t/τ))

#firstly there will be real market data downloaded, for US treasury yields, then calibrate the parameters with optimization techniques, and compare their effectiveness
#the NS model is a famous parametric model, where it describes the entire yield curve shape using a few parameters making it computationally efficient and easy to interpret
#the model represents a zero-coupon yield y(m) at maturity m using 4 parameters, so essentially the yields are going to be for zero coupon bonds with differing maturities which is interesting
#BETAzero is the interest rate level that the whole interest rate curve converges to

#getting the data will be from the 

#now importing all the required libraries
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
import datetime
import streamlit as st
import time
import scipy.optimize as optimize
import matplotlib.animation as animation
from tqdm import tqdm

# Define FRED ticker symbols for US Treasury Constant Maturity rates
# Source: 
fred_tickers = {
    '1M': 'DGS1MO', '3M': 'DGS3MO', '6M': 'DGS6MO',
    '1Y': 'DGS1', '2Y': 'DGS2', '3Y': 'DGS3', '5Y': 'DGS5',
    '7Y': 'DGS7', '10Y': 'DGS10', '20Y': 'DGS20', '30Y': 'DGS30'
}

# Define maturity in years corresponding to tickers
maturities_in_years = {
    '1M': 1/12, '3M': 3/12, '6M': 6/12,
    '1Y': 1, '2Y': 2, '3Y': 3, '5Y': 5,
    '7Y': 7, '10Y': 10, '20Y': 20, '30Y': 30
}

# Convert dictionary to sorted list of tuples for consistent order
maturity_map = sorted(maturities_in_years.items(), key=lambda item: item[1])
ordered_tickers = [fred_tickers[key] for key, _ in maturity_map]
ordered_maturities = [value for _, value in maturity_map]

st.header("Welcome to the Nielson-Siegel yield curve estimation model")
st.subheader("Here we are going to fit a continuous yield curve onto the datapoints retrieved from the FRED for different constant maturity rates")
st.write(ordered_maturities, ordered_tickers)

start_date = st.sidebar.date_input("Please input the starting date for retrieving rate data for the different maturitites", value="2020-01-01")
end_date = st.sidebar.date_input("Please input the last date, on which the FED has updated the rates", value="2025-06-02") 

#fetching the yield data from the FRED website now
yield_df = web.DataReader(ordered_tickers, 'fred', start_date, end_date)
# Rename columns to reflect maturities (e.g., '1Y', '10Y') rather than FRED codes
yield_df.columns = [key for key, _ in maturity_map]
st.write(yield_df)

#data cleaning , because there are missing values due to bank holidays etc. 
st.subheader("How many missing values we have from each type")
st.write(yield_df.isnull().sum())
#the data does not change much and it does not jump from day to day, that is why we can actually use forward fill to get rid of missing values
yield_df_cleaned = yield_df.ffill()
st.write("The cleaned dataframe with forward fill", yield_df_cleaned)
st.subheader("How many missing values we have from each type now")
st.write(yield_df_cleaned.isnull().sum())
#do backward fill because the first value is missing
if yield_df_cleaned.isnull().values.any():
    yield_df_cleaned = yield_df_cleaned.bfill()
st.subheader("Final cleaned dataframe")
st.write(yield_df_cleaned)

#because the Nelson-Siegel model expects the rates in decimals, so the pure mathematical form, we should have them in decimals
yield_df_decimal = yield_df_cleaned / 100
st.subheader("In decimals")
st.write(yield_df_decimal.head())

#visualizing the dataframe: there will be 2 plots, 1 with the yield curve snapshot, which is a scatter plot of yields versus the maturities for a single specific date 2. the times series evolution of the yield curves for different maturities
plot_date = st.sidebar.date_input("Input the date which is in the time window and want to see the rates on that date", value=start_date)
plot_date = pd.Timestamp(plot_date)

# If plot_date is not in your DataFrame (maybe it's a weekend/holiday), Streamlit will error out.
# To avoid this, let's do a safer selection:
if plot_date in yield_df_decimal.index:
    yields_on_date = yield_df_decimal.loc[plot_date]
else:
    # Pick the closest previous date if not available (typical with time series data)
    yields_on_date = yield_df_decimal.loc[yield_df_decimal.index <= plot_date].iloc[-1]
    st.info(f"Selected date not available, showing closest previous available date: {yields_on_date.name.strftime('%Y-%m-%d')}")

#now do the scatter plot and see what happens
fig0, ax = plt.subplots(figsize=(10, 6))
ax.scatter(ordered_maturities, yields_on_date.values, color='blue', label=f'Market Yields ({plot_date.strftime("%Y-%m-%d")})')
ax.set_title(f'US Treasury Yield Curve on {plot_date.strftime("%Y-%m-%d")}')
ax.set_xlabel('Maturity (Years)')
ax.set_ylabel('Yield (Decimal)')
ax.set_xticks(ordered_maturities)
ax.set_xticklabels([key for key, _ in maturity_map], rotation=45)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()
fig0.tight_layout()

# Streamlit display
st.subheader(f"The yields for different maturities on {plot_date} date")
st.pyplot(fig0)
########################################################################################################################################################################################################################
#now doing the plot for the time series of different maturities
# Select key maturities for time-series plot
key_maturities_plot = ['3M', '2Y', '10Y', '30Y']

fig2, ax = plt.subplots(figsize=(12, 7))
for maturity_label in key_maturities_plot:
    if maturity_label in yield_df_decimal.columns:
        ax.plot(
            yield_df_decimal.index,
            yield_df_decimal[maturity_label],
            label=f'{maturity_label} Yield'
        )

ax.set_title('Evolution of Key US Treasury Yields Over Time')
ax.set_xlabel('Date')
ax.set_ylabel('Yield (Decimal)')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)
fig2.tight_layout()

st.subheader('Evolution of Key US Treasury Yields Over Time')
st.pyplot(fig2)

#now stroing the relevant variables for subsequent sections, passing them to functions or storing them in a class instance
market_maturities = np.array(ordered_maturities)
yield_curve_data = yield_df_decimal #this is the full historical dataset

########################################################################################################################################################################################################################
#now we are going to implement the Nelson-Siegel model and see how it runs and then the actual yields etc.
#defining the NS model function
def nelson_siegel(maturity, beta0, beta1, beta2, tau1):
    """
    Nelson-Siegel yield curve model.

    Args:
        maturity (np.ndarray): Array of maturities in years.
        beta0 (float): Long-term level parameter.
        beta1 (float): Short-term slope parameter.
        beta2 (float): Medium-term curvature parameter.
        tau1 (float): Decay factor.

    Returns:
        np.ndarray: Array of calculated yields.
    """
    m = np.array(maturity)
    zero_maturity_mask = (m == 0)
    non_zero_maturity_mask = ~zero_maturity_mask
    results = np.zeros_like(m, dtype=float)

    # Avoid division by zero
    if tau1 < 1e-6:
        tau1 = 1e-6

    # Handle non-zero maturities
    m_nonzero = m[non_zero_maturity_mask]
    m_tau = m_nonzero / tau1
    exp_m_tau = np.exp(-m_tau)
    term1 = beta0
    term2 = beta1 * (1 - exp_m_tau) / m_tau
    term3 = beta2 * ((1 - exp_m_tau) / m_tau - exp_m_tau)
    results[non_zero_maturity_mask] = term1 + term2 + term3

    # Handle zero-maturity: y(0) = beta0 + beta1
    results[zero_maturity_mask] = beta0 + beta1

    return results

#defining the objective function for calibration (Sum of Squared Errors)
def sse_objective(params, maturities, market_yields):
    """
    Calculates the Sum of Squared Errors (SSE) between NS model and market yields.

    Args:
        params (list or tuple): List containing the NS parameters [beta0, beta1, beta2, tau1].
        maturities (np.ndarray): Array of market maturities.
        market_yields (np.ndarray): Array of observed market yields.

    Returns:
        float: Sum of Squared Errors.
    """
    beta0, beta1, beta2, tau1 = params
    model_yields = nelson_siegel(maturities, beta0, beta1, beta2, tau1)
    # Simple SSE - could add weights later if needed (e.g., by duration)
    return np.sum((model_yields - market_yields)**2)

# Select the specific date for calibration (using the last date from previous section)
calibration_date = yield_curve_data.index[-1]
market_yields_on_date = yield_curve_data.loc[calibration_date].values

# Define initial guesses for the parameters [beta0, beta1, beta2, tau1]
# Sensible starting points:
# beta0: long-term yield (e.g., yield at 30Y)
# beta1: short-term - long-term spread (e.g., 3M yield - 30Y yield)
# beta2: often starts around 0, related to hump shape
# tau1: decay factor, often around 1-2 years
initial_beta0 = market_yields_on_date[-1] # Longest maturity yield
initial_beta1 = market_yields_on_date[0] - market_yields_on_date[-1] # Short-Long spread
initial_beta2 = 0.0 # Start with no curvature
initial_tau1 = 1.5 # Common starting point for tau1
initial_guesses = [initial_beta0, initial_beta1, initial_beta2, initial_tau1]    
    
# Define parameter bounds (beta0, beta1, beta2 can be negative, tau1 must be positive)
# Bounds can help optimization convergence and ensure economic sense.
bounds = [
    (0, 0.2),     # beta0: Level (e.g., 0% to 20% yield)
    (-0.1, 0.1),   # beta1: Slope (can be positive or negative)
    (-0.2, 0.2),   # beta2: Curvature (can be positive or negative)
    (1e-3, 50)    # tau1: Decay (must be positive, reasonable upper limit)
]

# Perform the optimization using scipy.optimize.minimize
# 'L-BFGS-B' is a common choice that handles bounds
optimization_result = optimize.minimize(
    sse_objective,
    initial_guesses,
    args=(market_maturities, market_yields_on_date),
    method='L-BFGS-B',
    bounds=bounds
)

if optimization_result.success:
    optimized_params_ns = optimization_result.x
    beta0_opt, beta1_opt, beta2_opt, tau1_opt = optimized_params_ns
    st.markdown(f"### Nelson-Siegel Calibration Results ({calibration_date.strftime('%Y-%m-%d')})")
    st.success("Optimization Successful")
    st.write("**Optimized Parameters:**")
    st.write(f"- beta0: `{beta0_opt:.6f}`")
    st.write(f"- beta1: `{beta1_opt:.6f}`")
    st.write(f"- beta2: `{beta2_opt:.6f}`")
    st.write(f"- tau1:  `{tau1_opt:.6f}`")
else:
    st.markdown(f"### Nelson-Siegel Calibration Failed ({calibration_date.strftime('%Y-%m-%d')})")
    st.error("Optimization Failed")
    st.write(f"**Optimization Message:** {optimization_result.message}")
    # Fallback to initial guesses for plotting
    optimized_params_ns = initial_guesses
    beta0_opt, beta1_opt, beta2_opt, tau1_opt = optimized_params_ns

########################################################################################################################################################################################################################
# Calculate the fitted yield curve using the optimized parameters
# Create a denser set of maturities for a smoother curve plot
plot_maturities = np.linspace(market_maturities.min(), market_maturities.max(), 100)
fitted_yields_ns = nelson_siegel(plot_maturities, *optimized_params_ns)

# Calculate fitted yields at the original market maturities for RMSE calculation
fitted_yields_at_market_maturities_ns = nelson_siegel(market_maturities, *optimized_params_ns)

# Calculate Goodness-of-Fit: Root Mean Squared Error (RMSE)
sse_ns = optimization_result.fun if optimization_result.success else sse_objective(optimized_params_ns, market_maturities, market_yields_on_date)
rmse_ns = np.sqrt(sse_ns / len(market_maturities))
st.write(f"\nGoodness-of-Fit (Nelson-Siegel):")
st.write(f"  SSE:  {sse_ns:.8f}")
st.write(f"  RMSE (Root_Mean_Squared_Error): {rmse_ns:.8f} (Yield units, e.g., {rmse_ns*100:.4f}%)")
st.write("Often the RMSE is preferred rather than the SSE, because it is the square root of the average squared error, giving us an error measure in the same units as the yields")
########################################################################################################################################################################################################################
fig4, ax = plt.subplots(figsize=(10, 6))
ax.scatter(
    market_maturities, market_yields_on_date, color='blue',
    label=f'Market Yields ({calibration_date.strftime("%Y-%m-%d")})'
)
ax.plot(
    plot_maturities, fitted_yields_ns, color='red',
    label=f'Fitted Nelson-Siegel (RMSE={rmse_ns*100:.4f}%)'
)

ax.set_title(f'Nelson-Siegel Fit vs Market Yields ({calibration_date.strftime("%Y-%m-%d")})')
ax.set_xlabel('Maturity (Years)')
ax.set_ylabel('Yield (Decimal)')

# Create pretty labels (e.g., 3M, 2Y, etc.) 
maturity_labels = [
    f'{int(m*12)}M' if m < 1 else f'{int(m)}Y' for m in market_maturities
]
ax.set_xticks(market_maturities)
ax.set_xticklabels(maturity_labels, rotation=45)

ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()
fig4.tight_layout()

st.subheader("Nelson-Siegel Fit vs Market Yields")
st.pyplot(fig4)

########################################################################################################################################################################################################################
# --- Section 4: Implementing and Calibrating the Nelson-Siegel-Svensson Model ---

# Define the Nelson-Siegel-Svensson model function
def nelson_siegel_svensson(maturity, beta0, beta1, beta2, tau1, beta3, tau2):
    """
    Calculates yield using the Nelson-Siegel-Svensson model.

    Args:
        maturity (np.ndarray): Array of maturities in years.
        beta0 (float): Long-term level parameter.
        beta1 (float): Short-term slope parameter.
        beta2 (float): Medium-term curvature parameter.
        tau1 (float): First decay factor.
        beta3 (float): Second curvature parameter.
        tau2 (float): Second decay factor.

    Returns:
        np.ndarray: Array of calculated yields.
    """
    m = np.array(maturity)
    zero_maturity_mask = (m == 0)
    non_zero_maturity_mask = ~zero_maturity_mask

    results = np.zeros_like(m, dtype=float)

    # Ensure tau values are small positive numbers to avoid division by zero
    tau1 = max(tau1, 1e-6)
    tau2 = max(tau2, 1e-6)

    m_tau1 = m[non_zero_maturity_mask] / tau1
    m_tau2 = m[non_zero_maturity_mask] / tau2
    exp_m_tau1 = np.exp(-m_tau1)
    exp_m_tau2 = np.exp(-m_tau2)

    term1 = beta0
    term2 = beta1 * (1 - exp_m_tau1) / m_tau1
    term3 = beta2 * ((1 - exp_m_tau1) / m_tau1 - exp_m_tau1)
    term4 = beta3 * ((1 - exp_m_tau2) / m_tau2 - exp_m_tau2)

    results[non_zero_maturity_mask] = term1 + term2 + term3 + term4

    # Handle zero maturity case: y(0) = beta0 + beta1
    results[zero_maturity_mask] = beta0 + beta1

    return results

# Define the objective function for NSS calibration (Sum of Squared Errors)
def sse_objective_nss(params, maturities, market_yields):
    """
    Calculates the Sum of Squared Errors (SSE) between NSS model and market yields.

    Args:
        params (list or tuple): List containing NSS parameters [beta0, beta1, beta2, tau1, beta3, tau2].
        maturities (np.ndarray): Array of market maturities.
        market_yields (np.ndarray): Array of observed market yields.

    Returns:
        float: Sum of Squared Errors.
    """
    beta0, beta1, beta2, tau1, beta3, tau2 = params
    model_yields = nelson_siegel_svensson(maturities, beta0, beta1, beta2, tau1, beta3, tau2)
    return np.sum((model_yields - market_yields)**2)

# Use the same calibration date and market yields as NS
# calibration_date = yield_curve_data.index[-1] # Already defined
# market_yields_on_date = yield_curve_data.loc[calibration_date].values # Already defined

# Define initial guesses for the NSS parameters [beta0, beta1, beta2, tau1, beta3, tau2]
# Use NS results as starting point for the first four, initialize beta3 and tau2
# Check if optimized_params_ns exists and is valid before using it
if 'optimized_params_ns' in locals() and len(optimized_params_ns) == 4:
    initial_guesses_nss = list(optimized_params_ns) + [0.0, 3.0] # Add beta3=0, tau2=3 (different from tau1)
else:
    # Fallback if NS calibration failed or wasn't run
    initial_beta0_nss = market_yields_on_date[-1]
    initial_beta1_nss = market_yields_on_date[0] - market_yields_on_date[-1]
    initial_guesses_nss = [initial_beta0_nss, initial_beta1_nss, 0.0, 1.5, 0.0, 3.0]
    print("Warning: Using fallback initial guesses for NSS calibration.")

# Define parameter bounds for NSS
# Extend NS bounds, ensure tau1 > 0, tau2 > 0. Consider constraining tau2 != tau1 if needed.
bounds_nss = [
    (0, 0.2),     # beta0: Level
    (-0.1, 0.1),   # beta1: Slope
    (-0.2, 0.2),   # beta2: Curvature 1
    (1e-3, 50),    # tau1: Decay 1 (must be positive)
    (-0.2, 0.2),   # beta3: Curvature 2
    (1e-3, 50)     # tau2: Decay 2 (must be positive)
]

# Perform the optimization for NSS
optimization_result_nss = optimize.minimize(
    sse_objective_nss,
    initial_guesses_nss,
    args=(market_maturities, market_yields_on_date),
    method='L-BFGS-B', # or 'SLSQP'
    bounds=bounds_nss
)

########################################################################################################################################################################################################################
# === Reporting Calibration Results ===
if optimization_result_nss.success:
    optimized_params_nss = optimization_result_nss.x
    beta0_opt_nss, beta1_opt_nss, beta2_opt_nss, tau1_opt_nss, beta3_opt_nss, tau2_opt_nss = optimized_params_nss
    st.markdown(f"### Nelson-Siegel-Svensson Calibration Results ({calibration_date.strftime('%Y-%m-%d')})")
    st.success("Optimization Successful")
    st.write("**Optimized Parameters:**")
    st.write(f"- beta0: `{beta0_opt_nss:.6f}`")
    st.write(f"- beta1: `{beta1_opt_nss:.6f}`")
    st.write(f"- beta2: `{beta2_opt_nss:.6f}`")
    st.write(f"- tau1:  `{tau1_opt_nss:.6f}`")
    st.write(f"- beta3: `{beta3_opt_nss:.6f}`")
    st.write(f"- tau2:  `{tau2_opt_nss:.6f}`")
else:
    st.markdown(f"### Nelson-Siegel-Svensson Calibration Failed ({calibration_date.strftime('%Y-%m-%d')})")
    st.error("Optimization Failed")
    st.write(f"**Optimization Message:** {optimization_result_nss.message}")
    # Fallback to initial guesses for plotting/comparison
    optimized_params_nss = initial_guesses_nss
    beta0_opt_nss, beta1_opt_nss, beta2_opt_nss, tau1_opt_nss, beta3_opt_nss, tau2_opt_nss = optimized_params_nss

# === Calculate Fitted Yields and Goodness-of-Fit ===
fitted_yields_nss = nelson_siegel_svensson(plot_maturities, *optimized_params_nss)
fitted_yields_at_market_maturities_nss = nelson_siegel_svensson(market_maturities, *optimized_params_nss)

sse_nss = optimization_result_nss.fun if optimization_result_nss.success else sse_objective_nss(
    optimized_params_nss, market_maturities, market_yields_on_date)
rmse_nss = np.sqrt(sse_nss / len(market_maturities))

# === Model Comparison Output ===
st.markdown("### Model Comparison")
st.write(f"**Date:** {calibration_date.strftime('%Y-%m-%d')}")
st.write("#### Nelson-Siegel (NS):")
st.write(f"- SSE: `{sse_ns:.8f}`")
st.write(f"- RMSE: `{rmse_ns:.8f}` (`{rmse_ns*100:.4f}`%)")
st.write("#### Nelson-Siegel-Svensson (NSS):")
st.write(f"- SSE: `{sse_nss:.8f}`")
st.write(f"- RMSE: `{rmse_nss:.8f}` (`{rmse_nss*100:.4f}`%)")

# === Visualize the fitted curves ===
fig5, ax = plt.subplots(figsize=(12, 7))
ax.scatter(
    market_maturities, market_yields_on_date, color='blue',
    label=f'Market Yields ({calibration_date.strftime("%Y-%m-%d")})', zorder=5)
ax.plot(
    plot_maturities, fitted_yields_ns, color='red', linestyle='--',
    label=f'Fitted NS (RMSE={rmse_ns*100:.4f}%)')
ax.plot(
    plot_maturities, fitted_yields_nss, color='green',
    label=f'Fitted NSS (RMSE={rmse_nss*100:.4f}%)')

ax.set_title(f'NS vs NSS Fit Comparison ({calibration_date.strftime("%Y-%m-%d")})')
ax.set_xlabel('Maturity (Years)')
ax.set_ylabel('Yield (Decimal)')
ax.set_xticks(market_maturities)
ax.set_xticklabels(maturity_labels, rotation=45)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()
fig5.tight_layout()

st.subheader("NS vs NSS Fit Comparison")
st.pyplot(fig5)
########################################################################################################################################################################################################################
#next step is to analyse the parameter dynamics overtime and see how it changes and what actually happens
#now we are going to plot the parameter changes overtime in the equation, for this we are going to plot how the yield curve long term level, the slope and the curvature evolve overtime
#the main focus is going to be on the NS model, but for the NSS it is the same

# Function to calibrate NS for a single date, returning parameters or NaNs on failure
def calibrate_ns_single_date(date_yields, maturities, initial_guess, bounds):
    """
    Calibrates the Nelson-Siegel model for a single row of yield data.

    Args:
        date_yields (np.ndarray): Market yields for a single date.
        maturities (np.ndarray): Corresponding market maturities.
        initial_guess (list): Initial guess for [beta0, beta1, beta2, tau1].
        bounds (list): Parameter bounds for optimization.

    Returns:
        tuple: (optimized_params, success_flag)
               optimized_params is np.array or np.full(4, np.nan) on failure.
               success_flag is boolean.
    """
    # Handle rows with NaN yields - skip calibration for these dates
    # Note: Data cleaning in Section 2 should have removed NaNs, but this is a safeguard.
    if np.isnan(date_yields).any():
        return np.full(4, np.nan), False

    # Ensure initial guess conforms to bounds before starting optimization
    # This is good practice, especially when seeding from previous results.
    bounded_initial_guess = np.clip(initial_guess, [b[0] for b in bounds], [b[1] for b in bounds])
    # Ensure tau1 doesn't get clipped to exactly zero if lower bound is 1e-3
    if bounded_initial_guess[3] < bounds[3][0]:
         bounded_initial_guess[3] = bounds[3][0]


    result = optimize.minimize(
        sse_objective,
        bounded_initial_guess, # Use bounded guess
        args=(maturities, date_yields),
        method='L-BFGS-B',
        bounds=bounds
    )
    if result.success:
        return result.x, True
    else:
        # Return NaNs if optimization fails
        # print(f"Warning: Optimization failed for date with yields: {date_yields}. Message: {result.message}") # Optional warning
        return np.full(4, np.nan), False

########################################################################################################################################################################################################################
st.subheader("Starting Dynamic Calibration (Nelson-Siegel)")

# Prepare storage for parameters
parameter_history_ns = []
dates_processed = []

# Initialize guess for the first day (use logic from Section 3)
# Ensure yield_curve_data is available and has data
if not yield_curve_data.empty:
    first_day_yields = yield_curve_data.iloc[0].values
    # Recalculate initial guesses based on the *first* day's data
    initial_beta0_first = first_day_yields[-1] if not np.isnan(first_day_yields[-1]) else 0.03 # Fallback if NaN
    initial_beta1_first = (first_day_yields[0] - first_day_yields[-1]) if not np.isnan(first_day_yields[0]) and not np.isnan(first_day_yields[-1]) else 0.0 # Fallback
    last_successful_params = [initial_beta0_first, initial_beta1_first, 0.0, 1.5]
else:
    st.write("Error: yield_curve_data is empty. Cannot proceed with dynamic calibration.")
    # Exit or handle error appropriately
    exit()

# Iterate through historical yield data
# Using tqdm for progress bar (optional, remove if tqdm not available)
for date, daily_yields_row in tqdm(yield_curve_data.iterrows(), total=yield_curve_data.shape[0]):
    daily_yields = daily_yields_row.values

    # Use previous day's successful parameters as the initial guess for the current day
    # If the previous day failed, keep using the last known good parameters
    current_initial_guess = last_successful_params

    # Perform calibration for the current date
    optimized_params, success = calibrate_ns_single_date(
        daily_yields, market_maturities, current_initial_guess, bounds
    )

    # Store results
    parameter_history_ns.append(optimized_params)
    dates_processed.append(date)

    # Update the last successful parameters if current calibration succeeded
    if success:
        last_successful_params = optimized_params

# Convert parameter history to a DataFrame
param_cols = ['beta0', 'beta1', 'beta2', 'tau1']
params_df_ns = pd.DataFrame(parameter_history_ns, index=pd.DatetimeIndex(dates_processed), columns=param_cols)

st.write("\n--- Dynamic Calibration Complete ---")
st.write(f"Successfully calibrated {params_df_ns['beta0'].notna().sum()} out of {len(params_df_ns)} dates.")
st.write("\n--- Parameter History Sample (Nelson-Siegel) ---")
st.write(params_df_ns.tail()) # Show the last few estimated parameters

st.subheader("The meaning of the different parameters")
st.write("Beta0: this reflects the long-term anchor of the yield curve, where on the long term, the yield curve should converge to this value. This shows the shifts in the overall level of the long-term interest rates, which is often related to long-term inflation expectations and long-run growth outlooks. Beta0 represents the long-term level of the yield curve; mathematically, it’s the value the curve converges to as maturity increases (i.e., as t→∞). Rising long-term yields mean that the investors expect higher inflation, therefore higher compensation.  If the long-term inflation expectations fall, then it can be due to weaker central bank policy and growth outlook.")
st.write("Rising Beta0: signals that markets anticipate higher long-term interest rates. Often reflects rising long-term inflation expectations. It can suggest investor optimism.")
st.write("**Falling Beta0:** This suggests that markets expect lower long-term rates. Typically being associated with lower long-term inflation expectations or pessimism about long-run growth. A falling **Beta0** is common during/after recessions when investors expect rates to stay low for a prolonged period.")
st.write("**Sharp move in Beta0:** without a corresponding move in shorter-term rates, this can indicate a shift in market sentiment about the economy's long-run direction rather than just short-term factors.")
st.write("Beta1: This is the short-term slope of the yield curve, where a negative value corresponds to an upward sloping curve at the short end. This is a proxy for monetary policy stance. This value is the difference between short- and long-term rates, where the short-term rate equals Beta0 + Beta1. A positive beta1 means that the short-term rates are greater than the long-term rates, therefore the yield curve is downward sloping.")
st.write("**Sharp drop in Beta1:** This can signal short-term panic, as the short-term yields decrease majorly, signaling a flight to safety.")
st.write("In general, an upward sloping yield curve means expected economic expansion, a downward sloping signals economic recession and a hump-shaped signals uncertainty or midterm premium.")
st.write("Beta2: this relates to the hump or trough in the middle maturities. This reflects shifts in the relative value of medium-term bonds compared to short and long-term ones. A positive Beta2 usually contributes to a hump, a negative beta2 usually contributes to a trough.")

########################################################################################################################################################################################################################
# --- Analyze and Visualize Parameter Time Series ---

fig6, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Plot Beta0 (Level)
axes[0].plot(params_df_ns.index, params_df_ns['beta0'], label='Beta0 (Level)', color='blue')
axes[0].set_ylabel('Beta0')
axes[0].set_title('Nelson-Siegel Parameter Evolution Over Time')
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].legend()

# Plot Beta1 (Slope)
axes[1].plot(params_df_ns.index, params_df_ns['beta1'], label='Beta1 (Slope)', color='green')
axes[1].set_ylabel('Beta1')
axes[1].grid(True, linestyle='--', alpha=0.6)
axes[1].legend()

# Plot Beta2 (Curvature)
axes[2].plot(params_df_ns.index, params_df_ns['beta2'], label='Beta2 (Curvature)', color='red')
axes[2].set_ylabel('Beta2')
axes[2].set_xlabel('Date')
axes[2].grid(True, linestyle='--', alpha=0.6)
axes[2].legend()

fig6.tight_layout()

st.subheader("Nelson-Siegel Parameter Evolution Over Time")
st.pyplot(fig6)

st.markdown("""
**Economic Interpretation:**
- **Beta0:** Roughly tracks the overall long-term level of interest rates.
- **Beta1:** Captures the slope of the yield curve (difference between short and long rates). Negative Beta1 indicates an upward sloping curve initially.
- **Beta2:** Relates to the curvature or 'hump'/'dip' in the middle of the curve.
""")

########################################################################################################################################################################################################################
# --- Animated Yield Curve Visualization for Streamlit ---

st.subheader("Yield Curve Animation (Nelson-Siegel Fitted Curve)")

# Animation step control
animation_step = st.slider("Animation step (days):", 1, 90, 30, 1)
frame_delay = st.slider("Frame delay (seconds):", 0.01, 0.5, 0.15, 0.01)

# Robust date selection as in your code
try:
    animation_subset_dates = params_df_ns.dropna().last('5Y').iloc[::animation_step]
except TypeError:
    five_years_ago = params_df_ns.index.max() - pd.DateOffset(years=5)
    animation_subset_dates = params_df_ns.dropna()[params_df_ns.index >= five_years_ago].iloc[::animation_step]

if not animation_subset_dates.empty:
    plot_placeholder = st.empty()
    min_yield_hist = yield_curve_data.min().min() - 0.005 if not yield_curve_data.empty else 0.0
    max_yield_hist = yield_curve_data.max().max() + 0.005 if not yield_curve_data.empty else 0.06
    maturity_labels_anim = [f'{m*12:.0f}M' if m < 1 else f'{m:.0f}Y' for m in market_maturities]

    for anim_date in animation_subset_dates.index:
        current_params = animation_subset_dates.loc[anim_date].values
        current_market_yields = yield_curve_data.loc[anim_date].values

        fig_anim, ax_anim = plt.subplots(figsize=(10, 6))
        ax_anim.scatter(market_maturities, current_market_yields, color='blue', label='Market Yields', zorder=5)
        if not np.isnan(current_params).any():
            fitted_yields = nelson_siegel(plot_maturities, *current_params)
            ax_anim.plot(plot_maturities, fitted_yields, color='red', label='Fitted NS Curve')
            anim_title = f'Yield Curve Evolution: {anim_date.strftime("%Y-%m-%d")}'
        else:
            fitted_yields = [np.nan] * len(plot_maturities)
            ax_anim.plot(plot_maturities, fitted_yields, color='red', label='Fitted NS Curve')
            anim_title = f'Yield Curve Evolution: {anim_date.strftime("%Y-%m-%d")} (Calibration Failed)'
        ax_anim.set_title(anim_title)
        ax_anim.set_xlabel('Maturity (Years)')
        ax_anim.set_ylabel('Yield (Decimal)')
        ax_anim.set_xticks(market_maturities)
        ax_anim.set_xticklabels(maturity_labels_anim, rotation=45)
        ax_anim.grid(True, linestyle='--', alpha=0.6)
        ax_anim.legend(loc='upper left')
        ax_anim.set_ylim(min_yield_hist, max_yield_hist)
        fig_anim.tight_layout()

        plot_placeholder.pyplot(fig_anim)
        plt.close(fig_anim)
        time.sleep(frame_delay)
else:
    st.info("Skipping animation: No valid calibrated parameters found in the selected subset for animation.")   
########################################################################################################################################################################################################################
