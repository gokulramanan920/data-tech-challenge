# Generate: Data Branch Tech Challenge
Congratulations on making it to the second stage of the application! Thank you for applying to Generate and choosing the Data Branch! 

In this step of the application process, you will be completing a take-home challenge and have an interview. In the interview, you will share your screen and walk us through the process you went through to complete the challenge. Be creative! Show us what you know!

**DO NOT** use generative AI (ChatGPT, Claude, etc.) in any part of your challenge. It is important to think outside the box and try different things! If you have any questions about any part of the challenge, don't hesitate to reach out to any one of us!

Please finish this challenge before the date of your interview. Please read through this document thoroughly to make sure you follow ALL instructions. Try your best, and we can't wait to meet you!


# Synopsis
You're been brought in as a Data Scientist on a project with AeroConnect, an international airline focused on optimizing its routes and expanding profitable city pairs. The client wants to know:   
    a) Which routes have the highest and lowest passenger traffic over time?  
    b) Are there any trends or growth patterns across different cities or regions?  
    c) Can we predict traffic to help with resource allocations(aircrafts, crew, etc.)?  


# Your Task  
**CSV:** https://docs.google.com/spreadsheets/d/106VMqDhav1rPpEhVJHYQqEW8yZ_kxml7/edit?usp=sharing&ouid=110539250374045439410&rtpof=true&sd=true
1. Understanding the Data  
   a) Identify the most and least trafficked routes  
   b) Analyze trends and/or geographical patterns  
   c) Create visualizations to demonstrate trends & patterns determined in part b  

2. Build a Model  
   a) Your model should predict passenger traffic for the next 6–12 months on at least 1 city pair  
   **NOTE:** Make sure to use proper coding practices (i.e. commenting, camelcase, etc.)!  

4. Evaluate your model  
   a) Explain your model choices — why did you choose the elements you did  
   b) Evaluate the model's performance & report the accuracy of the model  

5. Provide Recommendations  
   a) Which routes should AeroConnect invest more in or scale back from?  
   b) How can AeroConnect use this model going forward?


# Deliverables   
1. Cleaned (if needed) data from the given CSV
2. Code for the model
3. Visualizations from Task 1c)
4. Answers to questions 1a, 1b, 2a, 2b, 3a, and 3b in PDF format
5. README file describing your process AND including the link to your cleaned data


# Important Steps   
1. Fork the repo -- This creates a copy of the repo under your account  
     a) At the top-right corner of the repo page, click "fork"  
     b) Choose the Github account as the destination  
     c) Clone the forked repo: "git clone https://github.com/YOUR-USERNAME/data-tech-challenge.git"  
   ** This is the model you will be sharing during your interview **
2. **DO NOT PUSH YOUR UPDATED DATA DIRECTLY TO THE FORKED REPO**  
   Instead, upload it to google drive and include the link in your README file.

Good luck, and have fun with it!

# Communication
If you have ANY questions, please do not hesitate to reach out to any of the following:    
- Haley Martin (Director of Data) : martin.hal@northeastern.edu
- Sonal Gupta (Chief of Data) : gupta.sonal@northeastern.edu
- Nandeenee Singh (Chief of Data) : singh.nand@northeastern.edu
- Kaydence Lin (Project Lead of Data) : lin.kay@northeastern.edu
- Tanisha Joshi (Project Lead of Data) : joshi.tani@northeastern.edu
- Ben Marler (Tech Lead of Data) : marler.b@northeastern.edu
- Jerome Rodrigo (Tech Lead of Data) : rodrigo.j@northeastern.edu

---

## Solution: AeroConnect Route Analysis and Forecasting Predictions 

This repository includes:
- The cleaned CSV used for analysis
- Identify most/least trafficked routes and geographical patterns
- Produce static visualizations of those patterns/trends
- An interactive panel dashboard using plotly.go for exploration of those trends
- Train and evaluate a basic SARIMA forecasting model for Top Australian Ports to Singapore

### Deliverables produced
- Static visualizations: `src/Outputs/figures/*.png`
- Model code and evaluation: `src/forecast.py` (SARIMA)
- Panel Dashboard: `src/dashboard.py`
- Answers summary: `Outputs/evaluations/Tech CHallenge Questions.pdf`

### Answers to Task 
- See `Outputs/Evaluations/Tech Challenge Questions.pdf` for answers to question 1-3
- Model performance: Backtest on Jan 1989–Jul 1989; metrics reported in `Outputs/models/city_sin_metrics.json` and chart in `Outputs/figures/city_sin_forecast.png`.
- Recommendations: Included in my slideshow presentation and towards the bottom of the README.

---

## My Analysis Process and Methodology

### Initial Data Exploration and Visualization Strategy

My analytical approach began with **exploratory data analysis (EDA)** through static visualizations to establish baseline understanding of the dataset. I created several key visualizations:

1. **Top/Bottom Routes Analysis**: Bar charts showing the most and least trafficked routes by total passenger volume
2. **Geographical Distribution**: Pie charts displaying passenger share by continent and country
3. **Seasonal Patterns**: Line plots showing monthly seasonality across all routes
4. **Time Series Trends**: Overall passenger traffic evolution from 1985-1988

This initial static analysis revealed that **Sydney ↔ Auckland** was indeed the highest-volume route, but the visual patterns suggested inconsistent seasonal behavior.

### Interactive Dashboard Development for Deeper Insights

Recognizing the limitations of static visualizations, I developed an **interactive Panel dashboard** using plotly.go to enable real-time data interaction. This dashboard incorporated:

- **Dynamic filtering** by Australian port, foreign port, continent, and country
- **Real-time y-axis switching** between passengers_total, z-scores, freight, mail, load_balance_ratio
- **Interactive time series plots** with adjustable date ranges
- **Top/Bottom N route rankings** with configurable thresholds

The dashboard proved invaluable, allowing me to identify critical patterns that weren't apparent in static analysis. Specifically, I discovered that while Sydney ↔ Auckland had the highest total volume, routes to **Singapore exhibited remarkably consistent seasonal patterns**.

### Key Discovery: Seasonal Consistency in Singapore Routes

Through the interactive dashboard, I observed that **Singapore-bound routes** (Perth → Singapore, Sydney → Singapore, Melbourne → Singapore) displayed:
- **Consistent local maxima and minima** occurring in the same months across all four years
- **Stable trend lines** without significant volatility or irregular spikes
- **Predictable seasonal cycles** that repeated annually with high fidelity

This contrasted sharply with Sydney ↔ Auckland, which showed **irregular seasonal patterns** and **inconsistent month-to-month variations**.

### Model Selection and Technical Implementation

Based on these insights, I implemented the SARIMA ML Model:

#### SARIMA (Seasonal ARIMA) Model
**Parameter Selection Justification:**
- **order=(1,1,1)**: 
  - p=1: One autoregressive term to capture immediate past dependencies
  - d=1: First differencing to achieve stationarity and remove long-term trend
  - q=1: One moving average term to model error patterns
- **seasonal_order=(1,1,1,12)**:
  - P=1: One seasonal autoregressive term for yearly patterns
  - D=1: Seasonal differencing to remove seasonal trends
  - Q=1: One seasonal moving average term for seasonal errors
  - s=12: 12-month seasonal period for monthly data

**Why SARIMA for Singapore Routes:**
SARIMA excels at capturing **seasonal non-stationary time series** because it:
- **Automatically identifies seasonal patterns** through the seasonal components
- **Handles both trend and seasonality** through differencing
- **Learns seasonal structure** from historical data and applies it to future predictions

### Model Performance Results: Singapore Routes Domination

The analysis confirmed my hypothesis about seasonal consistency. Here are the results for the top three Singapore routes:

#### Perth → Singapore (Most Predictable)
```json
{
  "mae": 483.36,
  "rmse": 493.87,
  "mape": 2.18%
}
```
**Analysis**: This route achieved **near-perfect accuracy** due to extremely consistent seasonal patterns and stable business travel demand.

#### Melbourne → Singapore
```json
{
  "mae": 750.54,
  "rmse": 873.99,
  "mape": 4.01%
}
```
**Analysis**: Strong performance with **excellent seasonal predictability**, slightly higher MAPE due to larger passenger volumes.

#### Sydney → Singapore
```json
{
  "mae": 1359.99,
  "rmse": 1694.16,
  "mape": 4.60%
}
```
**Analysis**: **Very good accuracy** despite being the highest-volume Singapore route, demonstrating consistent seasonal behavior.

### Technical Insights

**Reasons for SARIMA excellent performance**:

1. **Seasonal Decomposition**: SARIMA automatically identifies and models the 12-month seasonal component
2. **Trend Handling**: The (1,1,1) parameters effectively remove both linear and seasonal trends
3. **Pattern Recognition**: Learns that specific months consistently peak (December) or decline (February)
4. **Forecast Stability**: Consistent seasonal patterns are reliably captured into future periods

### Implications and Recommendations

Overall, my model reveals that routes with consistent seasonal patterns (like Singapore routes) achieve **2-5% MAPE accuracy**, while volatile routes (like Sydney ↔ Auckland) struggle with **20%+ MAPE**. This suggests:

1. **Invest heavily in predictable routes** (Singapore, other stable seasonal patterns)
2. **Use SARIMA for resource planning** on stable routes with near-perfect accuracy
3. **Scale back volatile routes** that are too risky to plan around
4. **Implement route predictability scoring** to guide investment/resourcing decisions including
 - pilots, cabin crew, ground staff, security guards, and gate allocations 
 - aircraft & maintenance scheduling
 - catering supply 
 - promotions/sales
 - baggage handling capacity

### Dataset Link

- See `src/Outputs/clean_data` on the Github Repo
- Or here: https://drive.google.com/drive/u/0/folders/1He7SFdFRxPD9aMT4FXXeJaSpEilcgg5r

