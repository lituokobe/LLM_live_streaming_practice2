# Deep Learning for Asset Price Movement Prediction

This project uses deep learning to predict the short-term movement of Adobe Inc.'s stock price. By training an LSTM model on five years of historical data (2020–2025), it forecasts whether the stock will rise or fall on the second trading day after a given input.

---

## 📌 Core Workflow

### 1. Data Preparation
- Analyzed Adobe's daily price movements over five years.
- Defined a reasonable threshold to classify price changes as "up" or "down".

### 2. Feature Engineering
- **Three categories of features**:
  - **Price-based**: volume, return rate, average price trends, etc.
  - **News-based**: sentiment scores from daily news articles about Adobe.
  - **Macro-based**: NASDAQ index features, Federal Reserve interest rates, etc.
- **Total features**: 105
- **Feature normalization**: Applied tailored normalization strategies based on distribution types (bounded, normal, skewed, long-tail, clustered).
- **Feature reduction**: Used SHAP, VIF, and K-means to address multicollinearity and reduced to 40 features.
- Split data into training, testing, and evaluation sets.

### 3. Model Construction & Selection
- Built multiple LSTM models with varying depths and dropout configurations.
- Trained models on training and testing sets.
- Used TensorBoard to monitor accuracy, AUC, and loss across epochs.
- Evaluated models on the evaluation set using metrics: accuracy, recall, F1-score, and AUC.
- Selected the best-performing model for backtesting.

### 4. Backtesting
- Applied the final model to five years of trading data.
- Compared model-driven strategy vs. buy-and-hold strategy across:
  - Per-share return
  - Rolling Sharpe ratio
  - Drawdown curve analysis

---

## 🧰 Tech Stack

- TensorFlow
- Keras
- TensorBoard
- LangChain

---

## 📊 Project Highlights

### Model Performance
- Final LSTM model achieved **30x per-share return** compared to buy-and-hold.
- Demonstrated strong resilience during market downturns and captured growth during uptrends.
- Maintained **positive rolling Sharpe ratio** most of the time, consistently outperforming buy-and-hold.
- Maximum drawdown stayed within **-20%**, far better than buy-and-hold’s **-60%**, and hovered near zero.

### Key Optimizations
- Used large language models to automatically analyze five years of daily news and extract structured sentiment indicators.
- Applied histogram, KDE, boxplot, and Q-Q plot to assess feature distributions and select appropriate normalization techniques.
- Visualized feature relationships and multicollinearity using SHAP plots, scatter plots, heatmaps, and elbow plots to guide feature reduction.
- Leveraged TensorBoard for detailed training logs and used confusion matrices and classification reports to evaluate model performance on the evaluation set.

---

## 🚀 Getting Started

This is still a work in progress, visit `Deep-Learning-Asset-Price-Prediction.ipynb` for the current workflow. Contributions and feedback are welcome!

