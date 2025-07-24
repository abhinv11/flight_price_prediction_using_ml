# Flight Price Prediction Using Machine Learning

A comprehensive machine learning project for predicting flight prices using various regression algorithms and advanced feature engineering techniques.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview

This project aims to predict flight prices using machine learning techniques. The model analyzes various factors such as airline, source, destination, departure time, arrival time, duration, number of stops, and class to predict the most accurate flight price.

### Key Objectives:
- Build an accurate flight price prediction model
- Compare multiple machine learning algorithms
- Implement comprehensive data preprocessing and feature engineering
- Provide insights into factors affecting flight prices
- Create a production-ready model for price prediction

## 📊 Dataset

The dataset contains flight booking information with the following characteristics:
- **Training Data**: Contains historical flight data with prices
- **Test Data**: Contains flight data for prediction (without prices)
- **Features**: Multiple categorical and numerical features affecting flight pricing

### Key Features:
- `airline`: Name of the airline company
- `source`: Source city/airport
- `destination`: Destination city/airport
- `departure_time`: Departure time
- `arrival_time`: Arrival time
- `duration`: Flight duration in hours
- `stops`: Number of stops (non-stop, 1 stop, 2+ stops)
- `class`: Travel class (Economy, Business, etc.)
- `days_left`: Days left for departure from booking date
- `price`: Target variable (flight price in ₹)

## ✨ Features

### Data Analysis & Visualization
- 📈 Comprehensive Exploratory Data Analysis (EDA)
- 📊 Interactive visualizations for price distribution and patterns
- 🔍 Correlation analysis between features
- 📉 Outlier detection and analysis

### Data Preprocessing
- 🧹 Missing value imputation using KNN and mode strategies
- 🏷️ Label encoding for categorical variables
- 📏 Standard scaling for numerical features
- 🔄 Duplicate removal and data cleaning

### Machine Learning
- 🤖 Multiple regression algorithms comparison
- ⚙️ Hyperparameter tuning with GridSearchCV
- 📊 Cross-validation for model evaluation
- 📈 Performance metrics analysis (R², RMSE, MAE)

## 🚀 Installation

### Prerequisites
- Python 3.7+
- Jupyter Notebook or JupyterLab

### Required Libraries
```bash
pip install pandas numpy matplotlib seaborn
pip install scikit-learn
pip install warnings
```

### Clone the Repository
```bash
git clone https://github.com/abhinv11/flight_price_prediction_using_ml.git
cd flight_price_prediction_using_ml
```

## 💻 Usage

1. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook flight_prediction.ipynb
   ```

2. **Run All Cells**: Execute all cells in sequence to:
   - Load and analyze the data
   - Perform data preprocessing
   - Train multiple ML models
   - Generate predictions
   - Create submission file

3. **Output**: The notebook will generate:
   - Comprehensive data analysis reports
   - Model performance comparisons
   - `submission.csv` file with predictions

## 🔄 Machine Learning Pipeline

### 1. Data Loading & Exploration
- Load training and test datasets
- Analyze data types and structure
- Generate descriptive statistics

### 2. Data Preprocessing
```python
# Missing value handling
KNNImputer(n_neighbors=5)  # For numerical features
SimpleImputer(strategy='most_frequent')  # For categorical features

# Feature encoding
LabelEncoder()  # For categorical variables
StandardScaler()  # For numerical features
```

### 3. Feature Engineering
- Remove non-predictive columns (ID, flight numbers)
- Encode categorical variables
- Scale numerical features
- Handle outliers appropriately

### 4. Model Training & Evaluation
- Split data (80% training, 20% validation)
- Train multiple models
- Hyperparameter tuning for top performers
- Cross-validation and performance metrics

## 🤖 Models Implemented

| Model | Description | Key Parameters |
|-------|-------------|----------------|
| **Linear Regression** | Basic linear relationship modeling | - |
| **Ridge Regression** | L2 regularized linear regression | alpha |
| **Lasso Regression** | L1 regularized linear regression | alpha, max_iter |
| **Decision Tree** | Tree-based regression | max_depth, min_samples_split |
| **Random Forest** | Ensemble of decision trees | n_estimators, max_depth |
| **Extra Trees** | Extremely randomized trees | n_estimators, max_depth |
| **Gradient Boosting** | Sequential boosting algorithm | learning_rate, n_estimators |
| **Support Vector Regression** | SVM for regression | kernel, C |
| **K-Nearest Neighbors** | Distance-based prediction | n_neighbors |

## 📈 Results

### Model Performance Comparison
The project evaluates models using multiple metrics:
- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **Overfitting Analysis**: Training vs Validation performance

### Best Performing Model
Based on validation performance, the best model achieves:
- High R² score indicating good fit
- Low RMSE for accurate predictions
- Minimal overfitting

### Key Insights
1. **Price Distribution**: Right-skewed with most flights in lower price range
2. **Airlines**: Significant price variation across different carriers
3. **Stops**: More stops generally correlate with higher prices
4. **Class**: Business class commands premium pricing
5. **Booking Time**: Last-minute bookings cost significantly more
6. **Routes**: Certain city pairs have consistently higher prices

## 🛠️ Technologies Used

### Programming & Libraries
- **Python 3.x**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms
- **Matplotlib & Seaborn**: Data visualization

### Development Environment
- **Jupyter Notebook**: Interactive development
- **Git**: Version control
- **VS Code**: Code editing and debugging

### Machine Learning Techniques
- **Supervised Learning**: Regression algorithms
- **Cross-Validation**: Model evaluation
- **Hyperparameter Tuning**: GridSearchCV
- **Feature Engineering**: Preprocessing and scaling

## 📁 Project Structure

```
flight_price_prediction_using_ml/
│
├── flight_prediction.ipynb    # Main Jupyter notebook
├── README.md                  # Project documentation
├── submission.csv             # Generated predictions (after running)
│
├── data/                      # Dataset directory
│   ├── train.csv             # Training data
│   └── test.csv              # Test data
│
└── outputs/                   # Generated outputs
    └── visualizations/        # Charts and plots
```

## 🔮 Future Enhancements

- [ ] **Deep Learning Models**: Implement neural networks
- [ ] **Feature Selection**: Advanced feature selection techniques
- [ ] **Ensemble Methods**: Custom ensemble models
- [ ] **Real-time Prediction**: API development for live predictions
- [ ] **Web Interface**: User-friendly web application
- [ ] **Time Series Analysis**: Seasonal price trend analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Abhinav Kumar**
- GitHub: [@abhinv11](https://github.com/abhinv11)
- LinkedIn: [Connect with me](https://linkedin.com/in/abhinavjaswal001)

## 🙏 Acknowledgments

- Kaggle for providing the dataset
- Scikit-learn community for excellent documentation
- Open source contributors for the libraries used

---

⭐ **Star this repository if you found it helpful!**

📧 **Contact**: Feel free to reach out for questions or collaborations!
