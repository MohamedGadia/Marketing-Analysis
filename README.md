# Marketing Analysis and Dashboard Project

## Project Overview
This project aims to analyze marketing data using SQL and Python, with a Power BI dashboard to visualize key insights. It provides a comprehensive overview of customer behavior, product performance, and conversion rates. The combination of data extraction, sentiment analysis, and interactive visualizations enables businesses to make informed decisions.

---

## Files in This Repository

### 1. SQL_Analysis.sql
- **Description**: A SQL script to extract marketing and customer review data from the database.
- **Features**:
  - Retrieves detailed customer reviews, ratings, and product-related information.
  - Provides structured data for further analysis.

### 2. Py_Analysis.py
- **Description**: A Python script for sentiment analysis of customer reviews.
- **Features**:
  - Analyzes sentiment using the VADER sentiment analysis tool.
  - Categorizes reviews as Positive, Negative, Neutral, Mixed Positive, or Mixed Negative.
  - Saves the output as `dbo_customer_reviews_with_sentiment.csv` for visualization.

### 3. Marketing_Analysis.pbix
- **Description**: A Power BI file containing a marketing analysis dashboard.
- **Features**:
  - Interactive visualizations of marketing KPIs and customer insights.

---

## Power BI Dashboard

### 1. Marketing Overview Page
- **Description**: Summarizes overall marketing performance.
- **Features**:
  - Tracks metrics like customer engagement, product conversions, and sentiment trends.
  - Provides a high-level view of the dataset.
  ![Marketing Overview Page](https://github.com/MohamedGadia/Marketing-Analysis/blob/main/Marketing%20Dashboard-Power%20BI/Overview_page.png?raw=true)

### 2. Customer Reviews Details
- **Description**: Deep dive into customer reviews and sentiments.
- **Features**:
  - Displays customer review ratings and sentiment distribution.
  - Visualizes the relationship between ratings and sentiment categories.
  - Analyzes customer feedback trends across time and products.
    ![Customer Reviews Page](https://github.com/MohamedGadia/Marketing-Analysis/blob/main/Marketing%20Dashboard-Power%20BI/Customer_Reviews_Page.png?raw=true)

### 3. Conversion Page
- **Description**: Focuses on customer journey and product conversions.
- **Features**:
  - Shows conversion rates by product and action (view, click, drop-off, purchase).
  - Analyzes conversion trends by day of the week and month.
  - Highlights top-performing products based on conversion rates.
    ![Conversion Page](https://github.com/MohamedGadia/Marketing-Analysis/blob/main/Marketing%20Dashboard-Power%20BI/Conversion_page.png?raw=true)

### 4. Social Media Details (Future Addition)
- **Description**: Planned to analyze social media metrics and their impact on marketing performance.
  ![Social Media Page](https://github.com/MohamedGadia/Marketing-Analysis/blob/main/Marketing%20Dashboard-Power%20BI/Social_Media_Page.png?raw=true)

---

## How to Use

### Prerequisites
- SQL Server or an equivalent database with marketing data.
- Python environment with the following libraries installed:
  - `pandas`
  - `nltk`
  - `pyodbc`
- Power BI Desktop to open and view the `.pbix` file.

### Steps
1. Run `SQL_Analysis.sql` to extract marketing data.
2. Execute `Py_Analysis.py` to perform sentiment analysis.
3. Open `Marketing_Analysis.pbix` in Power BI Desktop to explore the interactive dashboard.

---

## Output Insights
- **Marketing Trends**: Understand which products drive engagement and conversions.
- **Customer Sentiments**: Analyze customer feedback to improve product and service offerings.
- **Conversion Analysis**: Optimize marketing strategies based on product and time-based conversion trends.

---

## Project Presentation
This project includes a professional presentation summarizing key insights and findings from the marketing analysis. The presentation highlights:
- An overview of marketing performance metrics.
- Detailed analysis of customer reviews, social media engagement, and conversion trends.
- Recommendations based on data-driven insights to optimize marketing strategies.

The presentation is structured to provide actionable insights and visually appealing dashboards for stakeholders.

---

## Contribution
Feel free to contribute by raising issues or submitting pull requests to enhance the project.

---

## Programmer
**Mohamed Ahmed Gadia**  
Email: [mohamedgadia00@gmail.com]  
LinkedIn: [Mohamed Gadia](https://www.linkedin.com/in/mohamedgadia) 
