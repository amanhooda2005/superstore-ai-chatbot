import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing
import streamlit as st
import os
from openai import OpenAI  # updated import

# -------------------------------
# SuperstoreAgent Class
# -------------------------------
class SuperstoreAgent:
    def __init__(self, file_path):
        self.df = pd.read_excel(file_path)
        self.df['Order Date'] = pd.to_datetime(self.df['Order Date'])

    def category_summary(self):
        summary = self.df.groupby("Category").agg({
            "Sales": "sum",
            "Profit": "sum"
        })
        summary["Profit Ratio"] = summary["Profit"] / summary["Sales"]
        return summary

    def month_over_month_profit(self):
        self.df['Month'] = self.df['Order Date'].dt.to_period('M')
        return self.df.groupby('Month')['Profit'].sum()

    def season_wise_top_product(self):
        def season(month):
            return (
                'Winter' if month in [12, 1, 2] else
                'Spring' if month in [3, 4, 5] else
                'Summer' if month in [6, 7, 8] else
                'Autumn'
            )

        self.df['Season'] = self.df['Order Date'].dt.month.map(season)
        sales = self.df.groupby(['Season', 'Product Name'])['Sales'].sum().reset_index()
        return sales.loc[sales.groupby('Season')['Sales'].idxmax()]

    def forecast_all_products(self, min_months=12):
        forecasts = {}
        products = self.df['Product Name'].unique()
        for product in products:
            product_data = self.df[self.df['Product Name'] == product]
            monthly_sales = product_data.set_index('Order Date').resample('M')['Sales'].sum().fillna(0)
            if len(monthly_sales) < min_months:
                continue
            try:
                model = ExponentialSmoothing(monthly_sales, seasonal='add', seasonal_periods=12)
                fit = model.fit()
                forecast = fit.forecast(6)
                forecasts[product] = forecast
            except:
                continue
        return pd.DataFrame(forecasts)

    def count_unique_products(self):
        total = self.df['Product Name'].nunique()
        by_category = self.df.groupby('Category')['Product Name'].nunique()
        by_subcategory = self.df.groupby('Sub-Category')['Product Name'].nunique()
        return total, by_category, by_subcategory


# -------------------------------
# Streamlit App Interface
# -------------------------------
st.set_page_config(page_title="Superstore AI Chatbot", layout="wide")
st.title("🧠 Superstore AI Chatbot")

uploaded_file = st.file_uploader("Upload Superstore Excel File", type=["xlsx"])

if uploaded_file:
    agent = SuperstoreAgent(uploaded_file)

    option = st.selectbox("Choose a task", [
        "Category Summary",
        "Month-over-Month Profit",
        "Top Selling Products by Season",
        "Sales Forecast for All Products",
        "Count Unique Products"
    ])

    if option == "Category Summary":
        st.subheader("📊 Category-wise Sales, Profit & Profit Ratio")
        st.dataframe(agent.category_summary())

    elif option == "Month-over-Month Profit":
        st.subheader("📈 Month-over-Month Profit")
        st.line_chart(agent.month_over_month_profit())

    elif option == "Top Selling Products by Season":
        st.subheader("🌦️ Top Selling Product by Season")
        st.dataframe(agent.season_wise_top_product())

    elif option == "Sales Forecast for All Products":
        st.subheader("🔮 Sales Forecast for Next 6 Months")
        forecast_df = agent.forecast_all_products()
        st.dataframe(forecast_df)

    elif option == "Count Unique Products":
        st.subheader("🧮 Unique Product Counts")
        total, by_cat, by_sub = agent.count_unique_products()
        st.write(f"Total Unique Products: {total}")
        st.write("By Category")
        st.dataframe(by_cat)
        st.write("By Sub-Category")
        st.dataframe(by_sub)

    # -------------------------------
    # Natural Language Chatbot
    # -------------------------------
    st.markdown("---")
    st.header("💬 Ask a Question About the Data")

    user_question = st.text_input("Ask in natural language (e.g., What is the most profitable category?)")

    if user_question:
        openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

        if not openai_api_key:
            st.warning("⚠️ OpenAI API key not set. Add it to environment or Streamlit secrets.")
        else:
            client = OpenAI(api_key=openai_api_key)  # initialize client

            context = f"""
Dataset Columns: {', '.join(agent.df.columns)}
Sample Data:
{agent.df.head(3).to_csv(index=False)}
"""

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a data analyst for a Superstore dataset. Answer questions based on the uploaded data."},
                        {"role": "user", "content": context + "\n\nQuestion: " + user_question}
                    ]
                )
                answer = response.choices[0].message.content
                st.success(answer)
            except Exception as e:
                st.error(f"❌ Error from OpenAI: {str(e)}")
