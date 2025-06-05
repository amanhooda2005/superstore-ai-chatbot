import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing
import streamlit as st
from openai import OpenAI
import os

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
    st.subheader("🧠 Ask AI (via OpenRouter)")

    user_prompt = st.text_input("Enter a question for the AI (e.g., 'what is the most profitable category?')")

    if user_prompt:
        with st.spinner("Contacting AI model..."):
            try:
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key="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",  # Replace this with env or secret in prod
                )

                completion = client.chat.completions.create(
                    extra_headers={
                        "X-Title": "Superstore Dashboard",
                    },
                    extra_body={},
                    model="deepseek/deepseek-r1-0528-qwen3-8b:free",
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ]
                )

                st.success("AI Response:")
                st.write(completion.choices[0].message.content)

            except Exception as e:
                st.error(f"Error from OpenAI API: {e}")
else:
    st.info("👈 Upload an Excel file to get started.")
