import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing
import streamlit as st
import re
import openai

# Set your OpenAI API key here
#openai.api_key = st.secrets["openai_api_key"]  # Use Streamlit secrets for safety

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

    def render_custom_chart(self, keyword):
        filtered = self.df[self.df['Product Name'].str.contains(keyword, case=False, na=False)]
        if filtered.empty:
            return None
        trend = filtered.groupby(filtered['Order Date'].dt.to_period('M'))['Sales'].sum()
        trend.index = trend.index.to_timestamp()
        return trend

    def ask_question(self, question):
        context = """
        You are an AI assistant for a Superstore sales dashboard. You help analyze sales, profit, trends, forecasts,
        and product performance based on Excel data uploaded by the user.
        """
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": question}
            ]
        )
        return response.choices[0].message.content.strip()

def extract_keyword(text):
    match = re.search(r"sales(?: trend)?(?: of| for)?\s*(.+)", text, re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()

# Streamlit app interface
st.set_page_config(page_title="🧠 Superstore AI Chatbot", page_icon="📊")
st.title("🧠 Superstore AI Chatbot")

if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = st.file_uploader("Upload Superstore Excel File", type=["xlsx"])

if st.session_state.uploaded_file:
    agent = SuperstoreAgent(st.session_state.uploaded_file)

    option = st.selectbox("Choose a task", [
        "Category Summary",
        "Month-over-Month Profit",
        "Top Selling Products by Season",
        "Sales Forecast for All Products",
        "Count Unique Products",
        "Custom Sales Trend Chart",
        "Ask Chatbot"
    ])

    if option == "Category Summary":
        st.dataframe(agent.category_summary())

    elif option == "Month-over-Month Profit":
        st.dataframe(agent.month_over_month_profit())

    elif option == "Top Selling Products by Season":
        st.dataframe(agent.season_wise_top_product())

    elif option == "Sales Forecast for All Products":
        forecast_df = agent.forecast_all_products()
        st.write("Sales forecast for next 6 months")
        st.dataframe(forecast_df)

    elif option == "Count Unique Products":
        total, by_cat, by_sub = agent.count_unique_products()
        st.write(f"Total Unique Products: {total}")
        st.write("By Category")
        st.dataframe(by_cat)
        st.write("By Sub-Category")
        st.dataframe(by_sub)

    elif option == "Custom Sales Trend Chart":
        user_input = st.text_input("Ask a question like: Show sales for chairs")
        if user_input:
            keyword = extract_keyword(user_input)
            chart_data = agent.render_custom_chart(keyword)
            if chart_data is not None:
                st.line_chart(chart_data)
            else:
                st.warning("No sales data found for the given keyword.")

    elif option == "Ask Chatbot":
        user_question = st.text_area("Ask your question about the superstore data:")
        if user_question:
            response = agent.ask_question(user_question)
            st.markdown(response)
