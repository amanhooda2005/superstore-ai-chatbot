import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing
import streamlit as st
import requests
import json

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
# Knowledge Base
# -------------------------------
def load_knowledge_base(file="knowledge_base.json"):
    try:
        with open(file, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

knowledge_base = load_knowledge_base()

def get_kb_answer(user_prompt):
    for qa in knowledge_base:
        if qa["question"].lower() in user_prompt.lower():
            return qa["answer"]
    return None


# -------------------------------
# Helper: Ask Gemini
# -------------------------------
def ask_gemini(prompt):
    try:
        GEMINI_API_KEY = st.secrets["gemini_api_key"]

        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": GEMINI_API_KEY
        }
        payload = {
            "contents": [
                {"parts": [{"text": prompt}]}
            ]
        }

        response = requests.post(url, headers=headers, data=json.dumps(payload))
        result = response.json()

        if "candidates" in result:
            return result["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return f"Unexpected response: {result}"

    except Exception as e:
        return f"Error from Gemini API: {e}"


# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="Superstore AI Chatbot", layout="wide")
st.title("Superstore AI Chatbot")

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
        st.subheader("Category-wise Sales, Profit & Profit Ratio")
        st.dataframe(agent.category_summary())

    elif option == "Month-over-Month Profit":
        st.subheader("Month-over-Month Profit")
        st.line_chart(agent.month_over_month_profit())

    elif option == "Top Selling Products by Season":
        st.subheader("Top Selling Product by Season")
        st.dataframe(agent.season_wise_top_product())

    elif option == "Sales Forecast for All Products":
        st.subheader("Sales Forecast for Next 6 Months")
        forecast_df = agent.forecast_all_products()
        st.dataframe(forecast_df)

    elif option == "Count Unique Products":
        st.subheader("Unique Product Counts")
        total, by_cat, by_sub = agent.count_unique_products()
        st.write(f"Total Unique Products: {total}")
        st.write("By Category")
        st.dataframe(by_cat)
        st.write("By Sub-Category")
        st.dataframe(by_sub)

    # -------------------------------
    # AI Chat Section
    # -------------------------------
    st.subheader("Ask AI")

    user_prompt = st.text_input("Ask a question (Excel-related or theory-related)")

    if user_prompt:
        # 1. Check Knowledge Base
        kb_answer = get_kb_answer(user_prompt)
        if kb_answer:
            st.success("📘 From Knowledge Base:")
            st.write(kb_answer)

        else:
            # 2. Try to interpret as Excel question
            df_answer = None
            explanation = ""

            try:
                if "most profitable category" in user_prompt.lower():
                    result = agent.df.groupby("Category")["Profit"].sum().idxmax()
                    value = agent.df.groupby("Category")["Profit"].sum().max()
                    df_answer = f"Most profitable category: {result} (Profit = {value:.2f})"
                    explanation = "Calculated using groupby(Category).sum() on Profit."

                elif "total sales" in user_prompt.lower():
                    value = agent.df["Sales"].sum()
                    df_answer = f"Total Sales = {value:.2f}"
                    explanation = "Calculated using df['Sales'].sum()."

                elif "total profit" in user_prompt.lower():
                    value = agent.df["Profit"].sum()
                    df_answer = f"Total Profit = {value:.2f}"
                    explanation = "Calculated using df['Profit'].sum()."

                elif "unique categories" in user_prompt.lower():
                    cats = agent.df["Category"].unique().tolist()
                    df_answer = f"Unique Categories: {', '.join(cats)}"
                    explanation = "Calculated using df['Category'].unique()."

                elif "top product" in user_prompt.lower():
                    top = agent.df.groupby("Product Name")["Sales"].sum().idxmax()
                    value = agent.df.groupby("Product Name")["Sales"].sum().max()
                    df_answer = f"Top-selling product: {top} (Sales = {value:.2f})"
                    explanation = "Calculated using groupby(Product Name).sum() on Sales."

            except Exception as e:
                df_answer = None
                explanation = f"Error while processing dataframe: {e}"

            # If DataFrame gave an answer
            if df_answer:
                st.success("📊 From Excel Data:")
                st.write(df_answer)

                # Ask Gemini to rephrase nicely
                context_prompt = f"""
The user asked: "{user_prompt}"
From the Excel file I calculated: {df_answer}
Method: {explanation}

Please rephrase this result in a clear, natural explanation.
                """
                ai_response = ask_gemini(context_prompt)
                st.info("🤖 Gemini Explanation:")
                st.write(ai_response)

            else:
                # 3. If not dataset-related, ask Gemini directly
                ai_response = ask_gemini(user_prompt)
                st.success("🤖 Gemini Response:")
                st.write(ai_response)

else:
    st.info("Upload an Excel file to get started.")
