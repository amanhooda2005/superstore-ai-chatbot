import streamlit as st
import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing
from openai import OpenAI
import warnings

warnings.filterwarnings("ignore")

# Title
st.set_page_config(page_title="Superstore Analysis Dashboard", layout="wide")
st.title("📊 Superstore Sales Dashboard with AI Insights")

# Load Data
st.sidebar.header("Upload Excel File")
file = st.sidebar.file_uploader("Choose Excel file", type=["xlsx"])

if file:
    df = pd.read_excel(file)

    # ----- Category Summary -----
    st.subheader("🗂️ Sales and Profit Summary by Category")
    category_summary = df.groupby("Category").agg({
        "Sales": "sum",
        "Profit": "sum"
    })
    category_summary["Profit Ratio"] = category_summary["Profit"] / category_summary["Sales"]
    st.dataframe(category_summary.style.format({"Profit Ratio": "{:.2%}"}))

    # ----- MoM Profit Change -----
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Month'] = df['Order Date'].dt.to_period('M')
    monthly_profit = df.groupby('Month')['Profit'].sum().reset_index()
    monthly_profit['Month'] = monthly_profit['Month'].astype(str)
    monthly_profit['MoM Profit Change (%)'] = monthly_profit['Profit'].pct_change() * 100
    monthly_profit['MoM Profit Change (%)'] = monthly_profit['MoM Profit Change (%)'].round(2)

    st.subheader("📈 Month-over-Month Profit Change (Tabular View)")
    st.dataframe(monthly_profit)

    # ----- Seasonal Best Sellers -----
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'

    df['Season'] = df['Order Date'].dt.month.apply(get_season)
    season_product_sales = df.groupby(['Season', 'Product Name'])['Sales'].sum().reset_index()
    max_selling_products = season_product_sales.loc[season_product_sales.groupby('Season')['Sales'].idxmax()]

    st.subheader("🌦️ Top Selling Product by Season")
    st.dataframe(max_selling_products)

    # ----- Forecasting -----
    st.subheader("📊 6-Month Forecast for Each Product (Tabular View)")

    forecast_dict = {}
    products = df['Product Name'].unique()

    for product in products:
        product_df = df[df['Product Name'] == product]
        monthly_sales = (
            product_df
            .set_index('Order Date')
            .resample('M')['Sales']
            .sum()
            .fillna(0)
        )
        if len(monthly_sales) < 12:
            continue
        try:
            model = ExponentialSmoothing(monthly_sales, seasonal='add', seasonal_periods=12)
            fit = model.fit()
            forecast = fit.forecast(6)
            forecast_dict[product] = forecast
        except:
            continue

    if forecast_dict:
        forecast_df = pd.DataFrame(forecast_dict)
        forecast_df.index.name = "Forecast Month"
        forecast_df = forecast_df.reset_index()
        st.dataframe(forecast_df)
    else:
        st.warning("Not enough data for forecasting.")

    # ----- OpenAI Integration -----
    st.subheader("🧠 Ask AI (via OpenRouter)")

    user_prompt = st.text_input("Enter a question for the AI (e.g., 'What is the meaning of life?')")

    if user_prompt:
        with st.spinner("Contacting AI model..."):
            try:
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key="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",  # Replace this!
                )

                completion = client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "https://superstore-ai-chatbot-bjqjfdgzdmwx7svsvqdvj2.streamlit.app/",  # Optional
                        "X-Title": "Superstore Dashboard",           # Optional
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
