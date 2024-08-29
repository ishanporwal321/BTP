import streamlit as st
import pandas as pd
from prophet import Prophet

def forecast(df):
    # Create a Prophet model
    m = Prophet(changepoint_prior_scale=0.01)

    # Fit the model to the data
    m.fit(df)

    # Make future predictions
    future = m.make_future_dataframe(periods=3000, freq='15T')
    forecast = m.predict(future)

    # Return the forecast DataFrame
    return forecast

def main():
    # Set Streamlit app title and description
    st.title('Time Series Forecasting App')
    st.write('Upload your CSV file to forecast data.')

    # File uploader for CSV
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Load the uploaded CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)

        # Display the uploaded data
        st.write('Uploaded Data:')
        st.dataframe(df)

        # Perform forecasting
        forecast_data = forecast(df)

        # Display the forecast output
        st.write('Forecast Data:')
        st.dataframe(forecast_data)

if __name__ == '__main__':
    main()
