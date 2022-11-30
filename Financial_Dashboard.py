# =============================================================================
# # -*- coding: utf-8 -*-
# """
# Created on Sun Oct 30 01:17:27 2022
# 
# @author: sgollapalli
# """
# =============================================================================

# =============================================================================
# Initialising
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import streamlit as st
import yfinance as yf
import yahoo_fin.stock_info as si
from numerize import numerize
import plotly.graph_objects as go

# =============================================================================
# Page Configuration
# =============================================================================

#setting the page configuration to wide
st.set_page_config(page_title='FP_Shashank_Gollapalli',layout="wide")
#st.write(':flag-in:')

# =============================================================================
# Page layout and Reading-in data and taking the input from the User
# =============================================================================

#adding a title
st.markdown("<h1 style='text-align: center; color: #430297;'>Financial! Dashboard</h1>", unsafe_allow_html=True)

#scraping Wikipedia for the S&P 500 quotes
ticker_init = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']

#creating a drop-down to select a quote
selected_symb = st.selectbox("Quote Lookup", ticker_init)

#creating an update button
update = st.button('Update')

#adding functionality to the update button
if update:
    st.experimental_rerun()
 
#using the selected quote to pull financial details using yfinance
tick = yf.Ticker(selected_symb)
earnings = tick.earnings_dates.reset_index()

#head of the page
#splitting into 2 columns
col1, col2, col3 = st.columns(3)

with col1:
    #populating the first column
    #header
    st.header(str(tick.info['longName']) + " (" +str(tick.info['symbol'])+ ")")
        
    #caption as seen in Yahoo! Finance
    st.caption("NasdaqGS - NasdaqGS Real Time Price. Currency in USD")
    
with col2:
    #populating the first column
    #displaying the company logo
    st.image(tick.info['logo_url'], caption='Company logo')

#populating the third column with metrics of current price    
col3.metric("Current Price", round(tick.info['currentPrice'],2), round((tick.info['currentPrice'] - tick.info['previousClose']),2))

#structuring the page into 5 tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Summary", "Chart", "Financials", "MonteCarlo Simulation", "Analysis"])
    

# =============================================================================
# Tab 1: Summary
# =============================================================================

#Populating the first tab
with tab1:
    #creating 3 columns within tab1 to replicate the layout of Yahoo Finance
    col1, col2, col3 = st.columns(3)

    #Populating the first column
    with col1:
        #creating a dictionary with the necessary key : value
        col1_data = {
                        'Previous Close' : [tick.info['previousClose']], 
                        'Open' : [tick.info['open']], 
                        'Bid' : [str(tick.info['bid']) + " x " +str(tick.info['bidSize'])],
                        'Ask' : [str(tick.info['ask']) + " x " +str(tick.info['askSize'])],
                        "Day's Range" : [str(tick.info['dayLow']) + " - " + str(tick.info['dayHigh'])], 
                        '52 Week Range' : [str(tick.info['fiftyTwoWeekLow']) + " - " + str(tick.info['fiftyTwoWeekHigh'])], 
                        'Volume' : [tick.info['volume']], 
                        'Avg. Volume' : [tick.info['averageVolume']]
                    }
        
        #converting the dictionary to dataframe
        sum1 = pd.DataFrame.from_dict(col1_data, orient='index')
        sum1 = sum1.rename(columns = {0 : 'Value'})
        
        #Displaying the created dataframe as a stremalit table
        st.table(sum1)
                                   
    #Populating the second column
    with col2:
        #creating a dictionary with the necessary key : value
        col2_data = {
                        'Market Cap' : [numerize.numerize(tick.info['marketCap'],2)], 
                        'Beta (5Y Monthly)': [round(tick.info['beta'],2)],
                        'PE Ratio (TTM)': [round(tick.info['trailingPE'],2)], 
                        'EPS (TTM)': [round(tick.info['trailingEps'],2)],
                        'Earnings Date:': [str(si.get_next_earnings_date(selected_symb).date())  + " - " + str(si.get_next_earnings_date(selected_symb).date() + timedelta(days=5))], 
                        "Forward Dividend & Yield" : [str(tick.info['dividendRate']) + " (" + str(tick.info['dividendYield'] if tick.info['dividendYield'] == None else tick.info['dividendYield']*100) + "%)"], 
                        "Ex-Dividend Date":[str("None" if tick.info['exDividendDate'] == None else datetime.utcfromtimestamp(tick.info['exDividendDate']).strftime('%Y-%m-%d'))],
                        '1y Target Est' : [tick.info['targetMeanPrice']]
                    }
        
        #converting the dictionary to dataframe
        sum2 = pd.DataFrame.from_dict(col2_data, orient='index')
        sum2 = sum2.rename(columns = {0 : 'Value'})
        
        #Displaying the created dataframe as a stremalit table
        st.table(sum2)
            
    #Populating the third column
    with col3:
        #radio for duration selection
        duration = st.radio("Choose the duration", ('30D', '3M', '6M', 'YTD', '1 Year', '3 Yrs', '5 Yrs', 'MAX'),  horizontal=True, label_visibility= 'collapsed')
       
        #if statement to assing the value to duration
        if duration == '30D':
            dur = '1Mo'
        elif duration == '3M':
            dur = '3Mo'
        elif duration == '6M':
            dur = '6Mo'
        elif duration == 'YTD':
            dur = 'YTD'
        elif duration == '1 Year':
            dur = '1Y'
        elif duration == '3 Yrs':
            dur = '3Y'
        elif duration == '5 Yrs':
            dur = '5Y'
        else:
            dur = 'MAX'
        
        #creating dataframe which pull ticker data with iput duration from the user
        stock_price = yf.Ticker(selected_symb).history(period=dur,interval="1d")
        chart_data = stock_price[['Close', 'Volume']]
        chart_data['Volume'] = chart_data['Volume']/100000
        
        #plotting the chart for the above created dataframe
        st.area_chart(chart_data)
       
   #Pulling data for company profile
   #header
    st.subheader("Company Profile")
    
    #structuring the page
    col1, col2, col3 = st.columns(3)
    
    
    with col1:
        #Populating column 1 with basic company profile
        #creating a dictionary with the necessary key : value
        sum3_data = {
                                'Name' : [tick.info['longName']],
                                'Address' : [tick.info['address1'] + ", " + str(tick.info['city']) + ", " + str(tick.info['state']) + ", " + str(tick.info['country']) + " - " + str(tick.info['zip'])],
                                'Phone' : [tick.info['phone']],
                                'Website' : [tick.info['website']],
                                'Industry' : [tick.info['industry']],
                                'Number of Employees' : [numerize.numerize(tick.info['fullTimeEmployees'])],
                                'Total Cash' : [numerize.numerize(tick.info['totalCash'])],                    
                                'Total Debt' : [numerize.numerize(tick.info['totalDebt'])],
                                'Total Revenue' : [numerize.numerize(tick.info['totalRevenue'])],
                                'Earnings before interest, taxes, depreciation, and amortization' : [numerize.numerize(tick.info['ebitda'])],
                                'Gross Profits' : [numerize.numerize(tick.info['grossProfits'])]
                            }
        
        #converting the dictionary to dataframe
        sum3 = pd.DataFrame.from_dict(sum3_data, orient='index')
        sum3 = sum3.rename(columns = {0 : 'Value'})
        
        #Displaying the created dataframe as a stremalit table
        st.table(sum3)
        
        
    with col2:
        #Populating column 2 with major shareholders
        #mini header
        st.write("Major Shareholders:")
        
        #renaming the columns in the dataframe
        temp = tick.major_holders.rename(columns = {0 : 'Percentage', 1 : "Holders' type"})
        
        #calling the dataframe in a streamlit static table
        st.table(temp)
        
    with col3:
        #Populating column 3 with insititutional shareholders                            
        st.write("Institutional Shareholders:", tick.institutional_holders)
    
    #displaying the long business summary
    #header
    st.subheader('Business Summary')
    
    #display
    tick.info['longBusinessSummary']
    
   
# =============================================================================
# Tab 2: Charts
# =============================================================================

with tab2:
    #dividing the tab into 3 columns 
    col1, col2, col3, col4, col5 = st.columns(5)
    
    #Populating the first column    
    with col1:
        #creating exapander for duration selection
        with st.expander('Duration'):
            duration = st.radio("Choose the duration", ('30D', '3M', '6M', 'YTD', '1 Year', '3 Yrs', '5 Yrs', 'MAX'), key='chart',  horizontal=True, label_visibility= 'collapsed')
       
        #if statement for action basis the selected duration
        if duration == '30D':
            dur = '1mo'
        elif duration == '3M':
            dur = '3mo'
        elif duration == '6M':
            dur = '6mo'
        elif duration == 'YTD':
            dur = 'YTD'
        elif duration == '1 Year':
            dur = '1Y'
        elif duration == '3 Yrs':
            dur = '3Y'
        elif duration == '5 Yrs':
            dur = '5Y'
        else:
            dur = 'MAX'
    
    #Populating the second column  
    with col2:
        #creating exapander for duration selection
        with st.expander('Start Date'):
            start = st.date_input('Start Date', label_visibility= 'collapsed')
            
    #Populating the third column        
    with col3:
        #creating exapander for duration selection
        with st.expander('End Date'):
            end = st.date_input('End Date', label_visibility= 'collapsed')
    
    #Populating the fourth column 
    with col4:
        #creating exapander for interval selection
        with st.expander('Interval'):
            interval = st.radio("Choose the interval", ('1 Day', '1 Month', '1 Year'),  horizontal=True, label_visibility= 'collapsed')
            
        #if statement for action basis the selected interval
        if interval == '1 Day':
            inter = '1d'
        else:
            inter = '1mo'
        #else:
            #inter = '1Y'
    
    #dataframes with quote data
    #dataframe for pre-selected duration and intervals
    stock_info = yf.Ticker(selected_symb).history(period=dur, interval=inter).reset_index()
    
    #dataframe for custom daterange duration
    stock_info_cust = yf.Ticker(selected_symb).history(start = start, end = end,interval=inter).reset_index()
    
    #code to calculate moving average  for the above dataframes
    stock_info['Moving_AVG'] = stock_info['Close'].rolling(window=50,min_periods=1).mean()
    stock_info_cust['Moving_AVG'] = stock_info_cust['Close'].rolling(window=50,min_periods=1).mean()
    
    #function to plot the chart with candlesticks        
    def CandleChart(df):
        """
        Input: Dataframe
        Output: Chart with candlesticks
        
        This fucntion takes the argument and uses plotly to plot the chart with close price, volume and moving averages of a given quote
        """
        fig = go.Figure()  
        
        #adding the trace for stock price
        fig.add_trace(go.Candlestick(x=df['Date'], 
                                     open = df['Open'], 
                                     high = df['High'], 
                                     low = df['Low'], 
                                     close = df['Close'], 
                                     name='Stock Price'))
        
        colors = ['green' if row['Open'] - row['Close'] >= 0 else 'red' for index, row in df.iterrows()]
        
        #adding the trace for Volume
        fig.add_trace(go.Bar(x=df['Date'], 
                             y= (df['Volume']/1000000), 
                             marker_color = colors, 
                             name = 'Volume'))
        
        #adding the trace for Moving average
        fig.add_trace(go.Scatter(x=df['Date'], 
                                 y = df['Moving_AVG'], 
                                 line = dict(color = 'blue', width = 1), 
                                 name = 'Moving Average',
                                 connectgaps=True))
        
        #updating the final chart
        fig.update_layout(title = {'text' : str(selected_symb) + ' Share Prices over ' + str(duration) + ' duration in the interval of ' + str(inter), 'y':0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor':'top'}, 
                          yaxis_title = 'Stock Price in USD per share', 
                          xaxis_rangeslider_visible = False)
        
        return st.plotly_chart(fig, use_container_width=True)
    
    #function to plot the chart with Lines
    def LineChart(df):
        """
        Input: Dataframe
        Output: Chart with line
        
        This function takes the argument and uses plotly to plot the chart with close price, volume and moving averages of a given quote
        """
        fig = go.Figure()  
        
        #adding the trace for stock price
        fig.add_trace(go.Scatter(x=df['Date'], 
                                 y = df['Open'], 
                                 line = dict(color = 'navy', width = 2), 
                                 name = 'Stock Price',
                                 connectgaps=True))  
        
        colors = ['green' if row['Open'] - row['Close'] >= 0 else 'red' for index, row in df.iterrows()]
        
        #adding the trace for Volume
        fig.add_trace(go.Bar(x=df['Date'], 
                             y= (df['Volume']/1000000), 
                             marker_color = colors, 
                             name = 'Volume'))
        
        #adding the trace for Moving average
        fig.add_trace(go.Scatter(x=df['Date'], 
                                 y = df['Moving_AVG'], 
                                 line = dict(color = 'blue', width = 1), 
                                 name = 'Moving Average',
                                 connectgaps=True))
        
        #updating the final chart
        fig.update_layout(title = {'text' : str(selected_symb) + ' Share Prices over ' + str(duration) + ' duration in the interval of ' + str(inter), 'y':0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor':'top'}, 
                          yaxis_title = 'Stock Price in USD per share', 
                          xaxis_rangeslider_visible = False)
        
        return st.plotly_chart(fig, use_container_width=True)
        
    #Populating the fifth column  
    with col5:
        #creating exapander for chart type selection
        with st.expander('Chart Type'):
            chart_type = st.radio('Chart Type', ('Line', 'Candle'), label_visibility= 'collapsed')
    
    #if statement to display charts basis the options selected
    if chart_type == 'Line':
        if start != end:
            LineChart(stock_info_cust)
        else:
            LineChart(stock_info)
    elif chart_type == 'Candle':
        if start != end:
            CandleChart(stock_info_cust)
        else:
            CandleChart(stock_info)
    else:
        pass 
        
              
# =============================================================================
# Tab 3: Financials
# =============================================================================   
 
with tab3:
    #dividing the tab into 2 columns 
    col1, col2 = st.columns(2)
    
    with col1:
        #populating the first column
        #restructing the first column into three tabs
        tab1, tab2, tab3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
        
    with col2:
        #populating the second column
        #creating radio buttons to select the time period to view the data
        time = st.radio("label", ('Annual', 'Quarterly'), label_visibility='collapsed', horizontal=True)
    
    #populating the tabs created in column1
    with tab1:
        #creating if statement to output the annual or quaterly report
        if time == 'Annual':
            financials = tick.financials.fillna('No Info')
            st.table(financials)
        else:
            financials = tick.quarterly_financials.fillna('No Info')
            st.table(financials)
        
    with tab2:
        #creating if statement to output the annual or quaterly report
        if time == 'Annual':
            balance = tick.balance_sheet.fillna('No Info')
            st.table(balance)
        else:
            balance = tick.quarterly_balance_sheet.fillna('No Info')
            st.table(balance)   
       
    with tab3:
        #creating if statement to output the annual or quaterly report
        if time == 'Annual':
            cash = tick.cashflow.fillna('No Info')
            st.table(cash)
        else:
            cash = tick.quarterly_cashflow.fillna('No Info')
            st.table(cash)
            
            
# =============================================================================
# Tab 4: Monte Carlo Simulations
# =============================================================================

with tab4:
    col1, col2 = st.columns(2)
    
    with col1:
        sim = st.selectbox('Number of simulations',(200,500,1000))
    with col2:
        time = st.selectbox('Number of Days from today',(30,60,90))
    
    close_price = stock_price.Close
    daily_return = close_price.pct_change()
    daily_volatility = np.std(daily_return)
    
    np.random.seed(123)
    simulations = sim
    time_horizone = time
    
    # Run the simulation
    simulation_df = pd.DataFrame()
    
    for i in range(simulations):
        
        # The list to store the next stock price
        next_price = []
        
        # Create the next stock price
        last_price = close_price[-1]
        
        for j in range(time_horizone):
            # Generate the random percentage change around the mean (0) and std (daily_volatility)
            future_return = np.random.normal(0, daily_volatility)
    
            # Generate the random future price
            future_price = last_price * (1 + future_return)
    
            # Save the price and go next
            next_price.append(future_price)
            last_price = future_price
        
        # Store the result of the simulation
        next_price_df = pd.Series(next_price).rename('sim' + str(i))
        simulation_df = pd.concat([simulation_df, next_price_df], axis=1)
        
    # Plot the simulation stock price in the future
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 5, forward=True)
    
    plt.plot(simulation_df)
    plt.title('Monte Carlo simulation for ' + str(selected_symb) + ' stock price in next ' + str(time)+' days')
    plt.xlabel('Day')
    plt.ylabel('Price')
    
    plt.axhline(y=close_price[-1], color='red')
    plt.legend(['Current stock price is: ' + str(np.round(close_price[-1], 2))])
    ax.get_legend().legendHandles[0].set_color('red')
    
    st.pyplot(fig, use_container_width=True)
            
    
# =============================================================================
# Tab 5: Analysis
# =============================================================================

with tab5:
    col1, col2, col3, col4 = st.columns(4)
    #forcasted target price for the selected quote    
    col1.metric("Target Low Price", round(tick.info['targetLowPrice'],2), round((tick.info['targetLowPrice'] - tick.info['currentPrice']),2))
    col2.metric("Target Median Price", round(tick.info['targetMedianPrice'],2), round((tick.info['targetMedianPrice'] - tick.info['targetLowPrice']),2))
    col3.metric("Target Mean Price", round(tick.info['targetMeanPrice'],2), round((tick.info['targetMeanPrice'] - tick.info['targetLowPrice']),2))
    col4.metric("Target High Price", round(tick.info['targetHighPrice'],2), round((tick.info['targetHighPrice'] - tick.info['targetLowPrice']),2))
    
    #recommended key for the selected quote (buy, sell or hold)
    reco = tick.info['recommendationKey']
    reco_upper = reco.upper()
    st.subheader('Analyst Recommendation: ' + str(reco_upper) + "!")
    
    #creating a download option for the user
    #creating dataframe
    analysis = tick.recommendations
    
    #converting dataframe to csv
    csv = analysis.to_csv().encode('utf-8')
    
    #download button to download as csv
    st.download_button(
        label="Download Broker Recommendations",
        data=csv,
        file_name='Analyst_Recommedation_' + str(selected_symb) + '.csv', mime='text/csv')
    
    col1, col2 = st.columns(2)
    with col1:
        st.write('Latest News for ' + str(tick.info['longName']))
        news = pd.DataFrame(tick.news) 
        temp1 = news[['title', 'link']]
        st.dataframe(temp1, use_container_width = True)
    
    with col2:
        st.write('Related tickers to ' + str(tick.info['longName']))
        temp2 = news['relatedTickers']
        st.dataframe(temp2, use_container_width = True)
    
    #fun elements for the page basis the recommendation key
    if reco == 'hold':
        st.snow()
    elif reco == 'buy':
        st.balloons()
    else:
        st.warning('Sell', icon="⚠️")
    
#footer for the entire app
st.write("Data source: Yahoo Finance. URL: https://finance.yahoo.com/")   

 
# =============================================================================
# #Refernces
# =============================================================================

#Financial Programming Class materials on IESEG Online
#https://pypi.org/project/yfinance/
#https://blog.streamlit.io/
#https://docs.streamlit.io/
#https://www.geeksforgeeks.org/python-numerize-library/
#https://docs.streamlit.io/library/api-reference/text/st.markdown
#https://plotly.com/python/candlestick-charts/
#https://plotly.com/python/line-charts/
#https://pandas.pydata.org/docs/reference/api/pandas.Series.rolling.html
#https://stackoverflow.com/questions/69575993/getting-stock-earnings-date-from-yahoo-finance
#https://docs.python.org/3/library/datetime.html
