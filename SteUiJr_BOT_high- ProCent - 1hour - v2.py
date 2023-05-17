
import pandas as pd
user=
key=
server="RoboForex-Pro"
timeframesss="1hour"
symbols = ["USDJPY", "GBPJPY", "EURUSD", "GBPUSD", "USDCHF", "USDCAD","AUDCAD","AUDJPY", "AUDUSD", "AUDNZD", "AUDCHF", "CHFJPY", "EURGBP", "EURCHF", "EURNZD", "EURCAD", "EURAUD", "GBPCHF", "GBPJPY", "CADCHF"] 
ea_magic=1111265
lote=0.03
risk=0.002
account_currency="EUR"
val_variacion=0.12
variacion_datos=[]
tabla=[]
price=None
diferencia_pts=9.0
diferencia_en_porcentj_simbolo =0.05



f=1
while f>0:
    print(symbols)
    print(timeframesss)

    from tradingview_ta import TA_Handler, Interval
    import time
    from datetime import datetime
    



    now = datetime.now()
    fecha = now.strftime("%d-%m-%y %H:%M:%S")
    lista = symbols
    strongBuy_list = []
    strongSell_list = []
    for i in range(0,len(lista),1):
        tesla = TA_Handler()
        tesla.set_symbol_as(lista[i])
        tesla.set_exchange_as_forex()
        tesla.set_screener_as_forex()
        tesla.set_interval_as(Interval.INTERVAL_1_HOUR)
        print(lista[i])
        try:
            print(tesla.get_analysis().summary)
        except Exception as e:
            print("No Data")
            continue
        if((tesla.get_analysis().summary)["RECOMMENDATION"])=="STRONG_BUY":
            print(f" Compar más fuerte {i}", fecha)
            strongBuy_list.append(lista[i])
        elif((tesla.get_analysis().summary)["RECOMMENDATION"])=="STRONG_SELL":
            print(f" Compar más fuerte {i}", fecha)
            strongSell_list.append(lista[i])
            
    print("*** STRONG BUY LIST ***")

    print(strongBuy_list)

    print("*** STRONG SELL LIST ***")

    print(strongSell_list)

    #time.sleep(1


    lista = strongBuy_list
    strongBuy_list2 = []
    strongSell_list2 = []
    for i in range(0,len(lista),1):
        tesla = TA_Handler()
        tesla.set_symbol_as(lista[i])
        tesla.set_exchange_as_forex()
        tesla.set_screener_as_forex()
        tesla.set_interval_as(Interval.INTERVAL_4_HOURS)
        print(lista[i])
        try:
            print(tesla.get_analysis().summary)
        except Exception as e:
            print("No Data")
            continue
        if((tesla.get_analysis().summary)["RECOMMENDATION"])=="STRONG_BUY":
            print(f" Compar más fuerte {i}", fecha)
            strongBuy_list2.append(lista[i])
        elif((tesla.get_analysis().summary)["RECOMMENDATION"])=="STRONG_SELL":
            print(f" Compar más fuerte {i}", fecha)
            strongSell_list2.append(lista[i])
            
    print("*** STRONG BUY LIST2 ***")

    print(strongBuy_list2)

    print("*** STRONG SELL LIST2 ***")

    print(strongSell_list2)

    
    lista = strongSell_list
    strongBuy_list3 = []
    strongSell_list3 = []
    for i in range(0,len(lista),1):
        tesla = TA_Handler()
        tesla.set_symbol_as(lista[i])
        tesla.set_exchange_as_forex()
        tesla.set_screener_as_forex()
        tesla.set_interval_as(Interval.INTERVAL_4_HOURS)
        print(lista[i])
        try:
            print(tesla.get_analysis().summary)
        except Exception as e:
            print("No Data")
            continue
        if((tesla.get_analysis().summary)["RECOMMENDATION"])=="STRONG_BUY":
            print(f" Compar más fuerte {i}", fecha)
            strongBuy_list3.append(lista[i])
        elif((tesla.get_analysis().summary)["RECOMMENDATION"])=="STRONG_SELL":
            print(f" Compar más fuerte {i}", fecha)
            strongSell_list3.append(lista[i])
            
    print("*** STRONG BUY LIST3 ***")

    print(strongBuy_list3)

    print("*** STRONG SELL LIST3 ***")

    print(strongSell_list3)

    #time.sleep(1





    for i in strongBuy_list2 + strongSell_list3:
        symbol=i

        import MetaTrader5 as mt5
                # Function to start Meta Trader 5 (MT5)
        def start_mt5(username, password, server):
                    # Ensure that all variables are the correct type
                    uname = int(user) # Username must be an int
                    pword = str(key) # Password must be a string
                    trading_server = str(server) # Server must be a string
                    filepath = str("C:/Program Files/RoboForex - MetaTrader 5/terminal64.exe") # Filepath must be a string

                    # Attempt to start MT5
                    if mt5.initialize(login=uname, password=pword, server=trading_server, path=filepath):
                        # Login to MT5
                        if mt5.login(login=uname, password=pword, server=trading_server):
                            return True
                        else:
                            print("Login Fail")
                            quit()
                            return PermissionError
                    else:
                        print("MT5 Initialization Failed")
                        quit()
                        return ConnectionAbortedError
                            
        start_mt5(user, key, server)
        

        def get_info(symbol):
            '''https://www.mql5.com/en/docs/integration/python_metatrader5/mt5symbolinfo_py
            '''
            # get symbol properties
            info=mt5.symbol_info(symbol)
            return info


        def open_trade_buy2(action, symbol, lot, tp, sl, deviation, ea_magic_number, comment):
            '''https://www.mql5.com/en/docs/integration/python_metatrader5/mt5ordersend_py
            '''
            # prepare the buy request structure
            symbol_info = get_info(symbol)
            #print("symbol info",symbol_info)
            #print("llegas aquí?(buy)")
            from forex_python.converter import CurrencyRates
            def get_pip_value(symbol, account_currency):
                symbol_1 = symbol[0:3]
                symbol_2 = symbol[3:6]
                c = CurrencyRates()
                return c.convert(symbol_2, account_currency, c.convert(symbol_1, symbol_2, 1))
            def calc_position_size(symbol,stopLoss):
                print("Calculating position size for: ", symbol)
                account = mt5.account_info()
                balance = float(account.balance)
                pip_value = get_pip_value(symbol, account_currency)
                lot_size = (float(balance) * (float(risk)/100)) / (pip_value * stopLoss)
                lot_size = round(lot_size, 2)
                return lot_size
            def pip_calc(open, close):
                if str(open).index('.') >= 3:  # JPY pair
                    multiplier = 0.01
                else:
                    multiplier = 0.0001

                pips = round(( open - close) / multiplier)
                print("pips",pips)
                return int(pips)
            def pip_calctp(open, closetp):
                if str(open).index('.') >= 3:  # JPY pair
                    multiplier = 0.01
                else:
                    multiplier = 0.0001

                pipstp = round(( closetp - open) / multiplier)
                print("pipstp",pipstp)
                return int(pipstp)
            #symbol_info_tick_dict = mt5.symbol_info_tick(symbol)._asdict()
            #for prop in symbol_info_tick_dict:
            #        #print("  {}={}".format(prop, symbol_info_tick_dict[prop]))
            
            if not mt5.initialize():
                #print("initialize() failed, error code =",mt5.last_error())
                
                quit()
            
            # display the last GBPUSD tick
            lasttick=mt5.symbol_info(symbol)
            #print("lasttick",lasttick)
            #print("dígitos", symbol_info.digits)
            tp=round(tp,symbol_info.digits)
            sl=round(sl,symbol_info.digits)
            

            
            if action == 'buy':
                trade_type = mt5.ORDER_TYPE_BUY
                ##print(trade_type)
                
                pricetp = lasttick.bid
                pricesl = lasttick.ask
                price = round(pricetp,symbol_info.digits)
                print("price mt5",price) 
                open_tp=pricetp
                open_sl=pricesl
                close_sl=pricesl-sl
                closetp=pricetp+tp
                pipStopLoss=pip_calc(open_sl,close_sl)
                pipTakeProfit=pip_calctp(open_tp,closetp)
                calc_position_size(symbol,pipTakeProfit)
                lot_size=calc_position_size(symbol,pipTakeProfit)
                print("lot_size",lot_size)
                
                pass
            elif action =='sell':
                trade_type = mt5.ORDER_TYPE_SELL
                price = lasttick.bid
                price = round(price,symbol_info.digits)
                
                pass
            else:
                pass
            #print("precio + delta",round(price + (tp),symbol_info.digits)),
            if tp+price>price :
                

                if comment == "5min":
                    el_tiempo = int(300)
                elif comment == "15min":
                    el_tiempo = int(900)
                elif comment == "30min":
                    el_tiempo = int(1800)
                elif comment == "1hour":
                    el_tiempo = int(3600)
                elif comment == "1min":
                    el_tiempo = int(57)
                elif comment == "2min":
                    el_tiempo = int(118)
                elif comment == "90min":
                    el_tiempo = int(5400)
                else:
                    pass

                point = mt5.symbol_info(symbol).point
                #point = mt5.symbol_info(symbol).point
                #print("el precio mt5 es:", price)
                buy_request = {
                    "action": mt5.TRADE_ACTION_DEAL ,
                    "symbol": symbol,
                    "volume": lot_size,
                    "type": trade_type,
                    "price": price,
                    "sl": price - abs(pipStopLoss)*point,
                    "tp": price + abs(pipTakeProfit)*point,
                    "deviation": deviation,
                    "magic": ea_magic_number,
                    "comment": comment,
                    "type_time": mt5.ORDER_TIME_GTC,

                    "type_filling": mt5.ORDER_FILLING_IOC ,
                }
                # send a trading request
                result = mt5.order_send(buy_request)   
                #print("¿orden enviada?")    
                print("resultado de la orden", result) 
                print("symbol_info_tick() failed, error code =",mt5.last_error())
            else:
                #print("no puede vender")
                result = "no posible comprar"
                buy_request = "no posible comprar nene"
                el_tiempo = "no"
            return result, buy_request, el_tiempo , price

        def open_trade_sell2(action, symbol, lot, tp,sl, deviation, ea_magic_number,comment):
            '''https://www.mql5.com/en/docs/integration/python_metatrader5/mt5ordersend_py
            '''
            # prepare the buy request structure
            symbol_info = get_info(symbol)
            #print("symbol info",symbol_info)
            from forex_python.converter import CurrencyRates
            def get_pip_value(symbol, account_currency):
                symbol_1 = symbol[0:3]
                symbol_2 = symbol[3:6]
                c = CurrencyRates()
                return c.convert(symbol_2, account_currency, c.convert(symbol_1, symbol_2, 1))
            def calc_position_size(symbol,stopLoss):
                print("Calculating position size for: ", symbol)
                account = mt5.account_info()
                balance = float(account.balance)
                pip_value = get_pip_value(symbol, account_currency)
                lot_size = (float(balance) * (float(risk)/100)) / (pip_value * stopLoss)
                lot_size = round(lot_size, 2)
                return lot_size
            def pip_calc(open, close):
                if str(open).index('.') >= 3:  # JPY pair
                    multiplier = 0.01
                else:
                    multiplier = 0.0001

                pips = round((close - open ) / multiplier)
                print("pips",pips)
                return int(pips)
            def pip_calctp(open, closetp):
                if str(open).index('.') >= 3:  # JPY pair
                    multiplier = 0.01
                else:
                    multiplier = 0.0001

                pipstp = round((open - closetp ) / multiplier)
                print("pipstp",pipstp)
                return int(pipstp)
            #symbol_info_tick_dict = mt5.symbol_info_tick(symbol)._asdict()
            #for prop in symbol_info_tick_dict:
            #        #print("  {}={}".format(prop, symbol_info_tick_dict[prop]))
            
            if not mt5.initialize():
                #print("initialize() failed, error code =",mt5.last_error())
                quit()
            # display the last GBPUSD tick
            lasttick=mt5.symbol_info(symbol)
            #print(lasttick)
            #print("dígitos", symbol_info.digits)
            tp=round(tp,symbol_info.digits)
            sl=round(sl,symbol_info.digits)

            if action == 'buy':
                trade_type = mt5.ORDER_TYPE_BUY
                price = lasttick.ask
                price = round(price,symbol_info.digits)
                pass
            elif action =='sell':
                trade_type = mt5.ORDER_TYPE_SELL
                
                pricetp = lasttick.ask
                pricesl = lasttick.bid
                price = round(pricetp,symbol_info.digits)
                print("price mt5",price) 
                open_tp=pricetp
                open_sl=pricesl
                close_sl=pricesl+sl
                closetp=pricetp-tp
                pipStopLoss=pip_calc(open_sl,close_sl)
                pipTakeProfit=pip_calctp(open_tp,closetp)
                calc_position_size(symbol,pipStopLoss)
                lot_size=calc_position_size(symbol,pipStopLoss)
                print("lot_size",lot_size)
                
                
                pass
            else:
                pass
            #point = mt5.symbol_info(symbol).point
            #print("precio + delta",round(price + (tp),symbol_info.digits)),
            if -tp + price < price :
                #print("el precio mt5 es:", price)
                

                if comment == "5min":
                    el_tiempo = int(300)
                elif comment == "15min":
                    el_tiempo = int(900)
                elif comment == "30min":
                    el_tiempo = int(1800)
                elif comment == "1hour":
                    el_tiempo = int(3600)
                elif comment == "1min":
                    el_tiempo = int(57)
                elif comment == "2min":
                    el_tiempo = int(118)
                elif comment == "90min":
                    el_tiempo = int(5400)
                else:
                    pass
                point = mt5.symbol_info(symbol).point
                buy_request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": lot_size,
                    "type": trade_type,
                    "price": price,
                    "sl": price + abs(pipStopLoss)*point,
                    "tp": price - abs(pipTakeProfit)*point,
                    "deviation": deviation,
                    "magic": ea_magic_number,
                    "comment": comment,
                    "type_time": mt5.ORDER_TIME_GTC , # good till cancelled
                    
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                # send a trading request
                result = mt5.order_send(buy_request)   
                #print("¿orden enviada?")    
                print("resultado de la orden", result) 
                print("symbol_info_tick() failed, error code =",mt5.last_error())
            else:
                #print("no puede vender")
                result = "no posible comprar"
                buy_request = "no posible comprar nene"
                el_tiempo = "no"
            return result, buy_request, el_tiempo , price
        


        
        #timeframesss= "5min" # input("Insert pair as BTCUSDT: ")# "XRPUSDT" #lo_escogido_es# input("Insert pair as BTCUSDT: ")#
        #print(symbol)
        #symbol = symbol[:-3]
        #symbol = symbol + str("-")
        #symbol = symbol + str("USD")
        symbol = symbol + str("=X")
        va_bien_lr_s = []
        va_bien_lr_final_s = []
        va_bien_mlp_s = []
        va_bien_mlp_final_s = []
        # #print("¿llegaste hasta aquí?")

        ###############################################################

        COMPRANDO = [None]
        VENDIENDO = [None]

        # EXTRACTING DATA

        from datetime import datetime
        import pandas as pd
        import requests
        import typing
        import time 
        import yahooquery
        from yahooquery import Ticker 



        # time.sleep(5)
        #account = int(68107197)

        #utc_from = datetime(2020, 1, 1)
        #utc_to = datetime(2040,1,1)#fecha_ahora23

        symbol = symbol
        df=[]
        #print(symbol)
        #print(timeframesss)

        # get historical market data interval='1d'

            
        if timeframesss == '1hour':###################################################################### h1
            tickers = Ticker(symbol, asynchronous=True)
            df = tickers.history(period='3mo', interval='1h')
        elif timeframesss == '1day':###################################################################### h1
            tickers = Ticker(symbol, asynchronous=True)
            df = tickers.history( interval='1d')
        elif timeframesss == '5min':###################################################################### 5m
            tickers = Ticker(symbol, asynchronous=True)
            df = tickers.history(period='1mo', interval='5m')
        elif timeframesss == '15min':###################################################################### 15m
            tickers = Ticker(symbol, asynchronous=True)
            df = tickers.history(period='1mo', interval='15m')
        elif timeframesss == '30min':###################################################################### 30m
            tickers = Ticker(symbol, asynchronous=True)
            df = tickers.history(period='1mo', interval='30m')
        elif timeframesss == '1min':###################################################################### 30m
            tickers = Ticker(symbol, asynchronous=True)
            df = tickers.history(period='1mo', interval='1m')
        elif timeframesss == '2min':###################################################################### 30m
            tickers = Ticker(symbol, asynchronous=True)
            df = tickers.history(period='1mo', interval='2m')
        elif timeframesss == '90min':###################################################################### 30m
            tickers = Ticker(symbol, asynchronous=True)
            df = tickers.history(period='1mo', interval='90m')
        else:
            pass
        ##print("llegué a pasar la selección")
        ##print("df",df)
        df23=pd.DataFrame(df)
        ##print("df df",df23)
        df2=df23.reset_index()
        print("nono index",df2)




        ##print("yata ...")
        ################################################################## getting data

        #symbol = symbol[:-4]
        #symbol = symbol + str("USD")
        symbol = symbol[:-2]
        #interval = "1min"
        #from_date = from_date  # int(datetime.strptime(from_date, '%Y-%m-%d').timestamp() * 1000)
        #toDate = to_date  # int(datetime.strptime(to_date, '%Y-%m-%d').timestamp() * 1000)
        limit = 1000
        data=[]
        data2 = df2
        df=[]
        #df = data#pd.DataFrame(data)
        #startDate = from_date
        #endDate = toDate
        #########################################
        #tickerSymbol = symbol

        import pandas as pd  # Library that manages dataframes
        import numpy as np


        from pathlib import Path
        import frozendict



        datas = [(Path(frozendict.__path__[0]) / 'VERSION', 'frozendict')]

        # Change the date column to a pandas date time column

        # Define string format
        date_change = '%Y-%m-%d'

        # Create a new date column from the index
        #df2['time'] = df2.index

        # Perform the date type change
        #df2['time'] = pd.to_datetime(df2['time'], format=date_change)

        # Create a variable that is the date column
        #Dates = df2['time']
        #data2.drop(["dividends"])
        data=pd.DataFrame(data2)
        data.rename(columns={'date':'time'},inplace=True)
        data.dropna(axis='index',inplace=True)
        datados=data.copy()
        precio_ultimo=data.iloc[-1,2]
        precio_ultimo_l=data.iloc[-1,3]
        #print("precio_ultimo",precio_ultimo)
        #print("datassssssssssssss",data)
        from ta import add_all_ta_features
        data = add_all_ta_features(data, "close", "high", "low", "open", "volume", fillna=True, vectorized=True)

        ################################################################################################################################################
        data_low = add_all_ta_features(datados, "close", "high", "low", "open", "volume", fillna=True, vectorized=True)
        ##print("df.columns", df.columns)
        ################################################################################################################################################
        # Library that does date factors
        import fastai.tabular
        from fastai.tabular import add_cyclic_datepart, add_datepart

        ##########################

        ######################

        # Define the date parts
        fastai.tabular.add_datepart(data, 'time', drop='True')

        # Ensure the correct format
        data['time'] = pd.to_datetime(data.index.values, format=date_change)

        # Add the date parts
        fastai.tabular.add_cyclic_datepart(data, 'time', drop='True')

        # Define key model parameters

        # Set days out to predict
        shifts = [1]

        # Set a training percentage
        train_pct = .75




        # Ensure column types are correct

        def CorrectColumnTypes(data):
            # Input: dataframe
            # ouptut: dataframe (with column types changed)

            # Numbers
            for col in data.columns[1:80]:
                data[col] = data[col].astype('float')

            for col in data.columns[-10:]:
                data[col] = data[col].astype('float')

            # Categories
            for col in data.columns[80:-10]:
                data[col] = data[col].astype('category')

            return data

            # Create the lags


        def CreateLags(data, lag_size):
            # inputs: dataframe , size of the lag (int)
            # ouptut: dataframe ( with extra lag column), shift size (int)

            # add lag
            shiftdays = lag_size
            shift = -shiftdays
            data['high_lag'] = data['high'].shift(shift)
            return data, shift


        # Split the testing and training data
        def SplitData(data, train_pct, shift):
            # inputs: dataframe , training_pct (float between 0 and 1), size of the lag (int)
            # ouptut: x train dataframe, y train data frame, x test dataframe, y test dataframe, train data frame, test dataframe

            train_pt = int(len(data) * train_pct)

            train = data.iloc[:train_pt, :]
            test = data.iloc[train_pt:, :]

            x_train = train.iloc[:shift, 1:-1]
            y_train = train['high_lag'][:shift]
            x_test = test.iloc[:shift, 1:-1]
            y_test = test['high'][:shift]

            return x_train, y_train, x_test, y_test, train, test

        from sklearn.metrics import mean_squared_error  # Install error metrics
        from sklearn.linear_model import LinearRegression  # Install linear regression model
        from sklearn.neural_network import MLPRegressor  # Install ANN model
        from sklearn.preprocessing import StandardScaler  # to scale for ann


        # Regreesion Function

        def LinearRegression_fnc(x_train, y_train, x_test, y_test):
            # inputs: x train data, y train data, x test data, y test data (all dataframe's)
            # output: the predicted values for the test data (list)

            lr = LinearRegression()
            lr.fit(x_train, y_train)
            lr_pred = lr.predict(x_test)
            lr_MSE = mean_squared_error(y_test, lr_pred)
            lr_R2 = lr.score(x_test, y_test)
            #print('Linear Regression R2: {}'.format(lr_R2))
            #print('Linear Regression MSE: {}'.format(lr_MSE))
            if lr_R2 >= 0.99 and lr_R2 <= 1.01:
                #print("Compra o vende este par por LR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ", symbol)
                va_bien_rl = 1
            else:
                va_bien_rl = -1
                pass
            return lr_pred, va_bien_rl, lr_R2, lr_MSE





        def CalcProfit(test_data, lr_pred, j):
            pd.set_option('mode.chained_assignment', None)
            test_data['pred'] = np.nan
            test_data['pred'].iloc[:-j] = lr_pred
            test_data['change'] = test_data['high_lag'] - test_data['high']
            test_data['change_pred'] = test_data['pred'] - test_data['high']
            test_data['MadeMoney'] = np.where(test_data['change_pred'] / test_data['change'] > 0, 1, -1)
            test_data['profit'] = np.abs(test['change']) * test_data['MadeMoney']
            profit_dolars = test['profit'].sum()
            #print('Would have made: $ ' + str(round(profit_dolars, 1)))
            profit_days = len(test_data[test_data['MadeMoney'] == 1])
            print('Percentage of good trading days: ' + str(round(profit_days / (len(test_data) - j), 2)))

            return test_data, profit_dolars, profit_days





        # Go through each shift....

        for j in shifts:
                                                                            
            #a_las = now
            #print("ultimo precio de ese par: ", precio_ultimo)
            #print(str(j) + ' days out:')
            #print('------------')
            data_lag, shift = CreateLags(data, j)
            data_lag = CorrectColumnTypes(data_lag)
            x_train, y_train, x_test, y_test, train, test = SplitData(data, train_pct, shift)
            

            # Linear Regression
            #print("Linear Regression")
            lr_pred, va_bien_lr, lr_R2, lr_MSE = LinearRegression_fnc(x_train, y_train, x_test, y_test)
            CalcProfit(test, lr_pred, j)
            #print(symbol)
            #print("vabien_lr", va_bien_lr)
            #print("lr_R2", lr_R2)
            #print("escoger valor mas pequeño", (precio_ultimo) / (lr_R2 * 100))
            #print("ultimo precio de ese par: ", precio_ultimo)

            #print("lr_pred: ", lr_pred[-1])
            #print("diferencia de predicciones", (lr_pred[-1]) - (lr_pred[-2]))
            #print("precio donde poner el TP: ", ((lr_pred[-1]) - (lr_pred[-2])) + precio_ultimo)
            TP_lr1 = ((lr_pred[-1]) - (lr_pred[-2])) + precio_ultimo
            SL_lr1 = precio_ultimo - 2 * ((lr_pred[-1]) - (lr_pred[-2]))
            diferencia= None
            diferencia = (lr_pred[-1]) - (lr_pred[-2])
            print("ultimo",lr_pred[-1])
            print("penultimo",lr_pred[-2])
            print("diferencia", diferencia)
            variacion = None
            variacion = abs(((precio_ultimo*100)/(precio_ultimo + abs(diferencia)))-100)
            #print("diferencia",diferencia)
            diferencia_en_porcentaje= variacion

            ###########################################################################################################################################
        # Library that does date factors
        import fastai.tabular
        from fastai.tabular import add_cyclic_datepart, add_datepart

        ##########################

        ######################

        # Define the date parts
        fastai.tabular.add_datepart(data_low, 'time', drop='True')

        # Ensure the correct format
        data_low['time'] = pd.to_datetime(data_low.index.values, format=date_change)

        # Add the date parts
        fastai.tabular.add_cyclic_datepart(data_low, 'time', drop='True')

        # Define key model parameters

        # Set days out to predict
        shifts_l = [1]

        # Set a training percentage
        train_pct_l = .75




        # Ensure column types are correct

        def CorrectColumnTypes(data_low):
            # Input: dataframe
            # ouptut: dataframe (with column types changed)

            # Numbers
            for col in data_low.columns[1:80]:
                data_low[col] = data_low[col].astype('float')

            for col in data_low.columns[-10:]:
                data_low[col] = data_low[col].astype('float')

            # Categories
            for col in data_low.columns[80:-10]:
                data_low[col] = data_low[col].astype('category')

            return data_low

            # Create the lags


        def CreateLags(data_low, lag_size_l):
            # inputs: dataframe , size of the lag (int)
            # ouptut: dataframe ( with extra lag column), shift size (int)

            # add lag
            shiftdays_l = lag_size_l
            shift_l = -shiftdays_l
            data_low['low_lag'] = data_low['low'].shift(shift_l)
            return data_low, shift_l


        # Split the testing and training data
        def SplitData(data_low, train_pct_l, shift_l):
            # inputs: dataframe , training_pct (float between 0 and 1), size of the lag (int)
            # ouptut: x train dataframe, y train data frame, x test dataframe, y test dataframe, train data frame, test dataframe

            train_pt_l = int(len(data_low) * train_pct_l)

            train_l = data_low.iloc[:train_pt_l, :]
            test_l = data_low.iloc[train_pt_l:, :]

            x_train_l = train_l.iloc[:shift_l, 1:-1]
            y_train_l = train_l['low_lag'][:shift_l]
            x_test_l = test_l.iloc[:shift_l, 1:-1]
            y_test_l = test_l['low'][:shift_l]

            return x_train_l, y_train_l, x_test_l, y_test_l, train_l, test_l

        from sklearn.metrics import mean_squared_error  # Install error metrics
        from sklearn.linear_model import LinearRegression  # Install linear regression model
        from sklearn.neural_network import MLPRegressor  # Install ANN model
        from sklearn.preprocessing import StandardScaler  # to scale for ann


        # Regreesion Function

        def LinearRegression_fnc(x_train_l, y_train_l, x_test_l, y_test_l):
            # inputs: x train data, y train data, x test data, y test data (all dataframe's)
            # output: the predicted values for the test data (list)

            lr_l = LinearRegression()
            lr_l.fit(x_train_l, y_train_l)
            lr_pred_l = lr_l.predict(x_test_l)
            lr_MSE_l = mean_squared_error(y_test_l, lr_pred_l)
            lr_R2_l = lr_l.score(x_test_l, y_test_l)
            #print('Linear Regression R2: {}'.format(lr_R2))
            #print('Linear Regression MSE: {}'.format(lr_MSE))
            if lr_R2_l >= 0.99 and lr_R2_l <= 1.01:
                #print("Compra o vende este par por LR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ", symbol)
                va_bien_rl_l = 1
            else:
                va_bien_rl_l = -1
                pass
            return lr_pred_l, va_bien_rl_l, lr_R2_l, lr_MSE_l





        def CalcProfit(test_data_l, lr_pred_l, j_l):
            pd.set_option('mode.chained_assignment', None)
            test_data_l['pred'] = np.nan
            test_data_l['pred'].iloc[:-j_l] = lr_pred_l
            test_data_l['change'] = test_data_l['low_lag'] - test_data_l['low']
            test_data_l['change_pred'] = test_data_l['pred'] - test_data_l['low']
            test_data_l['MadeMoney'] = np.where(test_data_l['change_pred'] / test_data_l['change'] > 0, 1, -1)
            test_data_l['profit'] = np.abs(test_data_l['change']) * test_data_l['MadeMoney']
            profit_dolars_l = test_data_l['profit'].sum()
            #print('Would have made: $ ' + str(round(profit_dolars, 1)))
            profit_days_l = len(test_data_l[test_data_l['MadeMoney'] == 1])
            print('Percentage of good trading days: ' + str(round(profit_days_l / (len(test_data_l) - j_l), 2)))

            return test_data_l, profit_dolars_l, profit_days_l





        # Go through each shift....

        for j_l in shifts_l:
                                                                            
            #a_las = now
            #print("ultimo precio de ese par: ", precio_ultimo)
            #print(str(j) + ' days out:')
            #print('------------')
            data_lag_l, shift_l = CreateLags(data_low, j)
            data_lag_l = CorrectColumnTypes(data_lag_l)
            x_train_l, y_train_l, x_test_l, y_test_l, train_l, test_l = SplitData(data_low, train_pct_l, shift_l)
            

            # Linear Regression
            #print("Linear Regression")
            lr_pred_l, va_bien_lr_l, lr_R2_l, lr_MSE_l = LinearRegression_fnc(x_train_l, y_train_l, x_test_l, y_test_l)
            CalcProfit(test_l, lr_pred_l, j_l)
            #print(symbol)
            #print("vabien_lr", va_bien_lr)
            #print("lr_R2", lr_R2)
            #print("escoger valor mas pequeño", (precio_ultimo) / (lr_R2 * 100))
            #print("ultimo precio de ese par: ", precio_ultimo)

            #print("lr_pred: ", lr_pred[-1])
            #print("diferencia de predicciones", (lr_pred[-1]) - (lr_pred[-2]))
            #print("precio donde poner el TP: ", ((lr_pred[-1]) - (lr_pred[-2])) + precio_ultimo)
            TP_lr1_l = ((lr_pred_l[-1]) - (lr_pred_l[-2])) + precio_ultimo_l
            SL_lr1_l = precio_ultimo_l - 2 * ((lr_pred_l[-1]) - (lr_pred_l[-2]))
            diferencia_l= None
            diferencia_l = (lr_pred_l[-1]) - (lr_pred_l[-2])
            print("ultimo",lr_pred_l[-1])
            print("penultimo",lr_pred_l[-2])
            print("diferencia", diferencia_l)
            variacion_l = None
            variacion_l = abs(((precio_ultimo_l*100)/(precio_ultimo_l + abs(diferencia_l)))-100)
            #print("diferencia",diferencia)
            diferencia_en_porcentaje_l= variacion_l





            ############################################################################################################################################

            
            
            compra_vende=None
            price=None
            if diferencia > 0 and abs(diferencia_en_porcentaje)>diferencia_en_porcentj_simbolo:
                #print("!!!!!!!!!!!!QUE SIII QUE COMPRESSS por RL!!!!!  ", symbol)
                compra_vende="buy"
                print(compra_vende)
                result, buy_request, el_tiempo, price=open_trade_buy2(action="buy", symbol=symbol, lot=lote, tp=abs((lr_pred[-1]) - (lr_pred[-2])), sl=abs((lr_pred_l[-1]) - (lr_pred_l[-2])), deviation=100, ea_magic_number=ea_magic, comment=timeframesss)
                time.sleep(0)
        

            elif diferencia < 0 and abs(diferencia_en_porcentaje)>diferencia_en_porcentj_simbolo:
                #print("!!!!!!!!!!!!QUE SIII QUE VEndassssss por RL!!!!!  ", symbol)
                compra_vende="sell"
                print(compra_vende)
                result, buy_request, el_tiempo, price=open_trade_sell2(action="sell", symbol=symbol, lot=lote, tp=abs((lr_pred[-1]) - (lr_pred[-2])), sl=abs((lr_pred_l[-1]) - (lr_pred_l[-2])), deviation=100, ea_magic_number=ea_magic, comment=timeframesss)
                time.sleep(0)
                pass
            else:
                pass
        else:
            pass
        
        print("variacion",variacion)
        print("price", price)
        """
        now=datetime.now()
        row=[now, variacion, diferencia, symbol, timeframesss, price, compra_vende]
        variacion_datos.append(row)
        
        
        tabla=pd.DataFrame(variacion_datos, columns=['Date','Variacion', 'Diferencia', 'Symbol', 'TimeFrame', 'Price', 'Compra o Vende'])
        #print(tabla)
        #tabla.to_csv('tabla_variacion_ethusd_15m.csv', index=False)
        """

    f=f+1