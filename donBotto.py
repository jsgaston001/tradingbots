import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time




uname=
pword=
trading_server="RoboForex-Pro"
filepath="C:/Program Files/RoboForex - MetaTrader 5/terminal64.exe"
diferencia_en_porcentj_simbolo =0.06


mt5.initialize(login=uname, password=pword, server=trading_server, path=filepath)  # connect to MetaTrader 5

symbols = ['AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD', 'CADCHF', 'CADJPY', 'CHFJPY', 'GBPAUD', 'GBPCAD',
           'GBPCHF', 'GBPJPY', 'GBPNZD', 'GBPUSD', 'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 'EURJPY', 'EURNZD',
           'EURUSD', 'NZDCAD', 'NZDCHF', 'NZDJPY', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY']

prices = {}

for symbol in symbols:
    data3=[]
    data3 =pd.DataFrame( mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 100))
    print(data3)
    prices[symbol] = pd.DataFrame(data3)[["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]]
    print(prices[symbol])
    data2=[]
    data2=pd.DataFrame(prices[symbol])
    from ta import add_all_ta_features
    data = add_all_ta_features(data2, "open", "high", "low", "close", "tick_volume", fillna=True, vectorized=True)
    print("data de ta",data)
    print(data.columns.tolist())
    import fastai.tabular
    from fastai.tabular import add_cyclic_datepart, add_datepart

    date_change = '%Y-%m-%d'
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
        data['close_lag'] = data['close'].shift(shift)
        return data, shift


    # Split the testing and training data
    def SplitData(data, train_pct, shift):
        # inputs: dataframe , training_pct (float between 0 and 1), size of the lag (int)
        # ouptut: x train dataframe, y train data frame, x test dataframe, y test dataframe, train data frame, test dataframe

        train_pt = int(len(data) * train_pct)

        train = data.iloc[:train_pt, :]
        test = data.iloc[train_pt:, :]

        x_train = train.iloc[:shift, 1:-1]
        y_train = train['close_lag'][:shift]
        x_test = test.iloc[:shift, 1:-1]
        y_test = test['close'][:shift]

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
        test_data['change'] = test_data['close_lag'] - test_data['close']
        test_data['change_pred'] = test_data['pred'] - test_data['close']
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
        precio_ultimo = data2.iloc[-1,4]
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

        

        
        
        compra_vende=None
        price=None
        if diferencia > 0 and abs(diferencia_en_porcentaje)>diferencia_en_porcentj_simbolo:
            print("BUY  ", symbol)
            compra_vende="buy"
            print(compra_vende)
            #result, buy_request, el_tiempo, price=open_trade_buy2(action="buy", symbol=symbol, lot=lote, tp=abs((lr_pred[-1]) - (lr_pred[-2])), deviation=100, ea_magic_number=ea_magic, comment=timeframesss)
            time.sleep(0)
    

        elif diferencia < 0 and abs(diferencia_en_porcentaje)>diferencia_en_porcentj_simbolo:
            print(" SELL  ", symbol)
            compra_vende="sell"
            print(compra_vende)
            #result, buy_request, el_tiempo, price=open_trade_sell2(action="sell", symbol=symbol, lot=lote, tp=abs((lr_pred[-1]) - (lr_pred[-2])), deviation=100, ea_magic_number=ea_magic, comment=timeframesss)
            time.sleep(0)
            pass
        else:
            pass
    else:
        pass
    
    print("variacion",variacion)
    #print("price", price)