from flask import Flask, render_template, request, jsonify
import plotly.express as px
import numpy as np
import pandas as pd
import shinybroker as sb

app = Flask(__name__)


# Assuming you define your function A here
def A(stock_symbol, long_volatility, short_volatility, rsi_upper, rsi_lower):
    historical_data_hourly = sb.fetch_historical_data(
        contract=sb.Contract({
            'symbol': stock_symbol,
            'secType': "STK",
            'exchange': "SMART",
            'currency': "USD"
        }),
        barSizeSetting="1 hour",
        durationStr="6 M",
    )

    #### Get daily data as well because it speeds up the code
    ####   writing process.
    historical_data_daily = sb.fetch_historical_data(
        contract=sb.Contract({
            'symbol': stock_symbol,
            'secType': "STK",
            'exchange': "SMART",
            'currency': "USD"
        }),
        barSizeSetting="1 day",
        durationStr="6 M"
    )
    historical_data_hourly = historical_data_hourly['hst_dta']
    historical_data_daily = historical_data_daily['hst_dta']
    #### Fetch your liquid trading hours for the asset
    #### You'll need this later!
    liquid_hours = sb.fetch_contract_details(
        contract=sb.Contract({
            'symbol': stock_symbol,
            'secType': "STK",
            'exchange': "SMART",
            'currency': "USD"
        })
    )
    # print(historical_data_hourly)
    # print(historical_data_daily)
    liquid_hours = liquid_hours['liquidHours'][0]

    def calc_trd_prd(dt):
        return [
            ts.isocalendar()[0] + ts.isocalendar()[1] / 100 for ts in dt
        ]

    historical_data_daily['trd_prd'] = calc_trd_prd(
        historical_data_daily['timestamp']
    )
    historical_data_hourly['trd_prd'] = calc_trd_prd(
        historical_data_hourly['timestamp']
    )
    liquid_hours['trd_prd'] = calc_trd_prd(liquid_hours.index)
    historical_data_daily = historical_data_daily[
                            historical_data_daily.index[
                                historical_data_daily.groupby(
                                    'trd_prd')['trd_prd'].transform('count') == 5
                                ].min():
                            ]

    hourly_price = historical_data_hourly['close']
    daily_price = historical_data_daily['close']
    hourly_log_return = np.log(hourly_price.shift(1) / hourly_price)
    daily_log_return = np.log(daily_price.shift(1) / daily_price)
    historical_data_hourly['log_return'] = hourly_log_return
    obs_vol = historical_data_hourly.groupby('trd_prd')['log_return'].std() * np.sqrt(32.5)
    vol_calcs = pd.DataFrame({'obs_vol': obs_vol})
    vol_calcs['exp_vol'] = vol_calcs['obs_vol'].shift(1)
    vol_calcs.index.name = 'trd_prd'

    blotter = pd.DataFrame(
        data={
            'entry_timestamp': None,
            'qty': 0,
            'exit_timestamp': None,
            'entry_price': None,
            'exit_price': None,
            'success': pd.NA
        },
        index=historical_data_daily['trd_prd'].unique()[1:]
    )

    ledger = pd.DataFrame({
        'date': historical_data_daily.loc[
            historical_data_daily['trd_prd'] != historical_data_daily[
                'trd_prd'].iloc[0],
            'timestamp'
        ],
        'position': 0,
        'cash': 0.0,
        'mark': historical_data_daily.loc[
            historical_data_daily['trd_prd'] != historical_data_daily[
                'trd_prd'].iloc[0],
            'close'
        ],
        'mkt_value': 0
    })

    all_trd_prd = sorted(blotter.index)
    first_week_trd_prd = historical_data_hourly['trd_prd'].unique()
    first_week_trd_prd.sort()
    first_week = first_week_trd_prd[0]

    filtered_trd_prd = [tp for tp in all_trd_prd if tp > first_week]

    for trd_prd in filtered_trd_prd:
        entry_timestamp = historical_data_hourly.loc[
            historical_data_hourly['trd_prd'] == trd_prd, 'timestamp'
        ].iloc[0]
        entry_timestamp = pd.to_datetime(entry_timestamp)
        entry_date = entry_timestamp.date()
        entry_price = historical_data_hourly.loc[
            historical_data_hourly['trd_prd'] == trd_prd, 'open'
        ].iloc[0]

        prev_trd_prd = blotter.index[blotter.index.get_loc(trd_prd) - 1] if blotter.index.get_loc(trd_prd) > 0 else None
        prev_close_price = None
        if prev_trd_prd is not None:
            prev_close_price = historical_data_daily.loc[
                historical_data_daily['trd_prd'] == prev_trd_prd, 'close'
            ].iloc[-1]

        expected_vol = vol_calcs.loc[trd_prd, 'exp_vol']
        if prev_close_price is not None and entry_price > prev_close_price:
            qty = -100
            exit_price_limit = entry_price * (1 - expected_vol)
        else:
            qty = 100
            exit_price_limit = entry_price * (expected_vol + 1)

        trade_data = historical_data_hourly[
            historical_data_hourly['trd_prd'] == trd_prd
            ].copy()
        trade_data = trade_data.sort_values('timestamp')

        success = False
        exit_price = None
        exit_timestamp = None

        exit_mask = (trade_data['high'] >= exit_price_limit) if qty > 0 else (trade_data['low'] <= exit_price_limit)
        if exit_mask.any():
            first_exit_row = trade_data.loc[exit_mask].iloc[0]
            success = True
            exit_price = exit_price_limit
            exit_timestamp = first_exit_row['timestamp']
        else:
            last_trade_row = trade_data.iloc[-1]
            exit_price = last_trade_row['close']
            exit_timestamp = last_trade_row['timestamp']

        if not success:
            week_data = historical_data_daily[historical_data_daily['trd_prd'] == trd_prd]
            last_trade_day = week_data['timestamp'].max()
            exit_timestamp = pd.to_datetime(last_trade_day).replace(hour=16, minute=0, second=0)

        exit_timestamp = pd.to_datetime(exit_timestamp)
        liquid_hours['end_time'] = pd.to_datetime(liquid_hours['end_time'].astype(str), errors='coerce')
        last_ts_this_week = liquid_hours.loc[liquid_hours['trd_prd'] == trd_prd, 'end_time'].max()
        most_recent_ts = trade_data['timestamp'].max()

        if pd.notna(last_ts_this_week) and pd.notna(exit_timestamp):
            if exit_timestamp > last_ts_this_week:
                exit_price = pd.NA
                exit_timestamp = pd.NA
                success = pd.NA

        blotter.loc[trd_prd] = [
            entry_timestamp,
            qty,
            exit_timestamp,
            entry_price,
            exit_price,
            success
        ]
        blotter.ffill(inplace=True)
        pd.set_option('future.no_silent_downcasting', True)
        if pd.notna(exit_timestamp):
            exit_date = pd.to_datetime(exit_timestamp).date()
        else:
            exit_date = None

        ledger['date'] = pd.to_datetime(ledger['date']).dt.date

        mask_buy = ledger['date'] >= entry_date
        ledger.loc[mask_buy, 'position'] += qty
        ledger.loc[mask_buy, 'cash'] -= qty * entry_price

        if pd.notna(exit_timestamp):
            exit_date = pd.to_datetime(exit_timestamp).date()
            mask_sell = ledger['date'] >= exit_date

            ledger.loc[mask_sell, 'position'] -= qty
            ledger.loc[mask_sell, 'cash'] += qty * exit_price

    ledger['mkt_value'] = ledger['position'] * ledger['mark'] + ledger['cash']
    ledger_copy = ledger.copy()

    historical_data_daily['log_return'] = np.log(
        historical_data_daily['close'].shift(-1) / historical_data_daily['close'])
    ledger['log_return'] = np.log(ledger['mkt_value'].shift(-1) / ledger['mkt_value'])
    ledger['log_return'] = ledger['log_return'].replace(np.nan, 0)

    log_return_daily_full = [None] * len(historical_data_daily['timestamp'])
    for i in range(len(historical_data_daily['timestamp']) - 1):
        log_return_daily_full[i] = (np.log(historical_data_daily['close'][i + 1] / historical_data_daily['close'][i]))
    historical_data_daily['log_return'] = log_return_daily_full
    ####
    RSI = []
    for i in range(len(blotter['entry_timestamp'])):
        dt = blotter['entry_timestamp'].iloc[i].date()
        rsi_data_index = historical_data_daily[historical_data_daily['timestamp'] == dt].index[0]
        rsi_data = historical_data_daily['log_return'].iloc[list(range(rsi_data_index - 14, rsi_data_index))]
        negative_numbers = [np.abs(x) for x in rsi_data if x < 0]
        positive_numbers = [np.abs(x) for x in rsi_data if x > 0]
        if not positive_numbers:
            RSI.append(0)
        elif not negative_numbers:
            RSI.append(100)
        else:
            avg_gain, avg_loss = sum(negative_numbers) / len(rsi_data), sum(positive_numbers) / len(rsi_data)
            rsi_value = 100 - (100 / (1 + (avg_gain / avg_loss)))
            RSI.append(rsi_value)

    RSI_Z_score = (RSI - np.mean(RSI)) / np.std(RSI)
    blotter['Relative_strength_index'] = RSI
    blotter['RSI_Z_score'] = RSI_Z_score

    blotter['entry_timestamp'] = pd.to_datetime(blotter['entry_timestamp'])
    blotter['timestamp'] = blotter['entry_timestamp'].dt.date

    historical_data_daily['EMA_12'] = historical_data_daily['close'].ewm(span=12, adjust=False).mean()
    historical_data_daily['EMA_26'] = historical_data_daily['close'].ewm(span=26, adjust=False).mean()
    historical_data_daily['MACD'] = historical_data_daily['EMA_12'] - historical_data_daily['EMA_26']
    historical_data_daily['signal_line'] = historical_data_daily['MACD'].ewm(span=9, adjust=False).mean()
    historical_data_daily['MACD_exceed_signal_line'] = historical_data_daily['MACD'] > historical_data_daily[
        'signal_line']
    MACD_diff = historical_data_daily['MACD'] - historical_data_daily['signal_line']
    historical_data_daily['MACD_Z_score'] = (MACD_diff - np.mean(MACD_diff)) / np.std(MACD_diff)

    historical_data_daily_iv = sb.fetch_historical_data(
        contract=sb.Contract({
            'symbol': stock_symbol,
            'secType': 'STK',
            'exchange': 'SMART',
            'currency': 'USD'
        }),
        barSizeSetting='1 day',
        durationStr='1 Y',
        whatToShow='OPTION_IMPLIED_VOLATILITY'
    )['hst_dta']

    historical_data_daily_iv['trd_prd'] = historical_data_daily_iv['timestamp'].apply(
        lambda x: (int(x.isocalendar()[0]) + int(x.isocalendar()[1]) * 0.01)
    )
    historical_data_daily_iv['wap_exp'] = historical_data_daily_iv['wap'].shift(1)
    historical_data_daily_iv['timestamp'] = pd.to_datetime(historical_data_daily_iv['timestamp'])
    blotter['timestamp'] = pd.to_datetime(blotter['timestamp'])
    blotter = pd.merge(
        blotter,
        historical_data_daily_iv[['timestamp', 'wap_exp']],  # 只取需要的列
        on='timestamp',
        how='left'
    )
    """
    def fetch_yield_curve(YYYY):
        URL = 'https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=' + \
        'daily_treasury_yield_curve&field_tdr_date_value=' + str(YYYY)

        cmt_rates_page = requests.get(URL)

        soup = BeautifulSoup(cmt_rates_page.content, 'html.parser')

        table_html = soup.findAll('table', {'class': 'views-table'})

        df = pd.read_html(io.StringIO(str(table_html)))[0]
        df = df[['Date', '1 Mo', '2 Mo', '3 Mo', '4 Mo', '6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']]
        df.Date = pd.to_datetime(df.Date)
        return df

    yield_curve = None

    for year in range(2024, 2026):
        df = fetch_yield_curve(year)
        yield_curve = pd.concat([yield_curve, df], ignore_index=True) if yield_curve is not None else df

    yield_curve['trd_prd'] = yield_curve['Date'].apply(lambda x: x.isocalendar()[0] + x.isocalendar()[1]/100)

    # Calculate 10Y-2Y Spread and its 10-day SMA
    yield_curve['yield_spread'] = yield_curve['10 Yr'] - yield_curve['2 Yr']
    yield_curve['yield_spread_SMA10'] = yield_curve['yield_spread'].rolling(window=10).mean()
    yield_curve['yield_curve_trend'] = yield_curve['yield_spread'] > yield_curve['yield_spread_SMA10']  # Steepening trend
    yield_diff = yield_curve['yield_spread'] - yield_curve['yield_spread_SMA10']
    yield_curve['yield_Z_score'] = np.mean(yield_diff)/np.std(yield_diff)

    yield_curve['Date'] = pd.to_datetime(yield_curve['Date'])
    """
    yield_curve = pd.read_pickle('yield_curve.pkl')
    blotter['timestamp'] = pd.to_datetime(blotter['timestamp'])
    blotter = pd.merge(
        blotter,
        yield_curve[['Date', 'yield_spread', 'yield_spread_SMA10', 'yield_curve_trend', 'yield_Z_score']],  # 只取需要的列
        left_on='timestamp',
        right_on='Date',
        how='left'
    )

    historical_data_daily['timestamp'] = pd.to_datetime(historical_data_daily['timestamp'])
    blotter = pd.merge(
        blotter,
        historical_data_daily[
            ['timestamp', 'trd_prd', 'MACD', 'signal_line', 'MACD_exceed_signal_line', 'MACD_Z_score']],  # 只取需要的列
        on='timestamp',
        how='left'
    )
    blotter.set_index('trd_prd', inplace=True)

    ledger = pd.DataFrame({
        'date': historical_data_daily.loc[
            historical_data_daily['trd_prd'] != historical_data_daily[
                'trd_prd'].iloc[0],
            'timestamp'
        ],
        'position': 0,
        'cash': 0.0,
        'mark': historical_data_daily.loc[
            historical_data_daily['trd_prd'] != historical_data_daily[
                'trd_prd'].iloc[0],
            'close'
        ],
        'mkt_value': 0
    })

    rsi_low_threshold = rsi_lower
    rsi_high_threshold = rsi_upper

    stop_loss_pct_long = long_volatility
    stop_loss_pct_short = short_volatility

    for trd_prd in filtered_trd_prd:
        entry_timestamp = historical_data_hourly.loc[
            historical_data_hourly['trd_prd'] == trd_prd, 'timestamp'
        ].iloc[0]
        entry_timestamp = pd.to_datetime(entry_timestamp)
        entry_price = historical_data_hourly.loc[
            historical_data_hourly['trd_prd'] == trd_prd, 'open'
        ].iloc[0]

        # 获取上一周期的 trd_prd（如果有）
        prev_trd_prd = blotter.index[blotter.index.get_loc(trd_prd) - 1] if blotter.index.get_loc(trd_prd) > 0 else None
        prev_close_price = None
        if prev_trd_prd is not None:
            prev_close_price = historical_data_daily.loc[
                historical_data_daily['trd_prd'] == prev_trd_prd, 'close'
            ].iloc[-1]

        expected_vol = vol_calcs.loc[trd_prd, 'exp_vol']

        # --- 读取上一期的 RSI、MACD（可选） ---
        rsi_value = blotter.loc[prev_trd_prd, 'Relative_strength_index'] if prev_trd_prd is not None else None
        macd_value = blotter.loc[prev_trd_prd, 'MACD'] if prev_trd_prd is not None else None
        macd_signal = blotter.loc[prev_trd_prd, 'signal_line'] if prev_trd_prd is not None else None

        # --- 读取当前期的 Zscore 数据（用于加权打分） ---
        MACD_z = blotter.loc[trd_prd, 'MACD_Z_score']
        yield_z = blotter.loc[trd_prd, 'yield_Z_score']
        RSI_z = blotter.loc[trd_prd, 'RSI_Z_score']

        def decide_qty_and_exit(entry_price, prev_close_price, expected_vol,
                                rsi_value, macd_value, macd_signal,
                                MACD_z, yield_z, RSI_z):
            """
            利用加权得分计算默认多空信号，并返回 (qty, exit_price_limit, decision_info)
            decision_info 用于记录决策来源，便于调试。
            """
            # 计算加权得分，权重分别为 20%、50%、30%
            weighted_score = 0.2 * RSI_z + 0.5 * MACD_z + 0.3 * yield_z

            if weighted_score > 0:
                default_qty = 100
                default_exit_price_limit = entry_price * (1 + expected_vol)
                decision_info = f"weighted_score_long ({weighted_score:.2f})"
            else:
                default_qty = -100
                default_exit_price_limit = entry_price * (1 - expected_vol)
                decision_info = f"weighted_score_short ({weighted_score:.2f})"

            # 如果同时存在 RSI 和 MACD 原始指标，则可以选择进一步修正信号
            if pd.notna(rsi_value) and pd.notna(macd_value) and pd.notna(macd_signal):
                bullish = (rsi_value < rsi_low_threshold) and (macd_value > macd_signal)
                bearish = (rsi_value > rsi_high_threshold) and (macd_value < macd_signal)

                if bullish:
                    qty = 100
                    exit_price_limit = entry_price * (1 + expected_vol)
                    decision_info += " + RSI_MACD_bullish"
                elif bearish:
                    qty = -100
                    exit_price_limit = entry_price * (1 - expected_vol)
                    decision_info += " + RSI_MACD_bearish"
                else:
                    qty = default_qty
                    exit_price_limit = default_exit_price_limit
            else:
                qty = default_qty
                exit_price_limit = default_exit_price_limit

            return qty, exit_price_limit, decision_info

        # --- 调用辅助函数进行多空决策 ---
        qty, exit_price_limit, decision_info = decide_qty_and_exit(
            entry_price, prev_close_price, expected_vol,
            rsi_value, macd_value, macd_signal,
            MACD_z, yield_z, RSI_z
        )

        # --- 后续出场逻辑（加入 5% 实时止损） ---
        trade_data = historical_data_hourly[historical_data_hourly['trd_prd'] == trd_prd].copy()
        trade_data = trade_data.sort_values('timestamp')

        success = False
        exit_price = None
        exit_timestamp = None

        # 原先的止盈/离场条件
        exit_mask = (trade_data['high'] >= exit_price_limit) if qty > 0 else (trade_data['low'] <= exit_price_limit)

        # 新增：5% 止损逻辑（目前仅针对多头；空头可根据需要对称处理）
        if qty > 0:
            stop_loss_mask = (trade_data['low'] <= entry_price * (1 - stop_loss_pct_long))
        else:
            stop_loss_mask = pd.Series([False] * len(trade_data), index=trade_data.index)

        # 合并离场条件
        final_exit_mask = exit_mask | stop_loss_mask

        if final_exit_mask.any():
            first_exit_row = trade_data.loc[final_exit_mask].iloc[0]
            if stop_loss_mask.loc[first_exit_row.name]:
                success = False
                exit_price = entry_price * (1 - stop_loss_pct_long) if qty > 0 else entry_price * (
                            1 + stop_loss_pct_short)
            else:
                success = True
                exit_price = exit_price_limit
            exit_timestamp = first_exit_row['timestamp']
        else:
            last_trade_row = trade_data.iloc[-1]
            exit_price = last_trade_row['close']
            exit_timestamp = last_trade_row['timestamp']

        if not success:
            week_data = historical_data_daily[historical_data_daily['trd_prd'] == trd_prd]
            last_trade_day = week_data['timestamp'].max()
            exit_timestamp = pd.to_datetime(last_trade_day).replace(hour=16, minute=0, second=0)

        exit_timestamp = pd.to_datetime(exit_timestamp)
        liquid_hours['end_time'] = pd.to_datetime(liquid_hours['end_time'].astype(str), errors='coerce')
        last_ts_this_week = liquid_hours.loc[liquid_hours['trd_prd'] == trd_prd, 'end_time'].max()

        if pd.notna(last_ts_this_week) and pd.notna(exit_timestamp) and exit_timestamp > last_ts_this_week:
            exit_price = pd.NA
            exit_timestamp = pd.NA
            success = pd.NA

        blotter.loc[trd_prd, [
            'entry_timestamp', 'qty', 'exit_timestamp', 'entry_price', 'exit_price', 'success'
        ]] = [entry_timestamp, qty, exit_timestamp, entry_price, exit_price, success]
        blotter.loc[trd_prd, 'decision_info'] = decision_info

        blotter.ffill(inplace=True)
        pd.set_option('future.no_silent_downcasting', True)

        if pd.notna(exit_timestamp):
            exit_date = pd.to_datetime(exit_timestamp).date()
        else:
            exit_date = None

        ledger['date'] = pd.to_datetime(ledger['date']).dt.date
        mask_buy = ledger['date'] >= entry_timestamp.date()
        ledger.loc[mask_buy, 'position'] += qty
        ledger.loc[mask_buy, 'cash'] -= qty * entry_price

        if pd.notna(exit_timestamp):
            exit_date = pd.to_datetime(exit_timestamp).date()
            mask_sell = ledger['date'] >= exit_date
            ledger.loc[mask_sell, 'position'] -= qty
            ledger.loc[mask_sell, 'cash'] += qty * exit_price

    ledger['mkt_value'] = ledger['position'] * ledger['mark'] + ledger['cash']

    nav_fig = px.bar(
        ledger,
        x='date',
        y='mkt_value',
        color='position',
        title='End-of-Day Strategy Market Value: '
    ).update_layout(
        yaxis=dict(
            tickfont=dict(size=12),
            showgrid=False,
            title=dict(text="Market Value"),
            # titlefont=dict(size=15)
        ),
        xaxis=dict(
            tickfont=dict(size=12),
            title=dict(text="Date"),
            # titlefont=dict(size=15),
        ),
        yaxis_tickprefix='$',
        plot_bgcolor='gray'
    )

    ledger_comparison = pd.merge(ledger_copy[['date', 'mark', 'mkt_value']], ledger[['date', 'mkt_value']], on='date')
    ledger_comparison.columns = ['date', 'mark', 'mkt_value_old', 'mkt_value_new']

    fig = px.line(ledger_comparison, x='date', y=['mkt_value_old', 'mkt_value_new'], title='NAV Over Time')
    # Placeholder for your actual function A
    # Replace the following with your actual implementation

    df1 = blotter
    df2 = ledger

    return fig, df1, df2


# Route to display the homepage (index.html)
@app.route('/')
def index():
    return render_template('index.html')


# Route to process the form data and display the result
@app.route('/plot', methods=['POST'])
def plot():
    # Get input from the form
    stock_symbol = request.form['stock_symbol']
    long_volatility = float(request.form['long_volatility'])
    short_volatility = float(request.form['short_volatility'])
    rsi_upper = float(request.form['rsi_upper'])
    rsi_lower = float(request.form['rsi_lower'])

    # Call function A
    plot, blotter, ledger = A(stock_symbol, long_volatility, short_volatility, rsi_upper, rsi_lower)

    # Convert plot to HTML string to render in the template
    plot_html = plot.to_html(plot, full_html=False)
    blotter_html = blotter[['entry_timestamp', 'qty', 'exit_timestamp', 'entry_price', 'exit_price', 'success']].to_html(classes='table table-striped', index=False)
    ledger_html = ledger.to_html(classes='table table-striped', index=False)

    return render_template('index.html', plot_html=plot_html, blotter_html=blotter_html, ledger_html=ledger_html)


if __name__ == '__main__':
    app.run(debug=True)