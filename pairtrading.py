#############################
########## IMPORTS ##########
#############################

import sys
import argparse

from collections import deque
from abc import ABC, abstractmethod
from enum import Enum
from typing import NamedTuple, List, Tuple, Dict, Optional

from itertools import chain, count

from tqdm import tqdm

import numpy as np
import pandas as pd

from hurst import compute_Hc

import statsmodels.api as sm

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib.lines import Line2D

import seaborn as sns

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

sns.set_theme()

from dash import Dash 
from dash import dcc, html
import plotly.graph_objs as go
import logging






####################################
########## DATA STUCTURES ##########
####################################


# Define enumerations for tickers, sides, and statuses to standardize usage across the trading system.

Ticker = Enum("Ticker", ["ESc1", "NQc1"])
"""
Ticker: Represents the specific futures contracts being traded.
- ESc1: S&P 500 futures contract.
- NQc1: NASDAQ 100 futures contract.
"""

Side = Enum("Side", ["BUY", "SELL"])
"""
Side: Indicates the action of the trade.
- BUY: Indicates a purchase order.
- SELL: Indicates a sell order.
"""

Status = Enum("Status", ["CREATED", "STANDING", "EXECUTED"])
"""
Status: Defines the current state of an order.
- CREATED: Order has been created but not yet processed by the trading engine.
- STANDING: Order has been processed by the trading engine but could not be filled (at least not entirely).
- EXECUTED: Order has been fully executed.
"""



# Define Order class as a NamedTuple for immutable order attributes with optional price initialization.

class Order(NamedTuple):
    """
    Represents a trading order with defined attributes, NamedTuple was used as it's immutable and a lot faster than dataclass

    Attributes:
    - id (int): Unique identifier for the order.
    - trade_id (int): Unique identifier for the trade corresponding to that order (in pair trading, one trade is at least two orders).
    - timestamp (pd.Timestamp): Timestamp when the order was created.
    - ticker (Ticker): Enum specifying the futures contract (e.g., ESc1, NQc1).
    - side (Side): Enum indicating the trade action (BUY or SELL).
    - quantity (int): Number of contracts to be traded.
    - status (Status): Enum indicating the current status of the order (default to CREATED).
    - price (Optional[float]): Execution price of the order. None if the price is not yet determined.

    The default status is set to CREATED, and the price can optionally be set at order creation.
    """
    id: int
    trade_id: int
    timestamp: pd.Timestamp
    ticker: Ticker
    side: Side
    quantity: int
    status: Status = Status.CREATED
    price: Optional[float] = None




################################################
########## BACKTESTING HELPER CLASSES ##########
################################################



class Position:
    """
    Represents a trading position for a specific ticker, managing executed orders and calculating net position size.

    The Position class tracks the state of a trader's position in a particular financial instrument, maintaining
    records of executed buy and sell orders. It supports operations to update the position with new orders, and to
    calculate the net size of the position (the difference between long and short orders).

    Attributes:
        _ticker (Ticker): The ticker (futures contract) this position is for.
        _executed_orders (list[Order]): List of executed orders that have affected this position.
        _open_long (list[Order]): List of open long orders.
        _open_short (list[Order]): List of open short orders.

    Methods:
        _update(new_order: Order): Updates the position with a new order.
        _offset(new_quantity: int, open_opposite: list[Order], mul: int): Offsets a new order against existing opposite orders.
        orders: Property that returns a list of all executed orders.
        size: Property that calculates and returns the net size of the position.

    The position is designed to prevent direct manipulation of its attributes, encapsulating the logic for order
    management and position calculation within its methods.
    """
    def __init__(self, ticker: Ticker):
        self._ticker: Ticker = ticker
        self._executed_orders = []
        self._open_long = []
        self._open_short = []

    def _update(self, new_order: Order):
        # addping new order to the executed ones
        self._executed_orders.append(new_order)
        # copy quantity (by value) to not modify the order in _executed_orders
        new_quantity = new_order.quantity
        if new_order.side == Side.BUY:
            # offset position and get remaining quantity
            remaining_quantity = self._offset(new_quantity, self._open_short, -1)
            # append remaining quantity if its none zero
            if remaining_quantity > 0:
                self._open_long.append(new_order._replace(quantity=remaining_quantity))
        else:
            # offset position and get remaining quantity
            remaining_quantity = self._offset(new_quantity, self._open_long, 1)
            # append remaining quantity if its none zero
            if remaining_quantity > 0:
                self._open_short.append(new_order._replace(quantity=remaining_quantity))
        
        
    def _offset(self, new_quantity, open_opposite, mul):
        # offset position while it's possible
        while open_opposite and new_quantity > 0:
            # get last order and quantity (copy by value)
            last_order = open_opposite.pop()
            last_quantity = last_order.quantity
            # if order in the opposite side is larger than new order, append order with remaining quantity to open list
            if last_quantity > new_quantity:
                last_quantity -= new_quantity
                new_quantity = 0
                open_opposite.append(last_order._replace(quantity=last_quantity))
            # otherwise, simply offset the new order's quantity
            else:
                new_quantity -= last_order.quantity
        return new_quantity

    @property
    def orders(self):
        return self._executed_orders
        
    @property
    def size(self):
        # gives the number of quantities of the position
        if self._open_long:
            return sum([o.quantity for o in self._open_long])
        elif self._open_short:
            return -sum([o.quantity for o in self._open_short])
        else:
            return 0
            
    def __repr__(self):
        return "Position(" + self._ticker.name + ", " + str(self.size) + ")"






class TradingEngine(ABC):
    """
    Abstract base class for a trading engine, managing standing orders and execution logic.

    This class provides a framework for implementing trading strategies and execution mechanisms. It manages
    a queue of standing orders and provides methods for loading market data, calculating net standing quantities,
    and executing trades.

    Attributes:
        standing_orders (deque): A deque to hold standing orders for execution.
    """

    def __init__(self):
        # initialize with an empty deque for holding standing orders
        self.standing_orders = deque([])

    def _load_data(self, data):
        # load market data into the engine
        self.data = data

    @property
    def standing_quantities(self):
        # calculate and return the net standing quantities for each ticker
        standing = {t: 0 for t in Ticker} 
        for orders in self.standing_orders:
            # ensure orders is a list for iteration
            if not isinstance(orders, list):
                orders = [orders]
            # aggregate quantities by ticker, adjusting for buy/sell side
            for o in orders:
                standing[o.ticker] += o.quantity * (1 if o.side == Side.BUY else -1)
        return standing

    @abstractmethod
    def execute(self, tick, cash, orders: List[Order]) -> Tuple[float, List[Order]]:
        """
        Abstract method to be implemented by subclasses for executing trades.

        Parameters:
            tick: The current market tick or time step.
            cash: The current cash balance available for trading.
            orders (List[Order]): A list of orders to be executed.

        Returns:
            Tuple[float, List[Order]]: A tuple containing the updated cash balance and a list of executed orders.
        """
        raise NotImplementedError







class Strategy(ABC):
    """
    Abstract base class for trading strategies.

    Provides a framework for the implementation of trading strategies, including methods for loading data,
    generating unique order IDs, precomputing strategy-specific values, and evaluating the strategy to generate
    orders based on current market conditions.

    Attributes:
        _id (int): Internal counter for generating unique order IDs.
    """

    def __init__(self):
        # initialize the trade and order ID counter
        self._trade_id_generator = count(1)
        self._order_id_generator = count(1)

    def _load_data(self, data):
        # load market data into the strategy for analysis
        self.data = data

    def gen_trade_id(self): 
        # generate a unique trade ID by incrementing the internal counter
        return next(self._trade_id_generator)
        
    def gen_order_id(self):
        # generate a unique order ID by incrementing the internal counter
        return next(self._order_id_generator)

    @abstractmethod
    def _precompute(self):
        """
        Abstract method for precomputing values necessary for the strategy.

        This method should be implemented by subclasses to perform any precomputation or initialization
        needed by the strategy before the backtest is run (improves runtime efficiency)
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, tick, cash, positions: Dict[Ticker, Position], engine: TradingEngine) -> Optional[Dict[Ticker, Order]]:
        """
        Abstract method to evaluate the strategy and generate orders.

        Parameters:
            tick: The current market tick or time step.
            cash: The current cash balance available for trading.
            positions (Dict[Ticker, Position]): The current positions held, indexed by ticker.
            engine (TradingEngine): The trading engine to be used for order execution.

        Returns:
            Optional[Dict[Ticker, Order]]: A dictionary of orders to be executed, indexed by ticker, or None if no action is required.
        """
        raise NotImplementedError







class BacktestEngine:
    """

    This backtesting engine integrates market data, trading strategies, and execution logic to simulate the performance of trading strategies over historical data. 
    It tracks cash balances, positions, orders, and evaluates strategy performance, providing insights into the strategy's potential effectiveness in live trading.

    Attributes:
        _data (DataFrame): Historical market data used for backtesting.
        _ticks (Index): Timestamps extracted from the market data for iteration during the backtest.
        _trading_engine (TradingEngine): The engine responsible for executing orders based on market conditions and strategy signals.
        _strategy (Strategy): The trading strategy to be backtested, encapsulating the logic for generating trade signals.
        _cash (float): Initial cash balance for the backtest.
        _hist_cash (Series): Historical record of cash balances throughout the backtesting period.
        _positions (Dict[Ticker, Position]): Current open positions with Position objects, indexed by ticker.
        _hist_positions (DataFrame): Historical record of position sizes throughout the backtesting period.
        _strategy_orders (dict): Orders generated by the strategy, indexed by ticker.
        _standing_orders (dict): Orders that have been placed but not yet executed or canceled in the trading engine.
        _marked_to_market_value (DataFrame): Historical record of the marked-to-market value of all positions.

    Methods:
        set_data(data): Sets the market data for the backtest.
        set_trading_engine(trading_engine): Sets the trading engine to be used for order execution.
        set_strategy(strategy): Sets the trading strategy to be backtested.
        run(): Executes the backtest over the provided market data.
        _trade(tick): Processes trading signals and executes orders for a single tick.
        _eval_open_positions(tick): Evaluates and updates the marked-to-market value of open positions.
        positions: Property that returns the current positions as a DataFrame.
        strategy_orders: Property that returns all strategy-generated orders as a DataFrame.
        orders: Property that returns all executed orders as a DataFrame.
        standing_orders: Property that returns the current standing orders.
        standing_orders_history: Property that returns the history of standing orders as a DataFrame.
        evaluate(): Evaluates the overall performance of the backtested strategy, generating performance metrics and visualizations.
        
    The engine is designed to be flexible, allowing for the integration of different data sources, trading strategies, and execution mechanisms. It provides a framework for rigorous testing of trading strategies in a simulated market environment.
    """
    def __init__(self, init_cash):
        # initialize all attributes
        self._data = None
        self._ticks = None

        self._trading_engine = None
        self._strategy = None
        
        self._cash = init_cash
        self._hist_cash = None
        
        self._positions = {t: Position(t) for t in Ticker}
        self._hist_positions = None

        self._strategy_orders = {t: [] for t in Ticker} 
        self._standing_orders = None
        self._marked_to_market_value = None
        
        

    def set_data(self, data):
        # update attributes from backtesting data
        self._data = data
        self._ticks = data.unstack("symbol").index
        self._hist_cash = pd.Series(index=self._ticks)
        self._hist_positions = pd.DataFrame(index=self._ticks, columns=[t.name for t in Ticker])
        self._standing_orders = {}
        self._marked_to_market_value = pd.DataFrame(index=self._ticks, columns=[t.name for t in Ticker])
        return self

    def set_trading_engine(self, trading_engine):
        # set trading engine
        if self._data is None:
            raise ValueError("set data with set_data method before setting trading engine")
        self._trading_engine = trading_engine
        # adding data to the engine
        self._trading_engine._load_data(self._data)
        return self

    def set_strategy(self, strategy):
        if self._data is None:
            raise ValueError("set data with set_data method before setting strategy")
        self._strategy = strategy
        # adding data to the strategy and precompute before running
        self._strategy._load_data(self._data)
        self._strategy._precompute()
        return self
        
    def run(self):
        # main loop: iterates over each tick, trades and evaluates
        for tick in tqdm(self._ticks):
            self._trade(tick)
            self._eval_open_positions(tick)
            
    def _trade(self, tick):
        # processes trading signals and executes orders for a single tick
        # getting orders from strategy
        orders = self._strategy.evaluate(tick, self._cash, self._positions, self._trading_engine) or {}

        # if some orders have been given
        if orders:
            # store strategies
            for t, order in orders.items():
                self._strategy_orders[t].append(order)
        # execute strategy orders, or standing orders
        self._cash, executed_orders = self._trading_engine.execute(tick, self._cash, orders)

        # store current cash
        self._hist_cash[tick] = self._cash

        # update positions from executed orders
        for order in executed_orders:
            self._positions[order.ticker]._update(order)
            
        # store standing orders
        self._standing_orders[tick] = self._trading_engine.standing_orders.copy()
            
    def _eval_open_positions(self, tick):
        # evaluates and updates the marked-to-market value of open positions
        for t, pos in self._positions.items():
            self._marked_to_market_value.loc[tick, t.name] = pos.size * self._data.loc[t.name, tick]["mid_price"]
            self._hist_positions.loc[tick][t.name] = pos.size
    
    @property
    def positions(self):
        return pd.DataFrame({t.name: pos.size for t, pos in self._positions.items()}, index=["position"]).copy()

    @property
    def strategy_orders(self):
        return pd.DataFrame(o._asdict() for o in chain.from_iterable([orders for orders in self._strategy_orders.values()])).set_index(["timestamp", "ticker"]).sort_index().copy()
        
    @property
    def orders(self):
        orders_df = pd.DataFrame(o._asdict() for o in chain.from_iterable([p.orders for p in self._positions.values()])).copy()
        if orders_df.empty:
            orders_df = pd.DataFrame(columns=["id", "timestamp", "ticker", "side", "quantity", "price"])
        else:
            orders_df['ticker'] = orders_df['ticker'].apply(lambda x: x.name)
        return orders_df.set_index(["timestamp", "ticker"]).sort_index()

    @property
    def standing_orders(self):
        return self._trading_engine.standing_orders

    @property
    def standing_orders_history(self):
        return (
            pd.concat([pd.DataFrame([o._asdict() for o in orders], index=[key]*len(orders)) for key, orders in self._standing_orders.items()])
            .set_index(["timestamp", "ticker"], append=True)
            .sort_index()
        ).copy()
    
    def evaluate(self):
        
        # checking that backtest has been done before evaluating
        if self._hist_cash.isna().all():
            raise ValueError("Please run backtest before evaluating it")
            
        # calculate market_to_market cummulative PNL
        self._cummulative_pnl = self._hist_cash + self._marked_to_market_value.sum(axis=1)
    
        # calculate backtest return
        backtest_return = (self._cummulative_pnl.iloc[-1] - self._cummulative_pnl.iloc[0]) / self._cummulative_pnl.iloc[0]
    
        # calculate the multiplier to get yearly metrics
        num_days = (self._cummulative_pnl.index[-1] - self._cummulative_pnl.index[0]).seconds / (7 * 3600)
        yearly_multiplier = 252 / num_days
    
        
        # calculate yearly return and yearly volatility
        returns = self._cummulative_pnl.pct_change()
        yearly_return = returns.mean() * (7 * 3600) * yearly_multiplier
        yearly_std = returns.std()  * np.sqrt((7 * 3600) * yearly_multiplier)
        
        # calculate sharpe ratio (no risk free rate)
        sharpe =  yearly_return / yearly_std
    
        # calculate max_drawdown
        max_drawdown = (self._cummulative_pnl.cummax() - self._cummulative_pnl).max()
    
        # copy orders and to calculate by trade statistics 
        orders = self.orders
        # convert side to multiplier (-1 or 1)
        orders["side_mul"] = np.where(orders["side"]==Side.BUY, 1, -1)
        # get cash flow of each order
        orders["cash_flow"] = -orders["side_mul"] * orders["quantity"] * orders["price"]
    
        # get absolute returns  by trade
        trade_absolute_returns = orders.groupby("trade_id")["cash_flow"].sum()
    
        # calculate slippage from mid price for each trade
        # I decided to set the mid price at the time of the trade signal to be the reference price
        mid_prices = self._data["mid_price"].copy().swaplevel(0,1).sort_index()
        mid_prices.index.names = ["ticker", "timestamp"]
        orders["curr_mid_price"] = mid_prices
        # set mid price from first entering or exiting the trade as the reference  
        orders["reference_price"] = orders.groupby(["ticker", "trade_id", "side_mul"])["curr_mid_price"].transform("first")
        orders["slippage"] = (orders["reference_price"] - orders["price"]) * orders["side_mul"] * orders["quantity"]
        
        slippage_by_trade = orders.groupby("trade_id")["slippage"].sum() 

        return {
            "pnl_series": self._cummulative_pnl, 
            "positions": self._hist_positions, 
            "total_return": backtest_return, 
            "yearly_return": yearly_return, 
            "yearly_std": yearly_std, 
            "sharpe_ratio": sharpe, 
            "max_drawdown": max_drawdown, 
            "trade_absolute_returns": trade_absolute_returns, 
            "slippage_by_trade": slippage_by_trade
        }



##########################################################
########## FUNCTIONS REQUIRED FOR USED STRATEGY ##########
##########################################################


def train_val_iter(data, step, window):
    """
    Generates training and validation intervals for time series data
    Adjusts for irregular time intervals between data points, ensuring consistent training window sizes
    For example: if window="4H", the first validation period of a given day will have as train period, the last 4 hours
    of data of the last trading day

    WARNING: this function only works with frequencies that are at most in hours. If you want a window of 3D, please use
    "XH" where X = 3 * number of trading hours per day

    Parameters:
    - data (pd.DataFrame): Time series data with datetime index
    - step (str): Size of validation set (the increase in step), must be a pandas-compatible frequency string 
    - window (str): Size of the training window as a pandas-compatible frequency string.

    Yields:
    - tuple(pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp): Each yield returns a tuple containing:
      - train_start (pd.Timestamp): Start timestamp of the training interval
      - train_end (pd.Timestamp): End timestamp of the training interval, just before validation starts
      - val_start (pd.Timestamp): Start timestamp of the validation interval
      - val_end (pd.Timestamp): End timestamp of the validation interval

    """
    
    # convert step and window to timedelta
    one_sec = pd.Timedelta('1s')
    step_delta = pd.Timedelta(step)
    window_delta = pd.Timedelta(window) 

    # calculate time jumps between data (end of day to start of next day for example)
    time_jumps = data.index.to_series().diff().fillna(one_sec)
    # keep only the jumps that are not one second (so not continuous data)
    corrections = (time_jumps - one_sec)

    # identify validation start times, depending on step size
    val_starts = data.index[data.resample(step)._grouper.bins[:-1]].unique()
    first_train_start = data.index[0]

    # iterate over validation starts to yield train and val intervals
    for val_start in val_starts:
        
        # set end of validation period and start of train interval
        val_end = val_start + step_delta
        train_end = val_start - one_sec

        # set naive train_start (by window size)
        old_train_start = train_end - window_delta + one_sec
        # get the correction (sum of sizes of time jumps that are not 1s)
        correction = corrections.loc[old_train_start+one_sec: train_end+one_sec].sum()
        # correct size
        train_start = val_start - correction - window_delta
        
        # while there remains some corrections with updated train_start, update again
        while correction := corrections.loc[train_start+one_sec: old_train_start].sum():
            old_train_start = train_start
            train_start = old_train_start - correction
            
        # skip if train start before data start
        if train_start < first_train_start:
            continue

        # yield the slices
        yield train_start, train_end, val_start, val_end




def no_leakage_ols_spread(mid_prices, step="1H", window=None):
    """
    Fits an OLS model iteratively without lookahead bias to compute the spread and hedge ratio.

    Parameters:
    - mid_prices (pd.DataFrame): Mid prices both for ESc1 and NQc1
    - step (str): Incremental step for fitting as pandas frequency string (e.g., '1H').
    - max_window (str or None): Size of the fitting window as pandas frequency string. None for full span.

    Returns:
    - hedge_ratio (pd.Series): Fitted hedge ratio over time.
    - spread (pd.Series): Spread with fitted hedge ratio on previous step.

    Fits an OLS model on slices of data defined by `step` and `max_window`, updating the hedge ratio and spread
    for subsequent slices to prevent lookahead bias.
    """
    # initialize outputs
    hedge_ratio = pd.Series(index=mid_prices.index)
    spread = pd.Series(index=mid_prices.index)
    # store variables for fitting
    X = mid_prices["ESc1"]
    Y = mid_prices["NQc1"]

    # iterate over each period defined by step
    for fit_start, fit_end, eval_start, eval_end in tqdm(train_val_iter(mid_prices, step=step, window=window)):
        # fit ols on fitting period
        ols = sm.OLS(Y.loc[fit_start:fit_end], X.loc[fit_start:fit_end])
        ols_res = ols.fit()
        # evaluate on evaluation period
        hedge_ratio.loc[eval_start:eval_end] = ols_res.params["ESc1"]
        spread.loc[eval_start:eval_end] = Y - X.loc[eval_start: eval_end] * ols_res.params["ESc1"]
    return hedge_ratio, spread




################################################
########## IMPLEMENTATION OF STRATEGY ##########
################################################




class EWMZscoreSymmetric(Strategy):
    """
    Implements a symmetric mean reversion strategy based on the z-score of the spread between two assets,
    adjusted for exponential weighted moving average (EWMA). It enters and exits trades based on specified z-score thresholds.

    This strategy calculates the spread between two assets using an iterative OLS regression, in order to avoid leakage, then applies
    an EWMA to the spread to calculate z-scores. Trades are initiated when the z-score crosses predefined entry thresholds
    and are exited when the z-score reverts back within an exit threshold. Position sizing varies based on the z-score
    magnitude, allowing for dynamic allocation based on signal strength.
    """
    def __init__(self, z_enter, z_max, z_exit, min_sizing, max_sizing, ewm_halflife, ewm_burnout, ols_step="15T", ols_window="3H"):
        # initialize strategy 
        super().__init__()
        
        # zscore thresholds for trade entry and exit
        self._z_enter = z_enter
        self._z_max = z_max
        self._z_exit = z_exit
        
        # position sizing parameters
        self._max_sizing = max_sizing
        self._sizing_slope = (max_sizing - min_sizing) / (z_max - z_enter)
        self._sizing_intercept = min_sizing

        # ewm parameters for zscore
        self._ewm_halflife = ewm_halflife
        self._ewm_burnout = ewm_burnout
        
        # OLS regression parameters for hedge ratio calculation
        self._ols_step = ols_step
        self._ols_window = ols_window

        self._current_spread_side = 0
        

    def _precompute(self):
        # precomputes necessary data for trading decisions
        # calculate spread and zscores using ewma.
        self.mid_prices = self.data["mid_price"].unstack("symbol")
        self.hedge_ratio, self.fitted_spread = no_leakage_ols_spread(self.mid_prices, step=self._ols_step, window=self._ols_window)
        spread_ewm = self.fitted_spread.ewm(halflife=self._ewm_halflife)
        self.zscore = ((self.fitted_spread - spread_ewm.mean()) / spread_ewm.std()).transform(lambda x: x.mask(x.index.isin(x.index[:self._ewm_burnout]), 0))
        
        # calculate entry and exit signals based on z-scores
        self._enters = self.zscore.apply(self._enter_eval)
        self._buy_exits = self.zscore > -self._z_exit
        self._sell_exits = self.zscore < self._z_exit

        self._spread_sides = pd.Series(index=self.mid_prices.index)


    def _enter_eval(self, zscore):
        # evaluates if a new entry trade should be made based on zscore
        abs_z = abs(zscore)
        sign = np.sign(zscore)
        
        # determine position sizing based on zscore
        if abs_z < self._z_enter:
            return 0
        if abs_z >= self._z_max:
            return -sign * self._max_sizing
        return -sign * (self._sizing_slope * (abs_z - self._z_enter) + self._sizing_intercept)
        
    def _exit_eval(self, zscore):
        # determines if positions should be exited based on zscore
        return (-self._z_exit <= zscore) & (zscore <= self._z_exit)


    def evaluate(self, tick, cash, positions, engine):
        # main method to evaluate and execute trades
        
        # calculate current quantities 
        standing_qtys = engine.standing_quantities
        positions_qtys = {t:positions[t].size for t in Ticker}
        self.curr_qtys = {t:standing_qtys[t] + positions_qtys[t] for t in Ticker}

        
        # execute entry or exit trades based on current positions and signals
        # if enter signal and not invested
        if (enter_prop := self._enters[tick]) != 0 and (self._current_spread_side == 0):
            
            self._current_trade_id = self.gen_trade_id()
            self._current_spread_side = 1 if enter_prop > 0 else -1
            self._spread_sides.loc[tick] = self._current_spread_side
            
            # calculate quantity of NQc1
            NQc1_qty = cash * abs(enter_prop) / (self.mid_prices.loc[tick, "NQc1"] + self.hedge_ratio[tick] * self.mid_prices.loc[tick, "ESc1"])
            # calculate quantity of ESc1
            ESc1_qty = self.hedge_ratio[tick] * NQc1_qty
            
            if enter_prop > 0:
                return self._send_enter_orders(tick, int(np.floor(NQc1_qty)), -int(np.ceil(ESc1_qty)))
            else:
                return self._send_enter_orders(tick, -int(np.ceil(NQc1_qty)), int(np.floor(ESc1_qty)))
        
        elif (enter_prop := self._enters[tick]) != 0 and (np.sign(enter_prop) != self._current_spread_side):
            self._current_spread_side = 0
            self._spread_sides.loc[tick] = self._current_spread_side
            # send orders to close all positions
            return self._send_close_orders(tick)

        elif (self._buy_exits[tick] and self._current_spread_side == 1):
            self._current_spread_side = 0
            self._spread_sides.loc[tick] = self._current_spread_side
            # send orders to close all positions
            return self._send_close_orders(tick)
            
        elif (self._sell_exits[tick] and self._current_spread_side == -1):
            self._current_spread_side = 0
            self._spread_sides.loc[tick] = self._current_spread_side
            # send orders to close all positions
            return self._send_close_orders(tick)
            

    def _send_enter_orders(self, tick, NQc1_qty, ESc1_qty):
        # generates orders for entering positions
        return {
            t: Order(self.gen_order_id(), self._current_trade_id, tick, t, Side.BUY if qty > 0 else Side.SELL, abs(qty))
            for t, qty in zip(Ticker, [ESc1_qty, NQc1_qty]) if qty != 0
        }

    def _send_close_orders(self, tick):
        # generates orders to close existing positions
        return {
                t: Order(self.gen_order_id(), self._current_trade_id, tick, t, Side.BUY if self.curr_qtys[t] < 0 else Side.SELL, abs(self.curr_qtys[t]))
                for t in Ticker if self.curr_qtys[t] != 0
            }



#########################################################
########## IMPLEMENTATION OF 2 TRADING ENGINES ##########
#########################################################



class MidPriceInfLiquidity(TradingEngine):
    """
    An unrealistic trading engine that executes at mid price and there is unlimited liquidity
    """
        
    def execute(self, tick, cash, new_orders):
        """
        Executes standing and new orders for a given tick.
        
        Parameters:
            tick (datetime): The current time tick.
            cash (float): Current available cash.
            new_orders (dict): New orders to be executed.
        
        Returns:
            tuple: Updated cash and list of executed orders.
        """
        
        self.tick = tick
        self.current_cash = cash
        # retrieve current prices and depths for the tick
        self.current_prices = self.data.xs(tick, level="datetime")["mid_price"]
        
        executed_orders = []

        # process new orders
        for o in new_orders.values():
            executed_orders.append(self.trade(o))

        return (self.current_cash, executed_orders)

    def trade(self, order):
        """
        Executes a single order based on current market conditions.

        Parameters:
            order (Order): The order to be executed.

        Returns:
            Order: Updated order with execution details, or None if not executed.
        """
        # check that no order is negative
        if order.quantity <= 0:
            raise ValueError(f"at {self.tick}: {order} has negative quantity of {order.quantity}")

        # get mid price
        trade_price = self.current_prices.loc[order.ticker.name]

        cash_change = -1 if order.side == Side.BUY else 1
        
        self.current_cash += cash_change * (trade_price * order.quantity)
        
        # return order
        return order._replace(timestamp=self.tick, status=Status.EXECUTED, quantity=order.quantity, price=trade_price)



class BidAskConstTCost(TradingEngine):
    """
    A trading engine that executes orders based on bid and ask prices with constant transaction costs.
    It processes standing orders first, then new orders, while considering bidask spread, transaction costs 
    and liquidity, and updates cash and executed orders accordingly.
    """
    def __init__(self, tcost):
        # initialize engine 
        super().__init__()
        # T cost in dollars
        self._tcost = tcost
        
    def execute(self, tick, cash, new_orders):
        """
        Executes standing and new orders for a given tick.
        
        Parameters:
            tick (datetime): The current time tick.
            cash (float): Current available cash.
            new_orders (dict): New orders to be executed.
        
        Returns:
            tuple: Updated cash and list of executed orders.
        """
        
        self.tick = tick
        self.current_cash = cash
        # retrieve current prices and depths for the tick
        self.current_prices = self.data.xs(tick, level="datetime")[["bid_price", "ask_price"]].copy()
        self.current_depths = self.data.xs(tick, level="datetime")[["bid_size", "ask_size"]].copy()
        
        executed_orders = []

        # process standing orders
        while self.standing_orders:
            o = self.standing_orders.pop()
            if exec_order := self.trade(o):
                executed_orders.append(exec_order)
            else:
                break  # exit if no trade is executed

        # process new orders
        for o in new_orders.values():
            if exec_order := self.trade(o):
                executed_orders.append(exec_order)

        return (self.current_cash, executed_orders)

    def trade(self, order):
        """
        Executes a single order based on current market conditions.

        Parameters:
            order (Order): The order to be executed.

        Returns:
            Order: Updated order with execution details, or None if not executed.
        """
        # check that no order is negative
        if order.quantity <= 0:
            raise ValueError(f"at {self.tick}: {order} has negative quantity of {order.quantity}")

        # determine trade direction and corresponding market depth
        if order.side == Side.BUY:
            corresponding_price = "ask_price"
            corresponding_size = "ask_size"
            cash_change = -1  # cash outflow for buys
        else:
            corresponding_price = "bid_price"
            corresponding_size = "bid_size"
            cash_change = 1  # cash inflow for sells

        trade_price = self.current_prices.loc[order.ticker.name, corresponding_price]

        # handle case where there is no depth for the order
        if self.current_depths.loc[order.ticker.name, corresponding_size] == 0:
            self.standing_orders.append(order)
            return None

        # execute order partially or fully based on available depth
        elif order.quantity > self.current_depths.loc[order.ticker.name, corresponding_size]:
            max_quantity = self.current_depths.loc[order.ticker.name, corresponding_size]
            self.current_cash += cash_change * ((trade_price - self._tcost) * max_quantity)
            self.current_depths.loc[order.ticker.name, corresponding_size] = 0
            self.standing_orders.append(order._replace(timestamp=self.tick, status=Status.STANDING, quantity=order.quantity - max_quantity))
            return order._replace(timestamp=self.tick, status=Status.EXECUTED, quantity=max_quantity, price=trade_price)
        else:
            self.current_cash += cash_change * ((trade_price - self._tcost) * order.quantity)
            self.current_depths.loc[order.ticker.name, corresponding_size] -= order.quantity
            return order._replace(timestamp=self.tick, status=Status.EXECUTED, quantity=order.quantity, price=trade_price)






#################################################################
########## DASHBOARD CLASS TO SHOW BACKTESTING RESULTS ##########
#################################################################




class Dashboard:
    """
    Simple class to generate dashboard layout and then run on a localhost
    """
    def __init__(self, backtest_res):
        self._backtest_res = backtest_res
        self._app = Dash(__name__)
        self._font_dict = {"font-family": "Arial, sans-serif"}
        self._gen_layout()
        logging.getLogger('werkzeug').setLevel(logging.WARNING)

    def show(self, port=8050):
        # function to show the dashboard on given port
        try:
            self._app.run(debug=False, port=port)
        except Exception as e:
            print(f"Could not show Dash dashboard: {e}")
        return self

    
    def _gen_layout(self):
        # main function to contruct the layout
        self._app.layout = html.Div([
            html.H1('Backtesting Results of Pair Trading Strategy',  style=self._font_dict),
            html.Div([
                html.Div(self._gen_stats(), style={'width': '24%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                html.Div([dcc.Graph(id="pnl-graph", figure = self._gen_pnl_graph())], style={'width': '74%', 'display': 'inline-block'}),
            ], style={"margin-bottom": "20px"}),               
            html.Div([
                html.Div([dcc.Graph(id="dist-graph", figure= self._gen_dist_graph())], style={'width': '30%', 'display': 'inline-block'}),
                html.Div([dcc.Graph(id="positions-graph", figure=self._gen_position_graph())], style={'width': '65%', 'display': 'inline-block'})
            ])
        ], style={'margin': '20px'})

    
    ### ALL METHODS BELOW SIMPLY BUILD SOME PARTS OF THE LAYOUT
    
    def _format_stats(self):
        return {
            "Total Return": f"{self._backtest_res['total_return']*100:.4f} %",
            "Yearly Return": f"{self._backtest_res['yearly_return']*100:.4f} %",
            "Yearly Vol": f"{self._backtest_res['yearly_std']*100:.4f} %",
            "Yearly Sharpe": f"{self._backtest_res['sharpe_ratio']:.4f}",
            "Max Drawdown": f"{self._backtest_res['max_drawdown']:.2f} $",
            "Number of Trades": f"{self._backtest_res['trade_absolute_returns'].shape[0]:d}",
            "Hit Ratio": f"{(self._backtest_res['trade_absolute_returns'] > 0).sum() / self._backtest_res['trade_absolute_returns'].shape[0]:.2%}",
            "Mean Trade PNL": f"{self._backtest_res['trade_absolute_returns'].mean():.2} $",
            "Mean Slippage": f"{self._backtest_res['slippage_by_trade'].mean():.2f} $",
        }

    def _gen_stats(self):
        return [html.H3("Backtest Statistics",  style=self._font_dict)] + [
            html.Div([
                html.Div(f"{stat_label}:", style={**self._font_dict, **{'font-size': '18px','fontWeight': 'bold', 'display': 'inline-block', 'width': '180px'}}),
                html.Div(f"{stat_value}", style={**self._font_dict, **{'font-size': '18px', 'display': 'inline-block'}}),
            ], style={'marginBottom': '12px'})
            for stat_label, stat_value in self._format_stats().items()
        ]
    
    def _gen_pnl_graph(self):
        return (
            go.Figure(data=[go.Scatter(x=self._backtest_res["pnl_series"].index, y=self._backtest_res["pnl_series"])])
            .update_xaxes(
                rangebreaks=[
                    dict(bounds=["sat", "mon"]),  # hide weekend
                    dict(bounds=[21, 14], pattern="hour"),  # hide hours outside of 14:00-21:00
                ]
            )
            .update_layout(
                title={"text": "Marked-To-Market Cummulative PNL", "x": 0.5, "xanchor":"center"},
                margin=dict(l=25, r=25, t=45, b=25),
                height=400
            )
        ) 
        
    def _gen_dist_graph(self):
        return (
            go.Figure(
                data=[
                    go.Histogram(x=self._backtest_res["trade_absolute_returns"], name="Trade Returns", opacity=0.6),
                    go.Histogram(x=self._backtest_res["slippage_by_trade"], name="Trade Slippage", opacity=0.6)
                ]
            ).update_layout(
                title={"text": "Distribution of PNL and Slipppage by trade", "x": 0.5, "xanchor": "center"},
                barmode='overlay',
                legend={"x":0.5, "y":-0.05, "xanchor":'center', "yanchor":'top', "orientation":'h'},
                margin=dict(l=25, r=25, t=25, b=25),
                height=400,
            )
        )
    def _gen_position_graph(self):
        return (
            go.Figure(
                data=[
                    go.Scatter(x=self._backtest_res["positions"].index, y=self._backtest_res["positions"]["ESc1"], name="ESc1"),
                    go.Scatter(x=self._backtest_res["positions"].index, y=self._backtest_res["positions"]["NQc1"], name="NQc1", yaxis='y2'),
                ]
            )
            .update_xaxes(
                rangebreaks=[
                    dict(bounds=["sat", "mon"]),  # hide weekend
                    dict(bounds=[21, 14], pattern="hour"),  # hide hours outside of 14:00-21:00
                ]
            )
            .update_layout(
                title={"text": "Positions Evolution", "x": 0.5, "xanchor":"center"},
                yaxis1=dict(title="ESc1"),
                yaxis2=dict(title="NQc1", overlaying='y', side='right'),
                margin=dict(l=25, r=25, t=25, b=25),
                legend={"x":0.5, "y":-0.05, "xanchor":'center', "yanchor":'top', "orientation":'h'},
                height=400,
            )
        )


#########################################
########## CSV FILE PROCESSING ##########
#########################################

def process_csv_file(csv_file):
    # importing data
    try:
        raw_df = pd.read_csv(
            csv_file,
            index_col=["symbol", "year", "month", "day", "hour", "minute", "second"]    # adding directly the index columns
        ).sort_index()
    except:
        raise FileNotFoundError("Could not read the CSV file you provided")
    
    # checking for duplicates
    if not raw_df.index.is_unique:
        raise ValueError("The CSV file you provided has duplicates rows")

    # checking for NaNs
    if raw_df.isna().any(axis=None):
        raise ValueError("The CSV file you provided has NaNs, this backtest doens't handle NaNs as there were none in the given data")

    # checking for unvalid data
    if (raw_df<=0).any(axis=None):
        raise ValueError("The CSV file you provided has negative prices or quantities")
        
    df = raw_df.copy().unstack("symbol")
    
    # setting datetimeindex instead of multiindex of year, month, day, hour, minute, second (easier to manipulate)
    try:
        df["datetime"] = pd.to_datetime(df.index.to_frame())
    except:
        raise ValueError("At least one timestamp of the CSV file you provided couldn't be processed to datetime")
    
    df.set_index("datetime", inplace=True)

    return df.stack("symbol").swaplevel(0, 1).sort_index()



##########################
########## MAIN ##########
##########################


def main(csv_file, include_tcosts=True, dash_port=8050):
    
    print("\nProcessing csv file...")
    df = process_csv_file(csv_file)

    print("\nSetting up backtest and precomputing values..")
    backtest = (
        BacktestEngine(init_cash=10_000_000)
        .set_data(df)
        .set_trading_engine(BidAskConstTCost(tcost=0.1) if include_tcosts else MidPriceInfLiquidity())
        .set_strategy(EWMZscoreSymmetric(z_enter=1.8, z_max=4, z_exit=0.6, min_sizing=0.3, max_sizing=0.8, ewm_halflife=720, ewm_burnout=3600*4))
    ) 

    print("\nRunning backtest..")
    backtest.run()
    
    print("\nComputing evaluation metrics...")
    backtest_res = backtest.evaluate()

    print("\nLaunching dashboard")
    
    
    Dashboard(backtest_res).show(port=dash_port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a CSV file for backtesting.')
    
    parser.add_argument('csv_file', type=str, help='The name of the CSV file to process.')
    parser.add_argument('--exclude_tcosts', action=argparse.BooleanOptionalAction, help='Optional boolean to include transaction costs in backtest.')
    parser.add_argument('--dash_port', type=int, default=8050, help='Optional integer to set the port for Dash. Default is 8050.')
    args = parser.parse_args()

    main(args.csv_file, not args.exclude_tcosts, args.dash_port)