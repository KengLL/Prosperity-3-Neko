from collections import deque
from Algorithmic.datamodel import (
    Listing,
    Observation,
    Order,
    OrderDepth,
    ProsperityEncoder,
    Symbol,
    Trade,
    TradingState,
)
from typing import Any, List
import json
import jsonpickle


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: dict[Symbol, list[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(
                        state, self.truncate(state.traderData, max_item_length)
                    ),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(
        self, order_depths: dict[Symbol, OrderDepth]
    ) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


class Strategy:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.history = deque()
        self.ema = 0.0  # exponential moving average
        self.ema_multiplier = 0.3
        self.position = 0

    def update_ema(self, val: int):
        if not self.history:
            self.ema = val
        else:
            self.ema = self.ema_multiplier * val + (1 - self.ema_multiplier) * self.ema
        self.history.append(val)

        return self.ema

    def market_take_order(
        self, order_depth: OrderDepth, price: int, orders: list[Order]
    ):
        for bid, quantity in order_depth.buy_orders.items():
            if bid > price:
                self.position -= quantity
                orders.append(Order(self.symbol, bid, -quantity))

        for ask, quantity in order_depth.sell_orders.items():
            if ask < price:
                self.position += quantity
                orders.append(Order(self.symbol, ask, -quantity))

    def market_make(self, price: int, orders: list[Order], above, below):
        orders.append(Order(self.symbol, price - below, 50 - self.position))

        orders.append(Order(self.symbol, price + above, -50 - self.position))

    def market_make_neutral(self, price: int, orders: list[Order], above, below):
        if self.position > 0:
            orders.append(Order(self.symbol, price + above, -self.position))
        elif self.position < 0:
            orders.append(Order(self.symbol, price - below, -self.position))

    def run(self, state: TradingState):
        orders: List[Order] = []
        return orders


class Kelp(Strategy):
    def __init__(self, symbol: str):
        super().__init__(symbol)
        self.n = 20

    def run(self, state: TradingState):
        order_depth = state.order_depths[self.symbol]
        orders: List[Order] = []
        self.position = (
            state.position[self.symbol] if self.symbol in state.position else 0
        )
        if order_depth.buy_orders and order_depth.sell_orders:
            popular_bid = max(order_depth.buy_orders.items(), key=lambda k: k[1])[0]
            popular_ask = min(order_depth.sell_orders.items(), key=lambda k: k[1])[0]
            mid_price = (popular_bid + popular_ask) / 2
            self.history.append(mid_price)
            if len(self.history) > self.n:
                self.history.popleft()
            self.update_ema(mid_price)

        if len(self.history) < self.n:
            return orders

        # find the middle price of most popular (high volume) orders
        acceptable_price = int(round(mid_price))
        self.market_make(acceptable_price, orders, 1, 1)

        return orders


class RainforestResin(Strategy):
    def __init__(self, symbol: str):
        super().__init__(symbol)

    def run(self, state: TradingState):
        acceptable_price = 10000  # predefined price
        orders: List[Order] = []
        self.position = (
            state.position[self.symbol] if self.symbol in state.position else 0
        )
        # orders.append(Order(self.symbol, acceptable_price - 2, 50 - position))
        # orders.append(Order(self.symbol, acceptable_price + 2, -50 - position))

        self.market_make(acceptable_price, orders, 2, 2)
        return orders


class SquidInk(Strategy):
    def __init__(self):
        super().__init__("SQUID_INK")
        # long and short time periods for moving averages
        self.long = 15
        self.short = 3
        
        self.ema = 0.3
        self.history = []

    def run(self, state):
        order_depth = state.order_depths[self.symbol]
        orders: List[Order] = []
        self.position = (
            state.position[self.symbol] if self.symbol in state.position else 0
        )
        popular_bid = max(order_depth.buy_orders.items(), key=lambda k: k[1])[0]
        popular_ask = min(order_depth.sell_orders.items(), key=lambda k: k[1])[0]
        mid_price = (popular_bid + popular_ask) / 2

        self.history.append(mid_price)
        if len(self.history) > self.long:
            self.history.pop(0)

        # short long sma strategy
        if len(self.history) < self.long:
            return orders

        short_sma = sum(self.history[-self.short :]) / self.short
        long_sma = sum(self.history) / self.long
        if short_sma > long_sma:
            orders.append(Order(self.symbol, int(mid_price) - 1, 50 - self.position))

        elif short_sma < long_sma:
            orders.append(Order(self.symbol, int(mid_price) + 1, -50 - self.position))

        return orders


class Trader:
    def run(self, state: TradingState):
        # prev_trader_data = jsonpickle.decode(state.traderData)
        # Orders to be placed on exchange matching engine
        result = {}
        if not state.traderData:
            traderData = {}
            traderData["KELP"] = Kelp("KELP")
            traderData["RAINFOREST_RESIN"] = RainforestResin("RAINFOREST_RESIN")
            traderData["SQUID_INK"] = SquidInk()
        else:
            traderData = jsonpickle.decode(state.traderData)
        for product in state.order_depths:
            result[product] = traderData[product].run(state)

        # String value holding Trader state data required.
        # It will be delivered as TradingState.traderData on next execution.
        traderData = jsonpickle.encode(traderData)

        # Sample conversion request. Check more details below.
        conversions = 1
        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData
