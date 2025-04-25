from collections import defaultdict, deque
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
from statistics import NormalDist, stdev
import math
import numpy as np


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
        lo, hi = 0, min(len(value), max_length)
        out = ""

        while lo <= hi:
            mid = (lo + hi) // 2

            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."

            encoded_candidate = json.dumps(candidate)

            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return out


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

    def vwap(self, order_depth: OrderDepth):
        total_price = 0
        total_quantity = 0
        for price, quantity in order_depth.buy_orders.items():
            total_price += price * quantity
            total_quantity += quantity

        for price, quantity in order_depth.sell_orders.items():
            quantity = -quantity  # negative quantity for sell orders
            total_price += price * quantity
            total_quantity += quantity

        if total_quantity == 0:
            return 0

        return int(round(total_price / total_quantity))

    def mid_price(self, order_depth: OrderDepth):
        if order_depth.buy_orders and order_depth.sell_orders:
            popular_bid = max(order_depth.buy_orders.items(), key=lambda k: k[1])[0]
            popular_ask = min(order_depth.sell_orders.items(), key=lambda k: k[1])[0]
            return (popular_bid + popular_ask) / 2
        return 0

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
    def __init__(self, threshold=2, size=50):
        super().__init__("SQUID_INK")
        # long and short time periods for moving averages
        self.long = 45
        self.short = 1

        self.ema = 1
        self.ema_multiplier = 0.2
        self.history = []
        self.size = size
        self.threshold = threshold

    def run(self, state):
        order_depth = state.order_depths[self.symbol]
        orders: List[Order] = []
        self.position = (
            state.position[self.symbol] if self.symbol in state.position else 0
        )
        price = self.vwap(order_depth)
        self.update_ema(price)

        self.history.append(price)
        if len(self.history) > self.long:
            self.history.pop(0)
            avg_price = sum(self.history) / len(self.history)
            std = stdev(self.history)
            z_score = (price - avg_price) / std
            if z_score < -self.threshold:
                orders.append(Order(self.symbol, price, 50 - self.position))
            elif z_score > self.threshold:
                orders.append(Order(self.symbol, price, -50 - self.position))
            elif abs(z_score) < 0.3:
                orders.append(Order(self.symbol, price, -self.position))

        return orders


class ETF(Strategy):
    def __init__(self, symbol: str, holdings, threshold, limit, window, std):
        super().__init__(symbol)
        self.holdings = holdings
        self.position = 0
        self.threshold = threshold
        self.limit = limit
        self.history = deque()
        self.window = window
        self.std = std

    # Actual Item price
    def synthetic_price(self, state: TradingState):
        price = 0
        for product, quantity in self.holdings.items():
            if product in state.order_depths:
                order_depth = state.order_depths[product]
                product_price = self.vwap(order_depth)
                if product_price == 0:
                    return 0
                price += quantity * product_price

        return price

    def run(self, state):
        etf_price = self.vwap(state.order_depths[self.symbol])
        synthetic_price = self.synthetic_price(state)
        orders: List[Order] = []
        if synthetic_price == 0:
            return orders
        diff = etf_price - synthetic_price
        self.position = (
            state.position[self.symbol] if self.symbol in state.position else 0
        )
        # if diff < -self.threshold and self.position + size <= self.limit:
        #     orders.append(Order(self.symbol, etf_price, size))
        # elif diff > self.threshold and self.position - size >= -self.limit:
        #     orders.append(Order(self.symbol, etf_price, -size))
        # elif round(diff) == 0:
        #     orders.append(Order(self.symbol, etf_price, -self.position))

        self.history.append(diff)
        if len(self.history) > self.window:
            self.history.popleft()
            avg_price = sum(self.history) / len(self.history)
            std = stdev(self.history)
            if std == 0:
                return orders
            z_score = (diff - avg_price) / std

            if z_score < -self.threshold:
                orders.append(Order(self.symbol, etf_price, self.limit - self.position))
            elif z_score > self.threshold:
                orders.append(
                    Order(self.symbol, etf_price, -self.limit - self.position)
                )
            elif abs(z_score) < 0.8:
                orders.append(Order(self.symbol, etf_price, -self.position))

        return orders


class Coupons(Strategy):
    def __init__(
        self,
        symbol: str,
        underlying: str,
        strike: int,
        threshold=1,
    ):
        super().__init__(symbol)
        self.position = 0
        self.strike = strike
        self.underlying = underlying
        self.dte = 8
        self.limit = 200
        self.size = 50
        self.threshold = threshold
        self.history = deque(maxlen=30)
        self.residual = 0

    def black_scholes_delta(self, S, K, T, r, sigma):
        N = NormalDist()
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        delta = N.cdf(d1)
        return delta

    def black_scholes_call(self, spot, strike, time_to_expiry, volatility):
        d1 = (
            math.log(spot)
            - math.log(strike)
            + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * math.sqrt(time_to_expiry))
        d2 = d1 - volatility * math.sqrt(time_to_expiry)
        call_price = spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)
        return call_price

    def implied_volatility(
        self,
        call_price,
        spot,
        strike,
        time_to_expiry,
        max_iterations=200,
        tolerance=1e-10,
    ):
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0  # Initial guess as the midpoint
        for _ in range(max_iterations):
            estimated_price = self.black_scholes_call(
                spot, strike, time_to_expiry, volatility
            )
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                break
            elif diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility

    def calculate(self, state):
        self.expiry = (self.dte - state.timestamp / 1e6) / 365

        self.underlying_price = self.vwap(state.order_depths[self.underlying])
        self.option_price = self.vwap(state.order_depths[self.symbol])
        self.moneyness = math.log(self.strike / self.underlying_price) / math.sqrt(
            self.expiry
        )
        self.iv = self.implied_volatility(
            self.option_price, self.underlying_price, self.strike, self.expiry
        )
        self.delta = self.black_scholes_delta(
            self.underlying_price, self.strike, self.expiry, 0.0, self.iv
        )

    def run(self, state):
        orders: List[Order] = []
        self.position = (
            state.position[self.symbol] if self.symbol in state.position else 0
        )
        self.price = self.vwap(state.order_depths[self.symbol])

        self.history.append(self.price)
        if len(self.history) < self.history.maxlen:
            return orders

        # logger.print(f"{self.strike} {self.residual}")
        # avg_price = sum(self.history) / len(self.history)
        # std = stdev(self.history)
        avg_price = np.mean(self.history)
        std = np.std(self.history)
        if std == 0:
            return orders
        z_score = (self.price - avg_price) / std
        if z_score < -self.threshold:
            orders.append(Order(self.symbol, self.price, self.limit - self.position))
        elif z_score > self.threshold:
            orders.append(Order(self.symbol, self.price, -self.limit - self.position))
        return orders


class Hedge(Strategy):
    def __init__(self, symbol: str, limit):
        super().__init__(symbol)
        self.target = 0
        self.limit = limit

    def place_hedge(self, amount):
        self.target = amount

    def run(self, state: TradingState):
        price = self.vwap(state.order_depths[self.symbol])
        orders: List[Order] = []
        curr_position = (
            state.position[self.symbol] if self.symbol in state.position else 0
        )
        target_position = int(round(self.target))
        if target_position < -self.limit:
            target_position = -self.limit
        elif target_position > self.limit:
            target_position = self.limit

        orders.append(Order(self.symbol, price, target_position - curr_position))
        self.target = 0
        return orders


class Rock(Strategy):
    def __init__(self, symbol: str):
        super().__init__(symbol)
        self.position = 0
        self.limit = 400
        self.history = deque(maxlen=35)
        self.threshold = 1

    def run(self, state: TradingState):
        price = self.vwap(state.order_depths[self.symbol])
        orders: List[Order] = []
        curr_position = (
            state.position[self.symbol] if self.symbol in state.position else 0
        )

        self.history.append(price)
        if len(self.history) < self.history.maxlen:
            return orders

        avg_price = np.mean(self.history)
        std = np.std(self.history)
        z_score = (price - avg_price) / std
        if z_score < -self.threshold:
            orders.append(Order(self.symbol, price, self.limit - curr_position))
        elif z_score > self.threshold:
            orders.append(Order(self.symbol, price, -self.limit - curr_position))
        elif abs(z_score) < 0.2:
            orders.append(Order(self.symbol, price, -curr_position))
        return orders


class Macarons(Strategy):
    def __init__(self, symbol: str, limit):
        super().__init__(symbol)
        self.limit = limit
        self.conversions = 0
        self.threshold = 1
        self.CSI = 45
        self.checkpoint = 0
        self.projected = None
        self.history = deque(maxlen=50)

    def run(self, state: TradingState):
        orders: List[Order] = []
        position = state.position.get(self.symbol, 0)
        vwap = self.vwap(state.order_depths[self.symbol])
        # self.conversions = -position
        self.conversions = min(max(self.conversions, -10), 10)
        time = state.timestamp
        obs = state.observations.conversionObservations.get(self.symbol, None)
        if obs is None:
            return orders
        curr_sunlight = obs.sunlightIndex
        if time % 100_000 == 0:
            self.checkpoint = curr_sunlight
            self.projected = None
        elif time % 100_000 >= 10_000 and self.projected is None:
            sunlight_change = curr_sunlight - self.checkpoint
            self.projected = self.checkpoint + sunlight_change * 100_000 / (
                time % 100_000
            )

        best_bid = max(state.order_depths[self.symbol].buy_orders.keys())
        best_ask = min(state.order_depths[self.symbol].sell_orders.keys())

        # orders.append(
        #     Order(
        #         self.symbol,
        #         round(sell_price),
        #         self.limit,
        #     )
        # )

        if self.projected is not None:
            logger.print(self.projected)
            if self.projected <= self.CSI:
                orders.append(Order(self.symbol, vwap, self.limit - position))
            elif self.checkpoint <= self.CSI and self.projected > self.CSI:
                # recovery from sunlight
                orders.append(Order(self.symbol, best_bid, -self.limit - position))
            elif self.checkpoint > self.CSI and self.projected > self.CSI:
                # mean reversion
                self.history.append(vwap)
                if len(self.history) == self.history.maxlen:
                    avg_price = np.mean(self.history)
                    std = np.std(self.history)
                    z_score = (vwap - avg_price) / std
                    if z_score < -self.threshold:
                        orders.append(
                            Order(self.symbol, best_bid, self.limit - position)
                        )
                    elif z_score > self.threshold:
                        orders.append(
                            Order(self.symbol, best_ask, -self.limit - position)
                        )
        return orders


class Trader:
    dte = 8

    def run(self, state: TradingState):
        # prev_trader_data = jsonpickle.decode(state.traderData)
        # Orders to be placed on exchange matching engine
        result = defaultdict(list)
        if not state.traderData:
            traderData = {}
            traderData["KELP"] = Kelp("KELP")
            traderData["RAINFOREST_RESIN"] = RainforestResin("RAINFOREST_RESIN")
            traderData["SQUID_INK"] = SquidInk(threshold=1.5)
            traderData["PICNIC_BASKET1"] = ETF(
                "PICNIC_BASKET1",
                {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1},
                threshold=1.5,
                limit=60,
                window=45,
                std=90,
            )
            traderData["PICNIC_BASKET2"] = ETF(
                "PICNIC_BASKET2",
                {"CROISSANTS": 4, "JAMS": 2},
                threshold=1.5,
                limit=100,
                window=45,
                std=55,
            )
            traderData["VOLCANIC_ROCK_VOUCHER_10500"] = Coupons(
                "VOLCANIC_ROCK_VOUCHER_10500",
                "VOLCANIC_ROCK",
                10500,
            )
            traderData["VOLCANIC_ROCK_VOUCHER_10250"] = Coupons(
                "VOLCANIC_ROCK_VOUCHER_10250",
                "VOLCANIC_ROCK",
                10250,
            )
            traderData["VOLCANIC_ROCK_VOUCHER_10000"] = Coupons(
                "VOLCANIC_ROCK_VOUCHER_10000",
                "VOLCANIC_ROCK",
                10000,
            )
            traderData["VOLCANIC_ROCK_VOUCHER_9750"] = Coupons(
                "VOLCANIC_ROCK_VOUCHER_9750",
                "VOLCANIC_ROCK",
                9750,
            )
            traderData["VOLCANIC_ROCK_VOUCHER_9500"] = Coupons(
                "VOLCANIC_ROCK_VOUCHER_9500",
                "VOLCANIC_ROCK",
                9500,
            )
            traderData["MAGNIFICENT_MACARONS"] = Macarons(
                "MAGNIFICENT_MACARONS", limit=75
            )
            traderData["VOLCANIC_ROCK"] = Rock("VOLCANIC_ROCK")
        else:
            traderData = jsonpickle.decode(state.traderData)

        # overall option calculations
        # ivs = []
        # moneyness = []
        # for strike in [9500, 9750, 10000, 10250, 10500]:
        #     option = traderData[f"VOLCANIC_ROCK_VOUCHER_{strike}"]
        #     option.calculate(state)
        #     ivs.append(option.iv)
        #     moneyness.append(option.moneyness)

        # vol_smile = np.polyfit(moneyness, ivs, 2)
        # for strike in [9500, 9750, 10000, 10250, 10500]:
        #     option = traderData[f"VOLCANIC_ROCK_VOUCHER_{strike}"]
        #     option.residual = option.iv - np.polyval(vol_smile, option.moneyness)

        for product in state.order_depths:
            if product in traderData:
                orders = traderData[product].run(state)
                for order in orders:
                    result[order.symbol].append(order)

        # hedging
        croissants = Hedge("CROISSANTS", limit=250)
        djembe = Hedge("DJEMBES", limit=60)
        jams = Hedge("JAMS", limit=350)
        # volcanic_rock = Hedge("VOLCANIC_ROCK", limit=400)

        croissants.place_hedge(
            traderData["PICNIC_BASKET1"].position * -6
            + traderData["PICNIC_BASKET2"].position * -4
        )

        jams.place_hedge(
            traderData["PICNIC_BASKET1"].position * -3
            + traderData["PICNIC_BASKET2"].position * -2
        )
        djembe.place_hedge(traderData["PICNIC_BASKET1"].position * -1)

        # volcanic_rock.place_hedge(
        #     traderData["VOLCANIC_ROCK_VOUCHER_10500"].position
        #     * -traderData["VOLCANIC_ROCK_VOUCHER_10500"].delta
        #     + traderData["VOLCANIC_ROCK_VOUCHER_10250"].position
        #     * -traderData["VOLCANIC_ROCK_VOUCHER_10250"].delta
        #     + traderData["VOLCANIC_ROCK_VOUCHER_10000"].position
        #     * -traderData["VOLCANIC_ROCK_VOUCHER_10000"].delta
        #     + traderData["VOLCANIC_ROCK_VOUCHER_9750"].position
        #     * -traderData["VOLCANIC_ROCK_VOUCHER_9750"].delta
        #     + traderData["VOLCANIC_ROCK_VOUCHER_9500"].position
        #     * -traderData["VOLCANIC_ROCK_VOUCHER_9500"].delta
        # )
        result[croissants.symbol] = croissants.run(state)
        result[jams.symbol] = jams.run(state)
        result[djembe.symbol] = djembe.run(state)
        # result[volcanic_rock.symbol] = volcanic_rock.run(state)

        # String value holding Trader state data required.
        # It will be delivered as TradingState.traderData on next execution.
        conversions = traderData["MAGNIFICENT_MACARONS"].conversions

        traderData = jsonpickle.encode(traderData)
        # Sample conversion request. Check more details below.
        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData
