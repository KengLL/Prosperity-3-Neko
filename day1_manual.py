trades = {
    "SNOW": {"PIZZA": 1.45, "SILICON": 0.52, "SEASHELL": 0.72},
    "PIZZA": {"SNOW": 0.7, "SILICON": 0.31, "SEASHELL": 0.48},
    "SILICON": {"SNOW": 1.95, "PIZZA": 3.1, "SEASHELL": 1.49},
    "SEASHELL": {"SNOW": 1.34, "PIZZA": 1.98, "SILICON": 0.64},
}


def get_best_trade(trades, starting, remaining):
    if remaining == 1:
        val = trades[starting]["SEASHELL"] if starting != "SEASHELL" else 1
        return val, "SEASHELL"
    best_val, best_path = 0, ""
    for trade, rate in trades[starting].items():
        val, cont_trade = get_best_trade(trades, trade, remaining - 1)
        val = rate * val
        if val > best_val:
            best_val = val
            best_path = trade + " " + cont_trade

    return best_val, best_path


print(get_best_trade(trades, "SEASHELL", 5))
