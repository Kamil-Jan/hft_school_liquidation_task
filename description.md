---

# Сигнал с ликвидаций

Даны данные за 3 месяца:
binance: trades, bbo, liquidations
bybit: liquidations

Нужно построить сигнал, который фильтрует binance trades (предполагая, что мы мейкерски собираем сделки).

---

## Данные

Все таблицы лежат в `parquet` с колонкой `timestamp` типа `int64` — **микросекунды с UNIX epoch (UTC)**. Это и есть единственная временная ось, которую нужно использовать.

Юниверс — `perp:btcusdt` и `perp:ethusdt` (он же используется на скрытом тесте).

Файлы и их колонки:

| путь | колонки |
|---|---|
| `data/binance_trades/perp_<sym>.parquet` | `timestamp, ticker, side, price, amount` |
| `data/binance_booktickers/perp_<sym>.parquet` | `timestamp, ticker, bid_price, bid_amount, ask_price, ask_amount` |
| `data/binance_liquidations/perp_<sym>.parquet` | `timestamp, ticker, side, price, amount` |
| `data/bybit_liquidations/<sym>.parquet` | `timestamp, ticker, side, price, amount` |

`ticker` равен `perp:btcusdt` / `perp:ethusdt` для Binance и `btcusdt` / `ethusdt` для Bybit (в обоих случаях совпадает с именем файла).

Конвенция `side`:

* в `trades` — сторона **тейкера** (`buy` ⇒ тейкер купил, мейкер продал);
* в `liquidations` — сторона **ордера-ликвидации** (`buy` ⇒ закрывают шорт принудительной покупкой ⇒ давление вверх).

## Cross-exchange задержка

Bybit и Binance — разные биржи, между ними есть сетевая задержка. Считаем, что события Bybit становятся нам доступны не раньше чем через **200 мс** после их `timestamp`: при построении любых фичей timestamp ликвидаций Bybit нужно сдвинуть вперёд на +200 мс, прежде чем сопоставлять их с временем сделок Binance.

---

## Markout
Фиксируем 3 горизонта

τ ∈ {30s, 120s, 300s}
Для сделки i:

* p_i — цена сделки;
* m_i(τ) — Binance BBO mid в момент `t_i + τ`, **forward-fill** (последний наблюдаемый mid на момент `t_i + τ`); если `t_i + τ` выходит за пределы доступного BBO, сделка исключается из подсчёта;
* s_i = +1, если taker buy, то есть maker sell;
* s_i = -1, если taker sell, то есть maker buy;
* w_i = min(notional_i, 100_000).

Maker PnL в bps:

pnl_i(τ) = -s_i * (m_i(τ) - p_i) / p_i * 10_000 + 0.5
где +0.5 bps — maker rebate.

---

## Signal

Сигнал задает бинарный фильтр:

f_i(τ) = 1, если сделку фильтруем
f_i(τ) = 0, если сделку оставляем
---

## Score

Baseline:

PnL_all(τ) =
    sum_i w_i * pnl_i(τ) / sum_i w_i
PnL на оставленных сделках:

PnL_kept(τ) =
    sum_i (1 - f_i(τ)) * w_i * pnl_i(τ)
    /
    sum_i (1 - f_i(τ)) * w_i
Финальный score:

Score(τ) = PnL_kept(τ) - PnL_all(τ)
Чем выше Score(τ), тем лучше.

Также нужно репортить PnL на отфильтрованных сделках:

PnL_filtered(τ) =
    sum_i f_i(τ) * w_i * pnl_i(τ)
    /
    sum_i f_i(τ) * w_i
---

## Constraint

Средний дневной clipped turnover оставленных сделок должен быть не меньше:

500_000 USD per day
То есть:

KeptTurnoverPerDay =
    sum_i (1 - f_i(τ)) * w_i / number_of_days
    >= 500_000
---

## Split

* train: `2025-12-01 → 2026-01-31` (2 месяца);
* validation: `2026-02-01 → 2026-02-28` (1 месяц);
* финальная test выборка будет скрытой и недоступной.

На hidden test будет проверяться Score(τ) и выполнение turnover constraint.

---

## Формат сдачи

Решение — Python-функция, которая принимает четыре фрейма (`trades`, `bbo`, `liq_binance`, `liq_bybit`) с теми же схемами и колонками, что в публичных файлах, и возвращает для каждого `τ ∈ {30, 120, 300}` массив той же длины, что `trades`, со значениями `0` (оставить сделку) или `1` (отфильтровать).

На скрытом тесте функция будет вызвана на тех же 4 типах данных, но за другие даты.

---

## ML

Фильтр `f_i(τ)` можно строить с помощью ML-модели. Например классификация со взвешанными сэмплами.