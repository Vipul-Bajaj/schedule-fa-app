# Flask App: stock_analyzer_app.py
import datetime
import sys
from collections import OrderedDict
from urllib.request import Request, urlopen, URLError
from pyquery import PyQuery
import bisect
from typing import List, Dict, Tuple, Set, Optional, NamedTuple
import os  # For environment variables

from flask import Flask, render_template, request, flash, redirect, url_for

# --- Flask App Setup ---
app = Flask(__name__)

# Production configuration:
# 1. SECRET_KEY: Load from an environment variable for security.
#    In your production environment, set an environment variable, e.g.,
#    export FLASK_SECRET_KEY='your_very_long_random_secret_string'
#    Or for Windows: set FLASK_SECRET_KEY=your_very_long_random_secret_string
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev_default_secret_key_change_me')
if app.secret_key == 'dev_default_secret_key_change_me' and not app.debug:
    app.logger.warning(
        "WARNING: FLASK_SECRET_KEY is not set or is using the default development key in a non-debug environment!")

# 2. DEBUG mode: Should be False in production.
#    Set by environment variable FLASK_DEBUG=0 or FLASK_ENV=production
#    app.debug is automatically set by Flask based on FLASK_ENV or FLASK_DEBUG.
#    Explicitly: app.config['DEBUG'] = False (if not using FLASK_ENV)

# --- Constants ---
USER_AGENT = 'Mozilla/5.0'
YAHOO_FINANCE_URL_TEMPLATE = "https://finance.yahoo.com/quote/{symbol}/history/?period1={start_ts}&period2={end_ts}&interval=1d&filter=history&frequency=1d"
SBI_RATES_URL = "https://raw.githubusercontent.com/sahilgupta/sbi-fx-ratekeeper/main/csv_files/SBI_REFERENCE_RATES_USD.csv"

DATE_FORMAT_INPUT_SCRIPT = "%Y%m%d"
DATE_FORMAT_YAHOO_PARSE = "%b %d, %Y"
DATE_FORMAT_SBI_PARSE = '%Y-%m-%d'
DATE_FORMAT_DISPLAY = "%Y/%m/%d"


# --- Data Structures ---
class VestingEntry(NamedTuple):
    stock_symbol: str
    acquisition_date_str: str
    acquisition_dt: datetime.date
    num_shares: float


class ReportLine(NamedTuple):
    stock_symbol: str
    num_shares: float
    acquisition_date_str: str
    acq_price_display_str: str
    initial_value_display_str: str
    peak_output_str: str
    closing_output_str: str


class InputRow(NamedTuple):
    stock_symbol: str
    acquisition_date: str
    num_shares: str


StockSymbol = str
DateStringYYYYMMDD = str
Price = float
StockPricesForSymbol = OrderedDict[DateStringYYYYMMDD, Price]
AllStockPrices = Dict[StockSymbol, StockPricesForSymbol]
SbiExchangeRates = OrderedDict[DateStringYYYYMMDD, Price]


# --- Helper Functions ---

def get_date_obj_from_str(date_str: DateStringYYYYMMDD) -> Optional[datetime.date]:
    try:
        return datetime.datetime.strptime(date_str, DATE_FORMAT_INPUT_SCRIPT).date()
    except (ValueError, TypeError) as e:
        app.logger.error(f"Error converting date string '{date_str}': {e}")
        return None


def parse_vesting_data_from_form(
        stock_symbols_list: List[str],
        acquisition_dates_list: List[str],
        num_shares_list_str: List[str],
        calendar_year_str: str
) -> Tuple[Optional[List[VestingEntry]], Optional[int], Optional[Dict[StockSymbol, DateStringYYYYMMDD]]]:
    if not any(stock_symbols_list) and not any(acquisition_dates_list) and not any(num_shares_list_str):
        flash("At least one vesting entry is required.", "error")
        return None, None, None

    vesting_entries: List[VestingEntry] = []
    has_errors = False

    for i in range(len(stock_symbols_list)):
        symbol = stock_symbols_list[i].strip().upper()
        date_str = acquisition_dates_list[i].strip()
        shares_str = num_shares_list_str[i].strip()

        if not symbol and not date_str and not shares_str:
            continue

        if not symbol or not date_str or not shares_str:
            flash(f"Row {i + 1}: All fields (Symbol, Date, Shares) are required if any part of the row is filled.",
                  "error")
            has_errors = True
            continue

        acquisition_dt = get_date_obj_from_str(date_str)
        if not acquisition_dt:
            flash(f"Row {i + 1}: Invalid date format for '{date_str}'. Please use YYYYMMDD.", "error")
            has_errors = True
            continue

        try:
            num_shares = float(shares_str)
            if num_shares <= 0:
                flash(f"Row {i + 1}: Number of shares must be a positive number.", "error")
                has_errors = True
                continue
        except ValueError:
            flash(f"Row {i + 1}: Invalid number of shares '{shares_str}'. Must be a number.", "error")
            has_errors = True
            continue

        vesting_entries.append(VestingEntry(
            stock_symbol=symbol,
            acquisition_date_str=date_str,
            acquisition_dt=acquisition_dt,
            num_shares=num_shares
        ))

    if not vesting_entries and not has_errors:
        flash("At least one vesting entry is required.", "error")
        return None, None, None

    if has_errors:
        return None, None, None

    lowest_acquisition_dates: Dict[StockSymbol, DateStringYYYYMMDD] = {}
    for entry in vesting_entries:
        if entry.stock_symbol not in lowest_acquisition_dates or \
                entry.acquisition_date_str < lowest_acquisition_dates[entry.stock_symbol]:
            lowest_acquisition_dates[entry.stock_symbol] = entry.acquisition_date_str

    calendar_year_val: Optional[int] = None
    try:
        calendar_year_val = int(calendar_year_str)
        current_actual_year = datetime.date.today().year
        if not (1900 <= calendar_year_val <= current_actual_year + 10):
            flash(f"Invalid calendar year. Please enter a 4-digit year between 1900 and {current_actual_year + 10}.",
                  "error")
            return None, None, None
    except ValueError:
        flash("Calendar year must be a numeric value.", "error")
        return None, None, None

    return vesting_entries, calendar_year_val, lowest_acquisition_dates


def validate_vesting_data_web(vesting_entries: List[VestingEntry], calendar_year: int) -> bool:
    for entry in vesting_entries:
        if entry.acquisition_dt.year > calendar_year:
            flash(f"Error: Acquisition date {entry.acquisition_date_str} for {entry.stock_symbol} "
                  f"is later than the reporting calendar year {calendar_year}. Please fix.", "error")
            return False
    return True


# --- Data Fetching Functions ---
# Consider adding caching here for production (e.g., using Flask-Caching)
# @cache.cached(timeout=300, key_prefix='yahoo_stock_prices') # Example
def fetch_yahoo_stock_prices_for_symbol(
        stock_symbol: StockSymbol,
        earliest_acq_date_str: DateStringYYYYMMDD,
        calendar_year_to_process: int
) -> StockPricesForSymbol:
    prices_for_symbol: StockPricesForSymbol = OrderedDict()

    earliest_acq_dt = get_date_obj_from_str(earliest_acq_date_str)
    if not earliest_acq_dt:
        flash(f"Internal error: Invalid earliest acquisition date for {stock_symbol}.", "error")
        app.logger.error(f"Invalid earliest_acq_date_str '{earliest_acq_date_str}' for symbol {stock_symbol}")
        return prices_for_symbol

    fetch_start_dt_obj = earliest_acq_dt - datetime.timedelta(days=45)
    start_date_ts = int(datetime.datetime(
        fetch_start_dt_obj.year, fetch_start_dt_obj.month, fetch_start_dt_obj.day
    ).timestamp())

    end_date_dt = datetime.datetime(calendar_year_to_process + 1, 1, 15)
    end_date_ts = int(end_date_dt.timestamp())

    stock_prices_url = YAHOO_FINANCE_URL_TEMPLATE.format(
        symbol=stock_symbol, start_ts=start_date_ts, end_ts=end_date_ts
    )
    app.logger.info(f"Fetching stock prices for {stock_symbol} from {stock_prices_url}")

    try:
        req = Request(stock_prices_url, headers={'User-Agent': USER_AGENT})
        with urlopen(req, timeout=10) as response:
            html_content = response.read()
        pq = PyQuery(html_content)

        history_table_div = pq('div[data-testid="history-table"]')
        if not history_table_div.length:
            flash(
                f"Could not find history table for {stock_symbol}. The page structure might have changed or the symbol is invalid/delisted.",
                "warning")
            app.logger.warning(f"No history table div for {stock_symbol} at {stock_prices_url}")
            return prices_for_symbol

        table_element = history_table_div('table')
        if not table_element.length:
            flash(f"No table element found within history div for {stock_symbol}.", "warning")
            app.logger.warning(f"No table element for {stock_symbol} at {stock_prices_url}")
            return prices_for_symbol

        rows = table_element('tbody > tr')
        if not rows.length:
            flash(f"No data rows found in the history table for {stock_symbol}.", "warning")
            app.logger.warning(f"No data rows for {stock_symbol} at {stock_prices_url}")
            return prices_for_symbol

        for row_html_pq in rows.items():
            cells = row_html_pq('td')
            if len(cells) == 7:
                try:
                    date_text = cells.eq(0).text()
                    close_price_text = cells.eq(4).text()
                    if not date_text or not close_price_text or close_price_text == '-':
                        continue
                    dt_obj = datetime.datetime.strptime(date_text, DATE_FORMAT_YAHOO_PARSE)
                    formatted_date_key = dt_obj.strftime(DATE_FORMAT_INPUT_SCRIPT)
                    price_val = float(close_price_text.replace(',', ''))
                    prices_for_symbol[formatted_date_key] = round(price_val, 4)
                except ValueError:
                    app.logger.warning(f"Parsing ValueError for {stock_symbol} row: {cells.text()[:100]}")
                    continue
                except Exception as cell_err:
                    app.logger.error(f"Unexpected cell error for {stock_symbol}: {cell_err}. Row: {cells.text()[:100]}")
                    continue
        return OrderedDict(sorted(prices_for_symbol.items()))

    except URLError as e:
        flash(f"URL Error fetching stock prices for {stock_symbol}: {e}. Check network or symbol.", "error")
        app.logger.error(f"URLError for {stock_symbol} at {stock_prices_url}: {e}")
    except Exception as e:
        flash(f"Unexpected error for {stock_symbol} during price fetching: {e}", "error")
        app.logger.error(f"General Exception for {stock_symbol} at {stock_prices_url}: {e}")
    return prices_for_symbol


def fetch_all_stock_prices(
        symbols_to_fetch: Set[StockSymbol],
        lowest_acq_dates: Dict[StockSymbol, DateStringYYYYMMDD],
        calendar_year_to_process: int
) -> AllStockPrices:
    all_prices: AllStockPrices = {}
    for symbol in symbols_to_fetch:
        all_prices[symbol] = fetch_yahoo_stock_prices_for_symbol(
            symbol, lowest_acq_dates[symbol], calendar_year_to_process
        )
    return all_prices


# @cache.cached(timeout=3600, key_prefix='sbi_rates') # Example: Cache for 1 hour
def fetch_sbi_exchange_rates() -> SbiExchangeRates:
    app.logger.info(f"Fetching SBI exchange rates from {SBI_RATES_URL}")
    sbi_rates_data: SbiExchangeRates = OrderedDict()
    try:
        req = Request(SBI_RATES_URL, headers={'User-Agent': USER_AGENT})
        with urlopen(req, timeout=10) as response:
            html_content = response.read().decode('utf-8')

        lines = html_content.splitlines()

        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) >= 3:
                date_str_csv = parts[0][:10]
                rate_str = parts[2]
                # Check if rate_str is a valid number (can be float) and not zero
                try:
                    rate_val = float(rate_str)
                    if rate_val == 0.0:
                        continue
                except ValueError:
                    continue  # Skip if rate is not a valid number

                try:
                    dt_obj = datetime.datetime.strptime(date_str_csv, DATE_FORMAT_SBI_PARSE)
                    formatted_date_key = dt_obj.strftime(DATE_FORMAT_INPUT_SCRIPT)
                    sbi_rates_data[formatted_date_key] = round(rate_val, 4)
                except ValueError:
                    app.logger.warning(f"Skipping SBI rate line with invalid date format: {line}")
                    continue
    except URLError as e:
        flash(f"Critical Error: Could not fetch SBI rates: {e}. Check URL or network connection.", "error")
        app.logger.error(f"URLError fetching SBI rates: {e}")
    except Exception as e:
        flash(f"Critical Error: An unexpected error occurred during SBI rate fetching: {e}", "error")
        app.logger.error(f"General Exception fetching SBI rates: {e}")

    if not sbi_rates_data:
        flash("No SBI rates fetched. Calculations involving INR will be affected.", "warning")
    return OrderedDict(sorted(sbi_rates_data.items()))


# --- Calculation Logic Functions ---
def get_sbi_rate_on_or_before_date(
        target_date_str: Optional[DateStringYYYYMMDD],
        sbi_rates: SbiExchangeRates
) -> Optional[Price]:
    if not target_date_str or not sbi_rates:
        return None
    rate_dates = list(sbi_rates.keys())
    insertion_idx = bisect.bisect_right(rate_dates, target_date_str)
    if insertion_idx == 0:
        return sbi_rates[rate_dates[0]] if rate_dates else None
    actual_date_key = rate_dates[insertion_idx - 1]
    return sbi_rates[actual_date_key]


def get_stock_price_on_or_before_date(
        prices_for_symbol: StockPricesForSymbol,
        target_date_str: DateStringYYYYMMDD
) -> Tuple[Optional[DateStringYYYYMMDD], Optional[Price]]:
    if not prices_for_symbol:
        return None, None
    price_dates = list(prices_for_symbol.keys())
    insertion_idx = bisect.bisect_right(price_dates, target_date_str)
    if insertion_idx == 0:
        return None, None
    actual_date_key = price_dates[insertion_idx - 1]
    return actual_date_key, prices_for_symbol[actual_date_key]


def find_peak_stock_info(
        prices_for_symbol: StockPricesForSymbol,
        calendar_year: int,
        acquisition_dt: datetime.date
) -> Tuple[Optional[DateStringYYYYMMDD], Optional[Price]]:
    if not prices_for_symbol:
        return None, None
    cal_year_start_dt = datetime.date(calendar_year, 1, 1)
    search_effective_start_dt = max(acquisition_dt, cal_year_start_dt)
    search_end_dt = datetime.date(calendar_year, 12, 31)
    peak_price_val = -1.0
    peak_date_str: Optional[DateStringYYYYMMDD] = None
    for date_str, price in prices_for_symbol.items():
        current_dt_obj = get_date_obj_from_str(date_str)
        if not current_dt_obj: continue

        if search_effective_start_dt <= current_dt_obj <= search_end_dt:
            if price > peak_price_val:
                peak_price_val = price
                peak_date_str = date_str
    return peak_date_str, peak_price_val


def find_year_end_closing_stock_info(
        prices_for_symbol: StockPricesForSymbol,
        calendar_year: int
) -> Tuple[Optional[DateStringYYYYMMDD], Optional[Price]]:
    if not prices_for_symbol:
        return None, None
    last_dec_trade_date_str: Optional[DateStringYYYYMMDD] = None
    last_dec_trade_price: Optional[Price] = None
    for date_str, price in prices_for_symbol.items():
        dt_obj = get_date_obj_from_str(date_str)
        if not dt_obj: continue

        if dt_obj.year == calendar_year and dt_obj.month == 12:
            last_dec_trade_date_str = date_str
            last_dec_trade_price = price
    return last_dec_trade_date_str, last_dec_trade_price


# --- Output Formatting for Web ---
def format_report_line_for_web(
        entry: VestingEntry,
        acq_price_date_str: Optional[DateStringYYYYMMDD],
        acq_usd_price_val: Optional[Price],
        sbi_rate_on_acq: Optional[Price],
        peak_date_str_report: Optional[DateStringYYYYMMDD],
        peak_usd_price_val: Optional[Price],
        sbi_rate_on_peak: Optional[Price],
        closing_date_str_report: Optional[DateStringYYYYMMDD],
        closing_usd_price_val: Optional[Price],
        sbi_rate_on_closing: Optional[Price]
) -> ReportLine:
    num_shares = entry.num_shares
    acq_price_display = "N/A"
    initial_value_display = "N/A"
    peak_output_display = "N/A (N/A, N/A)"
    closing_output_display = "N/A (N/A, N/A)"

    if acq_usd_price_val is not None and sbi_rate_on_acq is not None:
        share_price_inr_on_acq = round(acq_usd_price_val * sbi_rate_on_acq, 4)  # Display with 2 decimal places
        initial_value_investment_inr = round(num_shares * share_price_inr_on_acq, 4)  # Display with 2 decimal places
        acq_price_display = f"{share_price_inr_on_acq:.2f}"
        initial_value_display = f"{initial_value_investment_inr:.2f}"

    if peak_date_str_report and peak_usd_price_val is not None and sbi_rate_on_peak is not None:
        peak_price_inr_per_share = round(peak_usd_price_val * sbi_rate_on_peak, 4)  # Display with 2 decimal places
        total_peak_value_inr = round(num_shares * peak_price_inr_per_share, 4)  # Display with 2 decimal places
        peak_date_obj = get_date_obj_from_str(peak_date_str_report)
        peak_date_formatted = peak_date_obj.strftime(DATE_FORMAT_DISPLAY) if peak_date_obj else "N/A"
        peak_output_display = f"{total_peak_value_inr:.2f} ({peak_date_formatted}, {peak_price_inr_per_share:.2f})"

    if closing_date_str_report and closing_usd_price_val is not None and sbi_rate_on_closing is not None:
        closing_price_inr_per_share = round(closing_usd_price_val * sbi_rate_on_closing,
                                            4)  # Display with 2 decimal places
        total_closing_value_inr = round(num_shares * closing_price_inr_per_share, 4)  # Display with 2 decimal places
        closing_date_obj = get_date_obj_from_str(closing_date_str_report)
        closing_date_formatted = closing_date_obj.strftime(DATE_FORMAT_DISPLAY) if closing_date_obj else "N/A"
        closing_output_display = f"{total_closing_value_inr:.2f} ({closing_date_formatted}, {closing_price_inr_per_share:.2f})"

    return ReportLine(
        stock_symbol=entry.stock_symbol,
        num_shares=entry.num_shares,
        acquisition_date_str=entry.acquisition_date_str,
        acq_price_display_str=acq_price_display,
        initial_value_display_str=initial_value_display,
        peak_output_str=peak_output_display,
        closing_output_str=closing_output_display
    )


# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    current_year = datetime.date.today().year - 1
    form_data_repopulation = []
    results_data = None

    if request.method == 'POST':
        stock_symbols_list = request.form.getlist('stock_symbol')
        acquisition_dates_list = request.form.getlist('acquisition_date')
        num_shares_list_str = request.form.getlist('num_shares')
        calendar_year_str = request.form.get('calendar_year', str(current_year))

        for i in range(len(stock_symbols_list)):
            form_data_repopulation.append(InputRow(
                stock_symbols_list[i],
                acquisition_dates_list[i],
                num_shares_list_str[i]
            ))
        if not form_data_repopulation:
            form_data_repopulation.append(InputRow("", "", ""))

        if not calendar_year_str:
            flash("Calendar year is required.", "error")
            return render_template('index.html',
                                   form_data=form_data_repopulation,
                                   calendar_year_val=calendar_year_str,
                                   current_year=current_year,
                                   results=None)

        vesting_data_list, calendar_year_int, lowest_acq_dates_map = parse_vesting_data_from_form(
            stock_symbols_list, acquisition_dates_list, num_shares_list_str, calendar_year_str
        )

        if vesting_data_list is None or calendar_year_int is None or lowest_acq_dates_map is None:
            return render_template('index.html',
                                   form_data=form_data_repopulation,
                                   calendar_year_val=calendar_year_str,
                                   current_year=current_year,
                                   results=None)

        if not validate_vesting_data_web(vesting_data_list, calendar_year_int):
            return render_template('index.html',
                                   form_data=form_data_repopulation,
                                   calendar_year_val=calendar_year_str,
                                   current_year=current_year,
                                   results=None)

        all_stock_symbols_set = set(entry.stock_symbol for entry in vesting_data_list)

        # Consider adding caching for these calls in production
        all_stock_prices_data = fetch_all_stock_prices(
            all_stock_symbols_set, lowest_acq_dates_map, calendar_year_int
        )
        sbi_rates_data = fetch_sbi_exchange_rates()

        report_data_for_template: List[ReportLine] = []

        for entry in vesting_data_list:
            symbol = entry.stock_symbol
            acq_date_str = entry.acquisition_date_str
            acq_dt = entry.acquisition_dt

            current_stock_prices = all_stock_prices_data.get(symbol, OrderedDict())

            if not current_stock_prices:
                flash(f"No stock price data retrieved for {symbol}. Results for this entry may be incomplete.",
                      "warning")
                report_data_for_template.append(
                    format_report_line_for_web(entry, None, None, None, None, None, None, None, None, None))
                continue

            price_date_for_acq_str, usd_price_on_acq = get_stock_price_on_or_before_date(
                current_stock_prices, acq_date_str
            )
            sbi_rate_for_acq = get_sbi_rate_on_or_before_date(
                price_date_for_acq_str, sbi_rates_data
            )

            peak_date_str, peak_usd_price = find_peak_stock_info(
                current_stock_prices, calendar_year_int, acq_dt
            )
            sbi_rate_for_peak = get_sbi_rate_on_or_before_date(
                peak_date_str, sbi_rates_data
            )

            closing_date_str, closing_usd_price = find_year_end_closing_stock_info(
                current_stock_prices, calendar_year_int
            )
            sbi_rate_for_closing = get_sbi_rate_on_or_before_date(
                closing_date_str, sbi_rates_data
            )

            report_line_obj = format_report_line_for_web(
                entry,
                price_date_for_acq_str, usd_price_on_acq, sbi_rate_for_acq,
                peak_date_str, peak_usd_price, sbi_rate_for_peak,
                closing_date_str, closing_usd_price, sbi_rate_for_closing
            )
            report_data_for_template.append(report_line_obj)

        results_data = report_data_for_template
        if not results_data and vesting_data_list:  # If there were entries but no results (e.g. all fetching failed)
            flash("No data processed. Please check your input or external data sources.", "info")

        return render_template('index.html',
                               results=results_data,
                               form_data=form_data_repopulation,
                               calendar_year_val=calendar_year_str,
                               current_year=current_year)

    if not form_data_repopulation:
        form_data_repopulation.append(InputRow("", "", ""))
    return render_template('index.html',
                           form_data=form_data_repopulation,
                           calendar_year_val=str(current_year),
                           current_year=current_year,
                           results=None)


if __name__ == "__main__":
    # For development:
    # app.run(debug=True)

    # For production, use a WSGI server like Gunicorn or Waitress.
    # Example with Gunicorn (install with: pip install gunicorn):
    # gunicorn -w 4 -b 0.0.0.0:8000 stock_analyzer_app:app
    #
    # Example with Waitress (install with: pip install waitress):
    # waitress-serve --host 0.0.0.0 --port 8000 stock_analyzer_app:app
    #
    # The following is for development ONLY. Ensure app.debug is False in production.
    if os.environ.get('FLASK_ENV') == 'production':
        app.config['DEBUG'] = False
        # from waitress import serve # Example for Waitress
        # serve(app, host='0.0.0.0', port=8000)
        print("Flask app is configured for production. Run with a WSGI server like Gunicorn or Waitress.")
        print("Example: gunicorn -w 4 schedule_fa_app:app")
    else:
        app.config['DEBUG'] = True
        print("Flask app running in DEBUG mode. Open http://127.0.0.1:5000/ in your browser.")
        app.run(host='0.0.0.0', port=5000)
