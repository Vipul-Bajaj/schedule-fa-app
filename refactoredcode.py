# In case of certificate error on MAC, Go to Applications Folder, Python and double click 'Install Certificates.command'
import datetime
import sys
from collections import OrderedDict
from urllib.request import Request, urlopen, URLError
from pyquery import PyQuery
import bisect  # For efficient searching in sorted lists
from typing import List, Dict, Tuple, Set, Optional, NamedTuple

# --- Constants ---
USER_AGENT = 'Mozilla/5.0'
YAHOO_FINANCE_URL_TEMPLATE = "https://finance.yahoo.com/quote/{symbol}/history?period1={start_ts}&period2={end_ts}&interval=1d&filter=history&frequency=1d"
SBI_RATES_URL = "https://raw.githubusercontent.com/sahilgupta/sbi-fx-ratekeeper/main/csv_files/SBI_REFERENCE_RATES_USD.csv"

DATE_FORMAT_INPUT = "%Y%m%d"  # For user input dates and internal storage (YYYYMMDD)
DATE_FORMAT_YAHOO_PARSE = "%b %d, %Y"  # For parsing dates like "May 16, 2025" from Yahoo HTML
DATE_FORMAT_SBI_PARSE = '%Y-%m-%d'  # For parsing dates like "2023-05-16" from SBI CSV
DATE_FORMAT_OUTPUT = "%Y%m%d"  # For display dates in the final report


# --- Data Structures ---
class VestingEntry(NamedTuple):
    """Represents a single vesting entry."""
    stock_symbol: str
    acquisition_date_str: str  # Format: YYYYMMDD
    acquisition_dt: datetime.date  # datetime.date object for easier comparison
    num_shares: int


# Type aliases for clarity
StockSymbol = str
DateStringYYYYMMDD = str
Price = float
StockPricesForSymbol = OrderedDict[DateStringYYYYMMDD, Price]
AllStockPrices = Dict[StockSymbol, StockPricesForSymbol]
SbiExchangeRates = OrderedDict[DateStringYYYYMMDD, Price]  # DateString (YYYYMMDD) -> Rate


# --- Input and Validation ---

def parse_vesting_line(line: str) -> Optional[VestingEntry]:
    """
    Parses a single line of vesting input.
    Expected format: "SYMBOL YYYYMMDD SHARES"
    """
    parts = line.split()
    if len(parts) == 3:
        stock_symbol, date_str, shares_str = parts
        try:
            return VestingEntry(
                stock_symbol=stock_symbol.upper(),
                acquisition_date_str=date_str,
                acquisition_dt=datetime.datetime.strptime(date_str, DATE_FORMAT_INPUT).date(),
                num_shares=int(shares_str)
            )
        except ValueError:
            print(f"Warning: Skipping malformed input line (invalid date/shares format): {line}")
            return None
    else:
        print(f"Warning: Skipping malformed input line (expected 3 parts): {line}")
        return None


def import_vesting_data() -> Tuple[List[VestingEntry], int, Dict[StockSymbol, DateStringYYYYMMDD]]:
    """
    Prompts the user for vesting data and the calendar year.
    Returns:
        - A list of VestingEntry objects.
        - The calendar year (int).
        - A dictionary mapping stock symbols to their earliest acquisition date string.
    """
    print("Enter the initial csv data in following format, one line for each vesting:")
    print("Nasdaq Stock Symbol <space> Vesting date in YYYYMMDD <space> No of shares")
    print("Eg: GOOG 20230325 8")
    print("When done just enter a blank line")

    raw_lines: List[str] = []
    while True:
        input_line = input()
        if not input_line:
            break
        raw_lines.append(input_line)

    if not raw_lines:
        sys.exit("No vesting data entered. Exiting.")

    # Sort raw lines primarily for consistent processing if multiple lines for the same
    # stock have different dates and we rely on processing order for lowest_acquisition_dates.
    # However, the explicit min check later is more robust.
    raw_lines.sort()

    vesting_entries: List[VestingEntry] = []
    for line in raw_lines:
        entry = parse_vesting_line(line)
        if entry:
            vesting_entries.append(entry)

    if not vesting_entries:
        sys.exit("No valid vesting data could be parsed. Exiting.")

    lowest_acquisition_dates: Dict[StockSymbol, DateStringYYYYMMDD] = {}
    for entry in vesting_entries:
        if entry.stock_symbol not in lowest_acquisition_dates or \
                entry.acquisition_date_str < lowest_acquisition_dates[entry.stock_symbol]:
            lowest_acquisition_dates[entry.stock_symbol] = entry.acquisition_date_str

    calendar_year_val = 0
    while True:
        try:
            year_str = input(f"Enter calendar year in YYYY format (e.g., {datetime.date.today().year}): ")
            calendar_year_val = int(year_str)
            if len(year_str) == 4:
                break
            else:
                print("Invalid year format. Please use YYYY.")
        except ValueError:
            print("Invalid year. Please enter a numeric year.")

    return vesting_entries, calendar_year_val, lowest_acquisition_dates


def validate_vesting_data(vesting_entries: List[VestingEntry], calendar_year: int) -> None:
    """Validates that acquisition dates are not later than the specified calendar year."""
    for entry in vesting_entries:
        if entry.acquisition_dt.year > calendar_year:
            sys.exit(
                f"Error: Acquisition date {entry.acquisition_date_str} for {entry.stock_symbol} "
                f"is later than the reporting calendar year {calendar_year}. Please fix."
            )


# --- Date Utilities ---
def get_date_obj_from_str(date_str: DateStringYYYYMMDD) -> datetime.date:
    """Converts YYYYMMDD string to datetime.date object."""
    return datetime.datetime.strptime(date_str, DATE_FORMAT_INPUT).date()


# --- Data Fetching ---
def fetch_yahoo_stock_prices_for_symbol(
        stock_symbol: StockSymbol,
        earliest_acq_date_str: DateStringYYYYMMDD,
        calendar_year_to_process: int
) -> StockPricesForSymbol:
    """Fetches historical stock prices for a single symbol from Yahoo Finance."""
    prices_for_symbol: StockPricesForSymbol = OrderedDict()

    # Fetch data up to Jan 15 of the year after the calendar_year
    end_date_dt = datetime.datetime(calendar_year_to_process + 1, 1, 15)
    end_date_ts = int(end_date_dt.timestamp())

    # Determine start date for fetching:
    # Go back ~45 days before the earliest acquisition date for this stock to ensure
    # there's data available for looking up the price on or before the acquisition date.
    earliest_acq_dt = get_date_obj_from_str(earliest_acq_date_str)
    fetch_start_dt_obj = earliest_acq_dt - datetime.timedelta(days=45)  # Approx 1.5 months prior

    start_date_ts = int(datetime.datetime(
        fetch_start_dt_obj.year, fetch_start_dt_obj.month, fetch_start_dt_obj.day
    ).timestamp())

    stock_prices_url = YAHOO_FINANCE_URL_TEMPLATE.format(
        symbol=stock_symbol, start_ts=start_date_ts, end_ts=end_date_ts
    )
    print(f"Fetching stock prices for {stock_symbol} from {stock_prices_url}")

    try:
        req = Request(stock_prices_url, headers={'User-Agent': USER_AGENT})
        html_content = urlopen(req).read()
        pq = PyQuery(html_content)

        history_table_div = pq('div[data-testid="history-table"]')
        if not history_table_div:
            print(f"Warning: Could not find history table container for {stock_symbol}.")
            return prices_for_symbol

        table_element = history_table_div('table')
        if not table_element:
            print(f"Warning: No table found within history div for {stock_symbol}.")
            return prices_for_symbol

        rows = table_element('tbody > tr')

        for row_html in rows.items():
            cells = row_html('td')
            if len(cells) == 7:  # Data rows typically have 7 cells
                try:
                    date_text = cells.eq(0).text()
                    close_price_text = cells.eq(4).text()  # Close* is the 5th column

                    if not date_text or not close_price_text or close_price_text == '-':
                        continue

                    dt_obj = datetime.datetime.strptime(date_text, DATE_FORMAT_YAHOO_PARSE)
                    formatted_date_key = dt_obj.strftime(DATE_FORMAT_INPUT)  # YYYYMMDD

                    price_val = float(close_price_text.replace(',', ''))
                    prices_for_symbol[formatted_date_key] = round(price_val, 2)
                except ValueError:
                    # print(f"Skipping row for {stock_symbol} due to parsing error: {row_html.text()[:100]}")
                    continue

        # Ensure prices are sorted by date (Yahoo usually is, but good practice)
        return OrderedDict(sorted(prices_for_symbol.items()))

    except URLError as e:
        print(f"Warning: Error fetching stock prices for {stock_symbol}: {e}. Returning any fetched data.")
    except Exception as e:
        print(f"Warning: An unexpected error occurred for {stock_symbol} during price fetching: {e}.")
    return prices_for_symbol  # Return whatever was collected, even if partial or empty


def fetch_all_stock_prices(
        symbols_to_fetch: Set[StockSymbol],
        lowest_acq_dates: Dict[StockSymbol, DateStringYYYYMMDD],
        calendar_year_to_process: int
) -> AllStockPrices:
    """Fetches stock prices for all unique symbols."""
    all_prices: AllStockPrices = {}
    for symbol in symbols_to_fetch:
        all_prices[symbol] = fetch_yahoo_stock_prices_for_symbol(
            symbol, lowest_acq_dates[symbol], calendar_year_to_process
        )
    return all_prices


def fetch_sbi_exchange_rates() -> SbiExchangeRates:
    """Fetches SBI USD/INR reference exchange rates and returns them as an OrderedDict."""
    print(f"Fetching SBI exchange rates from {SBI_RATES_URL}")
    sbi_rates_data: SbiExchangeRates = OrderedDict()
    try:
        req = Request(SBI_RATES_URL, headers={'User-Agent': USER_AGENT})
        html_content = urlopen(req).read().decode('utf-8')

        lines = html_content.splitlines()
        if not lines or "Date,TT CATEGORY,Rate (INR) Per UNIT OF FOREIGN CURRENCY" not in lines[0]:
            print("Warning: SBI rates CSV header not as expected. Parsing might fail.")

        for line in lines[1:]:  # Skip header
            parts = line.split(",")
            if len(parts) >= 3:
                date_str_csv = parts[0][:10]  # YYYY-MM-DD
                rate_str = parts[2]
                if rate_str and rate_str.replace('.', '', 1).isdigit() and float(rate_str) != 0.0:
                    try:
                        dt_obj = datetime.datetime.strptime(date_str_csv, DATE_FORMAT_SBI_PARSE)
                        formatted_date_key = dt_obj.strftime(DATE_FORMAT_INPUT)  # YYYYMMDD
                        rate = float(rate_str)
                        sbi_rates_data[formatted_date_key] = round(rate, 2)
                    except ValueError:
                        # print(f"Skipping SBI rate line with invalid date or rate: {line}")
                        continue

    except URLError as e:
        sys.exit(f"Critical Error: Could not fetch SBI rates: {e}. Check URL or network connection.")
    except Exception as e:
        sys.exit(f"Critical Error: An unexpected error occurred during SBI rate fetching: {e}")

    if not sbi_rates_data:
        print("Warning: No SBI rates fetched. Calculations involving INR will be affected.")

    return OrderedDict(sorted(sbi_rates_data.items()))  # Ensure sorted by date


# --- Calculation Logic ---

def get_sbi_rate_on_or_before_date(
        target_date_str: DateStringYYYYMMDD,
        sbi_rates: SbiExchangeRates
) -> Optional[Price]:
    """
    Finds the SBI exchange rate for the given date or the closest prior date.
    Assumes sbi_rates is an OrderedDict sorted by date (YYYYMMDD).
    """
    if not sbi_rates:
        return None

    rate_dates = list(sbi_rates.keys())

    insertion_idx = bisect.bisect_right(rate_dates, target_date_str)

    if insertion_idx == 0:
        # Target date is before the first recorded rate.
        # Original script's behavior was to use the first available if target was before all.
        # If we want to strictly use *on or before*, and target_date is before first, return None.
        # However, to match original behavior if rate_dates is not empty:
        return sbi_rates[rate_dates[0]] if rate_dates else None

    actual_date_key = rate_dates[insertion_idx - 1]
    return sbi_rates[actual_date_key]


def get_stock_price_on_or_before_date(
        prices_for_symbol: StockPricesForSymbol,
        target_date_str: DateStringYYYYMMDD
) -> Tuple[Optional[DateStringYYYYMMDD], Optional[Price]]:
    """
    Finds stock price on target_date_str or closest prior trading day.
    Assumes prices_for_symbol is an OrderedDict sorted by date (YYYYMMDD).
    """
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
        acquisition_dt: datetime.date  # The actual acquisition date as datetime.date
) -> Tuple[Optional[DateStringYYYYMMDD], Optional[Price]]:
    """
    Finds the peak stock price from the later of acquisition date or start of calendar year,
    up to the end of the calendar year.
    """
    if not prices_for_symbol:
        return None, None

    cal_year_start_dt = datetime.date(calendar_year, 1, 1)
    # The search for peak should start from the actual acquisition date,
    # but only consider prices within the specified calendar year.
    search_effective_start_dt = max(acquisition_dt, cal_year_start_dt)
    search_end_dt = datetime.date(calendar_year, 12, 31)

    peak_price_val = -1.0
    peak_date_str: Optional[DateStringYYYYMMDD] = None

    for date_str, price in prices_for_symbol.items():
        current_dt = get_date_obj_from_str(date_str)

        # Only consider prices within the calendar year and on or after the effective start date
        if search_effective_start_dt <= current_dt <= search_end_dt:
            if price > peak_price_val:
                peak_price_val = price
                peak_date_str = date_str

    return peak_date_str, peak_price_val


def find_year_end_closing_stock_info(
        prices_for_symbol: StockPricesForSymbol
) -> Tuple[Optional[DateStringYYYYMMDD], Optional[Price]]:
    """
    Finds the closing stock price on the last available trading day of December
    """
    if not prices_for_symbol:
        return None, None

    last_dec_trade_date_str: Optional[DateStringYYYYMMDD] = None
    last_dec_trade_price: Optional[Price] = None

    for date_str, price in sorted(prices_for_symbol.items(), reverse=True):
        dt_obj = get_date_obj_from_str(date_str)
        if dt_obj.month == 12:
            last_dec_trade_date_str = date_str
            last_dec_trade_price = price
            break

    return last_dec_trade_date_str, last_dec_trade_price


# --- Output ---
def format_output_line(
        entry: VestingEntry,
        acq_price_date_str: Optional[DateStringYYYYMMDD],
        acq_usd_price_val: Optional[Price],
        sbi_rate_on_acq: Optional[Price],
        peak_date_str: Optional[DateStringYYYYMMDD],
        peak_usd_price_val: Optional[Price],
        sbi_rate_on_peak: Optional[Price],
        closing_date_str: Optional[DateStringYYYYMMDD],
        closing_usd_price_val: Optional[Price],
        sbi_rate_on_closing: Optional[Price]
) -> str:
    """Formats a single line for the output report."""

    num_shares = entry.num_shares

    # Share Price on Acquisition Date (INR) and Initial Value
    if acq_usd_price_val is not None and sbi_rate_on_acq is not None:
        share_price_inr_on_acq = round(acq_usd_price_val * sbi_rate_on_acq, 2)
        initial_value_investment_inr = round(num_shares * share_price_inr_on_acq, 2)
        acq_price_display_str = f"{share_price_inr_on_acq}"
        initial_value_display_str = f"{initial_value_investment_inr}"
    else:
        acq_price_display_str = "N/A"
        initial_value_display_str = "N/A"

    # Peak Value
    if peak_date_str and peak_usd_price_val is not None and sbi_rate_on_peak is not None:
        peak_price_inr_per_share = round(peak_usd_price_val * sbi_rate_on_peak, 2)
        total_peak_value_inr = round(num_shares * peak_price_inr_per_share, 2)
        peak_date_formatted = get_date_obj_from_str(peak_date_str).strftime(DATE_FORMAT_OUTPUT)
        peak_output_str = f"{total_peak_value_inr}({peak_date_formatted}, {peak_price_inr_per_share})"
    else:
        peak_output_str = "N/A(N/A, N/A)"

    # Closing Value
    if closing_date_str and closing_usd_price_val is not None and sbi_rate_on_closing is not None:
        closing_price_inr_per_share = round(closing_usd_price_val * sbi_rate_on_closing, 2)
        total_closing_value_inr = round(num_shares * closing_price_inr_per_share, 2)
        closing_date_formatted = get_date_obj_from_str(closing_date_str).strftime(DATE_FORMAT_OUTPUT)
        closing_output_str = f"{total_closing_value_inr}({closing_date_formatted}, {closing_price_inr_per_share})"
    else:
        closing_output_str = "N/A(N/A, N/A)"

    return (
        f"{entry.stock_symbol}\t{entry.num_shares}\t{entry.acquisition_date_str}\t"
        f"{acq_price_display_str}\t{initial_value_display_str}\t"
        f"{peak_output_str}\t{closing_output_str}"
    )


def print_report(output_lines: List[str]) -> None:
    """Prints the formatted results to the console."""
    header = (
        "Nasdaq Symbol\tNum of Shares Acquired\tAcquisition Date\t"
        "Share Price on Acquisition Date (INR)\tInitial value of the investment (INR)\t"
        "Peak value of investment during the period (INR) (Peak Date YYYY/MM/DD, Peak Price/Share INR)\t"
        "Closing balance (INR) (Closing Date YYYY/MM/DD, Close Price/Share INR)"
    )
    print("\n--- Investment Report ---")
    print(header)
    print("-" * len(header))  # Adjust separator length based on header
    for line in output_lines:
        print(line)


# --- Main Execution ---
def main():
    """Main function to orchestrate the script's execution."""
    # In case of certificate error on MAC, Go to Applications Folder, Python and double click 'Install Certificates.command'

    vesting_data_list, calendar_year, lowest_acq_dates_map = import_vesting_data()
    validate_vesting_data(vesting_data_list, calendar_year)

    all_stock_symbols_set = set(entry.stock_symbol for entry in vesting_data_list)

    # Fetch all required data
    all_stock_prices_data = fetch_all_stock_prices(
        all_stock_symbols_set, lowest_acq_dates_map, calendar_year
    )
    sbi_rates_data = fetch_sbi_exchange_rates()

    report_lines: List[str] = []

    for entry in vesting_data_list:
        symbol = entry.stock_symbol
        acq_date_str = entry.acquisition_date_str
        acq_dt = entry.acquisition_dt

        current_stock_prices = all_stock_prices_data.get(symbol, OrderedDict())

        if not current_stock_prices:
            print(
                f"Warning: No stock price data available for {symbol}. Calculations for this entry will be incomplete.")
            report_lines.append(
                f"{symbol}\t{entry.num_shares}\t{acq_date_str}\tN/A\tN/A\tN/A(N/A, N/A)\tN/A(N/A, N/A)"
            )
            continue

        # Price on (or just before) acquisition
        price_date_for_acq_str, usd_price_on_acq = get_stock_price_on_or_before_date(
            current_stock_prices, acq_date_str
        )
        sbi_rate_for_acq = get_sbi_rate_on_or_before_date(
            price_date_for_acq_str, sbi_rates_data
        ) if price_date_for_acq_str else None

        # Peak Value Calculation
        peak_date_str, peak_usd_price = find_peak_stock_info(
            current_stock_prices, calendar_year, acq_dt
        )
        sbi_rate_for_peak = get_sbi_rate_on_or_before_date(
            peak_date_str, sbi_rates_data
        ) if peak_date_str else None

        # Closing Value Calculation
        closing_date_str, closing_usd_price = find_year_end_closing_stock_info(
            current_stock_prices
        )
        sbi_rate_for_closing = get_sbi_rate_on_or_before_date(
            closing_date_str, sbi_rates_data
        ) if closing_date_str else None

        report_lines.append(format_output_line(
            entry,
            price_date_for_acq_str, usd_price_on_acq, sbi_rate_for_acq,
            peak_date_str, peak_usd_price, sbi_rate_for_peak,
            closing_date_str, closing_usd_price, sbi_rate_for_closing
        ))

    print_report(report_lines)


if __name__ == "__main__":
    main()
