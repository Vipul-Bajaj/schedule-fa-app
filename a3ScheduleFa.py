# In case of certificate error on MAC, Go to Applications Folder, Python and double click 'Install Certificates.command'
import datetime
import sys
from collections import OrderedDict
from urllib.request import Request, urlopen

from pyquery import PyQuery

stock_symbols = set()
input_list = list()
calendar_year = datetime.date.today().year
stock_prices_dict = dict()
sbi_ttbr_rates = OrderedDict()
lowest_acquisition_date_for_stock_symbols = dict()


def import_initial_data():
    global calendar_year, input_list, stock_symbols
    print("Enter the initial csv data in following format, one line for each vesting:")
    print("Nasdaq Stock Symbol <space> Vesting date in YYYYMMDD <space> No of shares <space>")
    print("Eg: GOOG 20230325 8")
    print("When done just enter a blank line")

    while True:
        input_line = input()
        if len(input_line) == 0:
            break
        input_list.append(input_line)

    input_list = sorted(input_list)

    for input_line in input_list:
        stock_symbol, acquisition_date, num_of_shares = input_line.split(" ")
        stock_symbol = stock_symbol.upper()
        if stock_symbol not in lowest_acquisition_date_for_stock_symbols:
            lowest_acquisition_date_for_stock_symbols[stock_symbol] = acquisition_date
        else:
            lowest_acquisition_date_for_stock_symbols[stock_symbol] = min(acquisition_date,
                                                                          lowest_acquisition_date_for_stock_symbols[
                                                                              stock_symbol])
        stock_symbols.add(stock_symbol)

    calendar_year = int(input("Enter calendar year in yyyy for eg 2023: "))


def validate_input():
    for input_line in input_list:
        date = input_line.split(" ")[1]
        if int(date[:4]) > int(calendar_year):
            sys.exit("Acquire date is more than calendar year. Please fix.")


def fetch_stock_price():
    global stock_prices_dict
    end_date = datetime.datetime(calendar_year + 1, 1, 15, 0, 0).timestamp()
    for stock in stock_symbols:
        start_date = get_date_obj(lowest_acquisition_date_for_stock_symbols[stock])
        new_start_date = start_date
        if start_date.day < 16:
            if start_date.month < 2:
                new_start_date = datetime.date(start_date.year - 1, 12, 16)
            else:
                new_start_date = datetime.date(start_date.year, start_date.month - 1, 15)
        else:
            new_start_date = datetime.date(start_date.year, start_date.month, max(start_date.day - 15, 1))
        start_date = datetime.datetime(new_start_date.year, new_start_date.month, new_start_date.day, 0, 0).timestamp()
        stock_prices_dict[stock] = OrderedDict()
        stock_prices_url = "https://finance.yahoo.com/quote/{0}/history?period1={1}&period2={2}&interval=1d&filter=history&frequency=1d".format(
            stock, start_date, end_date)
        print("Fetching stock prices from " + stock_prices_url)
        req = Request(stock_prices_url, headers={'User-Agent': 'Mozilla/5.0'})
        html = urlopen(req).read()
        pq = PyQuery(html)
        # tag = pq('tr')
        # volume_found = 0
        # col_no = 0
        # for line in tag.text().split('\n'):
        #     print(line)
        #     if "Volume" in line:
        #         volume_found = 1
        #     if volume_found:
        #         if col_no == 0:
        #             raw_data = line.split(" ")
        #             if len(raw_data) < 4:
        #                 break
        #             formatted_date = raw_data[3] + '-' + raw_data[1] + '-' + raw_data[2]
        #             date_obj = datetime.datetime.strptime(formatted_date, '%Y-%b-%d,')
        #             date = date_obj.strftime('%Y%m%d')
        #         if col_no == 4:
        #             closing_price = line
        #             stock_prices_dict[stock][date] = round(float(closing_price), 2)
        #         if col_no == 5:
        #             col_no = -1
        #         col_no = col_no + 1
        history_table_div = pq('div[data-testid="history-table"]')
        table = history_table_div('table')

        rows = table('tbody > tr')

        # print(f"Found {len(rows)} rows in the table.")

        for row_html in rows.items():
            cells = row_html('td')

            # Data rows should have 7 cells: Date, Open, High, Low, Close*, Adj Close**, Volume
            # Dividend rows or other special rows might have fewer cells (e.g., colspan)
            if len(cells) == 7:
                try:
                    date_str = cells.eq(0).text()
                    close_price_str = cells.eq(4).text()  # Close* is the 5th column (index 4)

                    # Skip rows where date or price might be placeholders or comments
                    if not date_str or not close_price_str or close_price_str == '-':
                        # print(f"Skipping row with missing data: Date='{date_str}', Close='{close_price_str}'")
                        continue

                    # Parse date (e.g., "May 16, 2025")
                    # Some dates might be just "Dividend" or other text, strptime will fail
                    dt_obj = datetime.datetime.strptime(date_str, "%b %d, %Y")
                    formatted_date_key = dt_obj.strftime("%Y%m%d")

                    # Parse closing price
                    price_float = float(close_price_str.replace(',', ''))

                    stock_prices_dict[stock][formatted_date_key] = round(price_float,2)
                except ValueError as ve:
                    # This can happen if a cell doesn't contain what's expected (e.g. "Dividend" in date cell, or non-numeric price)
                    # print(f"Skipping row due to parsing error: {ve}. Row text: {row_html.text()[:100]}")
                    continue
                except Exception as e:
                    # print(f"An unexpected error occurred for a row: {e}. Row text: {row_html.text()[:100]}")
                    continue
            # else:
            # print(f"Skipping row with {len(cells)} cells (expected 7 for data rows): {row_html.text()[:100]}")


def find_closing_date(stock_symbol):
    # return last stock price in 12th month(December)
    for date in stock_prices_dict[stock_symbol].keys():
        month = get_date_obj(date).month
        if month == 12:
            return date


def find_peak_date(stock_symbol, acquisition_date):
    peak_value = -1
    peak_date = -1
    if get_date_obj(acquisition_date).year < calendar_year:
        acquisition_date = str(datetime.date(calendar_year, 1, 1)).replace("-", "")
    for date, price in stock_prices_dict[stock_symbol].items():
        date_obj = get_date_obj(date)
        if peak_date == -1 and date_obj.year == calendar_year:
            peak_value = price
            peak_date = date
            if date <= acquisition_date:
                break
        elif peak_date != -1 and acquisition_date <= date:
            if peak_value < price:
                peak_value = stock_prices_dict[stock_symbol][date]
                peak_date = date
            if date == acquisition_date:
                break
        elif peak_date != -1 and acquisition_date > date:
            if peak_value < price:
                peak_value = stock_prices_dict[stock_symbol][date]
                peak_date = date
            break
        if peak_date != -1 and date_obj.year != calendar_year:
            break
    return peak_date


def find_stock_price_and_last_date_before_acquisition_date(stock_symbol, acquisition_date):
    for date, price in stock_prices_dict[stock_symbol].items():
        if date >= acquisition_date:
            continue
        return date, price


def fetch_sbi_rates():
    global sbi_ttbr_rates
    sbi_rates_url = "https://raw.githubusercontent.com/sahilgupta/sbi-fx-ratekeeper/main/csv_files/SBI_REFERENCE_RATES_USD.csv"
    print("Fetching sbi rates from " + sbi_rates_url)
    req = Request(sbi_rates_url, headers={'User-Agent': 'Mozilla/5.0'})
    html = urlopen(req).read()
    for line in html.decode().split("\n"):
        sbi_rate = line.split(",")
        if len(sbi_rate[0]) >= 10 and sbi_rate[2] != "0.00" and sbi_rate[2] != "0":
            sbi_ttbr_rates[(sbi_rate[0][:10]).replace("-", "")] = round(float(sbi_rate[2]), 2)


def get_sbi_rate_for_date(date_for_which_rate_is_required):
    for date, price in reversed(sbi_ttbr_rates.items()):
        if date > date_for_which_rate_is_required:
            continue
        return sbi_ttbr_rates[date]


def get_date_obj(date_str):
    return datetime.datetime.strptime(date_str, '%Y%m%d')


def export_data(output_list):
    print(
        "Nasdaq Symbol\tNum of Shares Acquired\tAcquisition Date\tShare Price on Acquisition Date\tInitial value of the investment\tPeak value of investment during the period(Peak Date & Price)\tClosing balance(Closing Date & Price)")
    for output_line in output_list:
        print(output_line)


if __name__ == "__main__":
    import_initial_data()
    validate_input()
    fetch_stock_price()
    fetch_sbi_rates()

    output_list = []
    for input_line in input_list:
        stock_symbol, acquisition_date, num_of_shares = input_line.split(" ")
        stock_symbol = stock_symbol.upper()
        # print(stock_prices_dict[stock_symbol])
        last_date_before_acquisition_date, stock_price_on_last_date_before_vesting = find_stock_price_and_last_date_before_acquisition_date(
            stock_symbol, acquisition_date)
        peak_date_of_stock_price = find_peak_date(stock_symbol, acquisition_date)
        stock_price_on_peak_date = float(stock_prices_dict[stock_symbol][peak_date_of_stock_price])
        closing_date_of_stock_price = find_closing_date(stock_symbol)
        stock_price_on_closing_date = float(stock_prices_dict[stock_symbol][closing_date_of_stock_price])

        input_values = input_line.split(' ')
        initial_value_of_investment = round(float(num_of_shares) * float(
            stock_price_on_last_date_before_vesting) * float(get_sbi_rate_for_date(last_date_before_acquisition_date)),
                                            2)
        peak_value_of_investment = round(float(num_of_shares) * stock_price_on_peak_date * float(
            get_sbi_rate_for_date(peak_date_of_stock_price)), 2)
        closing_value_of_investment = round(float(num_of_shares) * stock_price_on_closing_date * float(
            get_sbi_rate_for_date(closing_date_of_stock_price)), 2)

        output_line = ("{0}\t{1}\t{2}\t{3}\t{4}\t{5}({6}, {7})"
                       "\t{8}({9}, {10})").format(stock_symbol, num_of_shares,
                                                  acquisition_date,
                                                  round(float(stock_price_on_last_date_before_vesting) * float(
                                                      get_sbi_rate_for_date(last_date_before_acquisition_date)), 2),
                                                  initial_value_of_investment,
                                                  peak_value_of_investment,
                                                  peak_date_of_stock_price,
                                                  round(float(stock_price_on_peak_date) * float(
                                                      get_sbi_rate_for_date(peak_date_of_stock_price)), 2),
                                                  closing_value_of_investment,
                                                  closing_date_of_stock_price,
                                                  round(float(stock_price_on_closing_date) * float(
                                                      get_sbi_rate_for_date(closing_date_of_stock_price)), 2))

        output_list.append(output_line)

    export_data(output_list)
