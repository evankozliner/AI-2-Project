""" 
    This class extracts balanace sheet data from the ScraXBRL project (which has been slightly modified to do
    more than print fundamentals data)
    
    It's most important feature is its get_all_data method which returns a pandas dataframe containing stock data 
    imposed with balance sheet data. It's important to note that each stock price value is joined with 
    it's *previous* balance sheet information (this is because hypothetically the previous balance sheet
    dictates the price).
    
    Evan Kozliner 
"""


import bt
import os
import numpy as np
import ScraXBRL.DataViewer as dv
import datetime as dt
import pandas as pd

DEFAULT_BALANCE_SHEET_ITEMS = [
        'CashAndCashEquivalentsAtCarryingValue', 
        'Assets', 
        'LiabilitiesCurrent']

class BalanceSheetDataExtractor:

    # Currently needs to be supplied a start date that exists in the 10-K or 
    def __init__(self, symbol, start_date, 
            balance_sheet_items = DEFAULT_BALANCE_SHEET_ITEMS,
            data_path = 'ScraXBRL/data/extracted_data/{0}/'):
        self.symbol = symbol
        self.balance_sheet_items = balance_sheet_items
        self.last_balance_sheet_date = start_date
        # Start date in format YYYY-MM-DD
        self.date_format = '%Y-%m-%d'
        self.data_path = data_path
        self.stock_data = bt.get('{0}:Open,{0}:High,{0}:Low,{0}:Close'.format(symbol), 
                start = start_date)
        self.balance_sheet_data = self.get_balance_sheet_data(self.last_balance_sheet_date)
        #self.data = self.join_quarterly_finances_to_stock_data()

    def get_balance_sheet_data(self, start_date):
        # Builds an array of tuples, reports, containing (date, report_type)
        # and an array of dicts, reports_data, in the same order as reports containing the 
        # balance_sheet_data items as keys with their respective (scraped) values 

        self.reports = self.get_reports_after_date(start_date)
        report_data = []

        for report_date, report_type in self.reports:
            data_view = dv.DataView(self.symbol, 
                    report_date.strftime(self.date_format), 
                    report_type, 
                    start_path='ScraXBRL/')
            balance_sheet_values = {}
            for fact in self.balance_sheet_items:
                facts_listed = data_view.get_balance_sheet_value(fact)
                # Balance sheet lists values for years other than one in quesiton, this removes
                # those duplicates
                for year in facts_listed.keys():
                    if year == report_date.strftime(self.date_format):
                        balance_sheet_values[fact] = facts_listed[year]
            report_data.append(balance_sheet_values)

        # Sort by the date of the report and join the data into a larger vector
        return self.to_dataframe(sorted(zip(self.reports, report_data), key=lambda x: x[0][0]))

    def to_dataframe(self,data):
        dates = [x[0][0] for x in data]
        final_df = pd.DataFrame()

        for item in self.balance_sheet_items:
            s = pd.Series([x[1][item] for x in data], index=dates, name=item)
            final_df = pd.concat([final_df, s], axis=1)

        return final_df

    def get_reports_after_date(self, start_date):
        """ Returns all reports after the given start date in [(date, type)...]"""
        all_reports = self.get_reports("10-K") + self.get_reports("10-Q")
        start_date = dt.datetime.strptime(start_date, self.date_format)
        return filter(lambda report: report[0] >= start_date, all_reports) 

    def get_reports(self, report_type):
        reports = []
        path = self.data_path.format(self.symbol) + report_type + '/xml'
        for report in os.listdir(path):
            reports.append((dt.datetime.strptime(report, self.date_format), report_type))
        return reports

    def get_all_data(self):
        """ Builds a pandas dataframe indexed by date with stock prices as well as the fundamentals
            prior to the stock prices as rows
        """
        data = self.stock_data.copy()

        for fact in self.balance_sheet_items:
            fact_vector = self.build_fact_vector(fact)
            data = pd.concat([data, fact_vector], axis=1)

        return data

    def build_fact_vector(self, fact):
        fact_data = self.balance_sheet_data[fact]
        ordered_facts = [self.get_fact_for_time(fact, time) for time in self.stock_data.index]
        return pd.Series(ordered_facts, name=fact, index=self.stock_data.index)

    def get_fact_for_time(self, fact, time):
        dates = self.balance_sheet_data.index
        balance_sheet_time = np.max(dates[np.where(pd.to_datetime(time) >= dates)])
        return self.balance_sheet_data[fact][balance_sheet_time]
