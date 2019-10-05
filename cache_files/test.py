import collections
import datetime
import logging
import sklearn
import dateutil.relativedelta
import pandas as pd
from pandas.io.json import json_normalize


def json_parser(consumer_bureau):
    try:
        infodata = collections.defaultdict(lambda: "0")
        infodata.update(consumer_bureau)

        total = infodata['summaryInfo']
        total_df = json_normalize(total['total'])

        monthlydetails = json_normalize(infodata['monthlyDetails'])

        try:
            statement_details = infodata['statementdetails'][0]['statementAccounts']

            latest_statement_dates_list = []
            for latest_statement in statement_details:
                latest_statement_dates_list.append(latest_statement.get('xnsEndDate', ''))
            latest_statement_date = max(latest_statement_dates_list)
        except:
            latest_statement_date = ''

        try:
            # statementStatus flag
            statement_details_status = infodata['statementdetails']
            statement_status = 'N/A'
            for status_dict in statement_details_status:
                status = status_dict.get('statementStatus',0)
                if status:
                    statement_status = status
                    break
        except Exception as e:
            statement_status = 'N/A'


        Average_Lowest_ABB = infodata['MonthlySummarydetails'][0].get('lowestABB', 0)
        annualizedBankingCredit = infodata['MonthlySummarydetails'][0].get('annualizedBankingCredit', 0)

        if infodata['inwEMIBounceXns'] == '0':
            Nil_EMI_Bounce = 'N/A'
        else:
            Nil_EMI_Bounce = len(infodata['inwEMIBounceXns'])

        Account_No = total['accNo']

        fullMonthCount = total['fullMonthCount']

        customerInfo = infodata['customerInfo']

        Bank_name = customerInfo['bank']

        AdditionalMonthlyDetails = infodata['AdditionalMonthlyDetails']
        amd_dataframe = pd.DataFrame(AdditionalMonthlyDetails)

        if 'percentagePeakUtilization' in amd_dataframe.columns:
            percentagePeakUtilization = amd_dataframe['percentagePeakUtilization']
        else:
            percentagePeakUtilization = 'N/A'

        Lowest_ABB = round(float(amd_dataframe['ABB'].min()), 4)

        avg_abb = round(float(amd_dataframe['ABB'].mean()), 4)

        eod_balance = infodata['eODBalances']
        eod_balance_df = pd.DataFrame(eod_balance)

        add_cust_info = infodata['AdditionalCustomerInfo']
        cust_facility = add_cust_info[0]['facility']

        bouncedOrPenalXns = infodata['bouncedOrPenalXns']

        xns = infodata['accountXns'][0]['xns']
        df = pd.DataFrame(xns)

        df['date'] = pd.to_datetime(df['date'])
        df['yr_month'] = df['date'].apply(lambda x: x.strftime('%Y-%m'))

        today = datetime.date.today()
        last_month = today + dateutil.relativedelta.relativedelta(months=-1)
        string = last_month.isoformat()[:7]
        loan_last_month = df[(df.category == "Loan") & (df.yr_month.str.startswith(string))]['amount'].tolist()
        dataframes = {'total_df': total_df, 'Account_No': Account_No, 'Bank_name': Bank_name,
                  'fullMonthCount': fullMonthCount, 'eod_balance': eod_balance_df,
                  'percentagePeakUtilization': percentagePeakUtilization, 'cust_facility': cust_facility,
                  'bouncedOrPenalXns': bouncedOrPenalXns, 'loan_last_month': loan_last_month,
                  'Nil_EMI_Bounce': Nil_EMI_Bounce, 'latest_statement_date': latest_statement_date,
                  'Average_Lowest_ABB': Average_Lowest_ABB, 'annualizedBankingCredit': annualizedBankingCredit,
                  'Lowest_ABB': Lowest_ABB, 'avg_abb': avg_abb, 'statement_status': statement_status, 'monthlydetails': monthlydetails
                  }
        return dataframes

    except KeyError as error:
        logging.error(error)
        return str(error)
    except Exception as exception:
        logging.error(exception)
        return str(exception)




