# Index Futures Pair Trading Strategy Backtesting

## Project Structure

The files in this project are the following:
- pairtrading.py: python script that was asked to develop
- N_Buchwalder_Summary.pdf: a written summary of my research
- N_Buchwalder_Research.ipynb: a cleaned up and commented notebook that highlights all my research
- N_Buchwalder_Research.html: a html copy of the notebook if the notebook can't be run
- Backtest_Dashboard.pdf: screenshot of one of the dashboards if dash doesn't work for you
- requirements.txt: the python modules requirements

## Main Python Script

This script processes a given CSV file containing index futures data to backtest a pair trading mean reversion strategy. It allows for the visualization of backtest results through a Dash dashboard and supports including or excluding transaction costs from the backtest.

### Requirements
Before running the script, ensure that you have Python 3.12 installed, create a virtual environment and installed required packages as follows:

```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt

```

### Usage
Run the script from the command line, specifying the CSV file to be processed. There are options to exclude transaction costs from the backtest and to set a custom port for the Dash dashboard:

```
python3 pairtrading.py csv_file [--exclude_tcosts] [--dash_port PORT]
```
Arguments
csv_file: Required. The path to the CSV file containing the data for backtesting.
--exclude_tcosts: Optional. Exclude transaction costs from the backtest. By default, transaction costs are included.
--dash_port: Optional. Specify the port for the Dash dashboard. The default port is 8050.

At the end of the script, Dash will give a localhost address. Please paste that address into your browser to access to interactive dashboard.
