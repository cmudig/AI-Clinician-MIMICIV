import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import argparse
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from google_auth_oauthlib import flow
from google.cloud import bigquery

from preprocessing.columns import RAW_DATA_COLUMNS

file_list = [
    'culture',
    'microbio',
    'abx',
    'demog',
    'ce',
    'labs_ce',
    'labs_le',
    'uo',
    'preadm_uo',
    'fluid_mv',
    'preadm_fluid',
    'vaso_mv',
    'mechvent',
    'mechvent_pe'
]

SQL_DIR = os.path.join(os.path.dirname(__file__), 'sql')

def load_data(bq_client, elixhauser_table, file_name, output_dir, skip_if_present=False):
    """
    Loads data from BigQuery using the SQL query given by the file name. For
    ce (chartevents), loads data in 10 chunks to save time.
    """

    with open(os.path.join(SQL_DIR, file_name + '.sql'), 'r') as f:
        query = f.read()

    if file_name == 'ce':
        # Read in batches of size id_step going up to id_max, where the
        # actual ID numbers are offset by id_conversion
        id_step = int(1e6)
        id_max = int(1e7)
        id_conversion = int(3e7)
        for i in range(0, id_max, id_step):
            out_path = os.path.join(output_dir, file_name + str(i) + str(i + id_step) + '.csv')
            if skip_if_present and os.path.exists(out_path):
                print('file exists, skipping')
                return
            
            query_ = query.format(id_conversion + i, id_conversion + id_step + i)
            query_result = bq_client.query(query_)

            result = pd.DataFrame([dict(zip(range(len(result)), result))
                                   for result in tqdm(query_result, desc=file_name + str(i) + str(i + id_step))])
            result.columns = RAW_DATA_COLUMNS['ce']
            result.to_csv(out_path, index=False)
        return
    
    out_path = os.path.join(output_dir, file_name + '.csv')
    if skip_if_present and os.path.exists(out_path):
        print('file exists, skipping')
        return
    
    if file_name == 'demog':
        # Sub in the elixhauser table name
        query = query.format(elixhauser_table)
    query_result = bq_client.query(query)

    result = pd.DataFrame([dict(zip(range(len(result)), result))
                           for result in tqdm(query_result, desc=file_name)])
    result.columns = RAW_DATA_COLUMNS[file_name]
    result.to_csv(out_path, index=False)

def main():
    global bqclient

    parser = argparse.ArgumentParser(description=('Loads data from BigQuery and '
        'saves them as local CSV files.\n\nBefore running this script, be sure '
        'to create an OAuth client to use BigQuery, and download the client '
        'secret file as client_secret.json in the directory containing this '
        'script. Also, you will need to generate the elixhauser_quan table for '
        'MIMIC-IV by copying the contents of elixhauser.sql into the BigQuery '
        'editor, and saving the results to a table in your Google account. Then, '
        'pass the identifier of the table (e.g. `project-name.dataset_name.table_name`) '
        'as the second parameter to this script.'))
    parser.add_argument('secret', type=str, help='Path to BigQuery client secret')
    parser.add_argument('gcp_project', type=str, help='Name of the project within GCP that will be authenticated')
    parser.add_argument('elixhauser', type=str, help='BigQuery table identifier for the elixhauser_quan table (pre-generate this using the sql/elixhauser.sql script)')
    parser.add_argument('--out', dest='output_dir', type=str, default=None,
                        help='Directory in which to output (default is data directory)')
    parser.add_argument('--skip-existing', dest='skip_existing', action='store_true', default=False,
                        help='If passed, skip existing CSV files')
    parser.add_argument('--auth-console', dest='launch_browser', action='store_false', default=True,
                        help='If passed, auth will be performed in the console instead of the browser')

    args = parser.parse_args()
    assert os.path.exists(args.secret), "Please create an OAuth client for BigQuery and place the client_secret.json file in the directory containing this script."
    appflow = flow.InstalledAppFlow.from_client_secrets_file(
        args.secret, scopes=["https://www.googleapis.com/auth/bigquery"]
    )

    if args.launch_browser:
        appflow.run_local_server()
    else:
        appflow.run_console()
    project = args.gcp_project
    credentials = appflow.credentials
    bq_client = bigquery.Client(project=project, credentials=credentials)

    out_dir = args.output_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data', 'raw_data')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for i, file_name in enumerate(file_list):
        load_data(bq_client, args.elixhauser, file_name, out_dir, skip_if_present=args.skip_existing)
        print(str(i + 1) + '/' + str(len(file_list)))

if __name__ == '__main__':
    main()