from sqlalchemy import create_engine
import pandas as pd

def get_df():
    connection_line = 'mysql+pymysql://root:@localhost/datasets'

    engine = create_engine(connection_line)

    q = 'select * from `cars`'

    return pd.read_sql(q, engine)

if __name__ == '__main__':
    df = get_df()
    print(df.head())