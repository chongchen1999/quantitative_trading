import yfinance as yf
import pandas as pd
import os

def download_us_indices_data(start_date, end_date, output_dir="historical_data"):
    """
    下载美国主要股指（标普500、纳斯达克100、道琼斯工业平均指数）的历史数据并保存到CSV文件。

    Args:
        start_date (str): 数据下载的开始日期，格式为 'YYYY-MM-DD'。
        end_date (str): 数据下载的结束日期，格式为 'YYYY-MM-DD'。
        output_dir (str): 保存CSV文件的目录。
    """

    # 美股主要指数的股票代码
    indices_tickers = {
        "S&P_500": "^GSPC",
        "NASDAQ_100": "^NDX",
        "Dow_Jones_Industrial_Average": "^DJI"
    }

    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建目录: {output_dir}")

    print(f"正在下载 {start_date} 到 {end_date} 期间的数据...")

    for name, ticker in indices_tickers.items():
        try:
            print(f"正在下载 {name} ({ticker})...")
            # 使用 yfinance 下载数据
            data = yf.download(ticker, start=start_date, end=end_date)

            if not data.empty:
                file_path = os.path.join(output_dir, f"{name}_historical_data.csv")
                data.to_csv(file_path)
                print(f"'{name}' 数据已成功保存到: {file_path}")
            else:
                print(f"未找到 '{name}' ({ticker}) 在指定日期范围内的数据。")

        except Exception as e:
            print(f"下载 '{name}' ({ticker}) 时发生错误: {e}")
    print("所有指数数据下载完成。")

if __name__ == "__main__":
    # 设置下载数据的日期范围
    # 您可以根据需要修改这些日期
    start_date = "1980-01-01"
    # 由于今天是2025年7月9日，我们将结束日期设置为第二天以获取到今天为止的完整数据
    end_date = "2025-07-10"

    download_us_indices_data(start_date, end_date)