import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Dict

def calculate_annualized_return(start_price: float, end_price: float, years: float) -> float:
    """
    计算年化收益率
    """
    return (pow(end_price / start_price, 1 / years) - 1) * 100

def analyze_buy_and_hold(file_path: str, fund_name: str) -> Dict:
    """
    分析单个基金的买入并持有策略
    """
    # 读取数据
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # 计算每天的平均价格
    df['Avg_Price'] = (df['High'] + df['Low']) / 2
    
    # 持有期列表（年）
    holding_periods = [1, 3, 5, 10, 15, 20, 30]
    
    results = {}
    
    for period in holding_periods:
        period_results = []
        
        # 遍历所有可能的买入时间
        for i in range(len(df)):
            buy_date = df.iloc[i]['Date']
            buy_price = df.iloc[i]['Avg_Price']
            
            # 计算卖出日期（期望日期）
            target_sell_date = buy_date + pd.DateOffset(years=period)
            
            # 找到最接近期望卖出日期的实际交易日
            future_dates = df[df['Date'] >= target_sell_date]
            if len(future_dates) == 0:
                continue
                
            sell_idx = future_dates.index[0]
            sell_date = df.loc[sell_idx, 'Date']
            sell_price = df.loc[sell_idx, 'Avg_Price']
            
            # 计算实际持有年数
            actual_years = (sell_date - buy_date).days / 365.25
            
            # 计算年化收益率
            annualized_return = calculate_annualized_return(buy_price, sell_price, actual_years)
            
            period_results.append({
                'buy_date': buy_date,
                'sell_date': sell_date,
                'buy_price': buy_price,
                'sell_price': sell_price,
                'annualized_return': annualized_return,
                'actual_years': actual_years
            })
        
        # 按年化收益率排序
        period_results.sort(key=lambda x: x['annualized_return'], reverse=True)
        
        # 保存结果
        results[period] = {
            'all_results': period_results,
            'top_5': period_results[:5] if len(period_results) >= 5 else period_results,
            'bottom_5': period_results[-5:] if len(period_results) >= 5 else period_results
        }
    
    return results

def print_results(fund_name: str, results: Dict):
    """
    打印分析结果
    """
    print(f"\n{'='*80}")
    print(f"{fund_name} - Buy & Hold Strategy Analysis")
    print(f"{'='*80}")
    
    for period, data in results.items():
        if len(data['all_results']) == 0:
            print(f"\n{period}年持有期：没有足够的数据")
            continue
            
        print(f"\n{period}年持有期分析：")
        print(f"总共分析了 {len(data['all_results'])} 个买入时间点")
        
        # 计算统计数据
        returns = [r['annualized_return'] for r in data['all_results']]
        avg_return = np.mean(returns)
        median_return = np.median(returns)
        std_return = np.std(returns)
        
        print(f"\n统计数据：")
        print(f"  平均年化收益率: {avg_return:.2f}%")
        print(f"  中位数年化收益率: {median_return:.2f}%")
        print(f"  标准差: {std_return:.2f}%")
        
        # 打印最好的5个时间段
        print(f"\n最佳5个时间段：")
        print(f"{'买入日期':<12} {'卖出日期':<12} {'买入价格':<10} {'卖出价格':<10} {'年化收益率':<12}")
        print("-" * 60)
        for result in data['top_5']:
            print(f"{result['buy_date'].strftime('%Y-%m-%d'):<12} "
                  f"{result['sell_date'].strftime('%Y-%m-%d'):<12} "
                  f"{result['buy_price']:<10.2f} "
                  f"{result['sell_price']:<10.2f} "
                  f"{result['annualized_return']:<12.2f}%")
        
        # 打印最差的5个时间段
        print(f"\n最差5个时间段：")
        print(f"{'买入日期':<12} {'卖出日期':<12} {'买入价格':<10} {'卖出价格':<10} {'年化收益率':<12}")
        print("-" * 60)
        for result in data['bottom_5']:
            print(f"{result['buy_date'].strftime('%Y-%m-%d'):<12} "
                  f"{result['sell_date'].strftime('%Y-%m-%d'):<12} "
                  f"{result['buy_price']:<10.2f} "
                  f"{result['sell_price']:<10.2f} "
                  f"{result['annualized_return']:<12.2f}%")

def main():
    """
    主函数
    """
    # 文件路径和名称
    files = [
        ('Dow_Jones_Industrial_Average_historical_data.csv', 'Dow Jones Industrial Average'),
        ('NASDAQ_100_historical_data.csv', 'NASDAQ 100'),
        ('S&P_500_historical_data.csv', 'S&P 500')
    ]
    
    # 分析每个基金
    for file_path, fund_name in files:
        try:
            results = analyze_buy_and_hold(file_path, fund_name)
            print_results(fund_name, results)
        except Exception as e:
            print(f"\n处理 {fund_name} 时出错: {str(e)}")
            continue

    # 创建综合比较图表（可选）
    print("\n\n分析完成！")

if __name__ == "__main__":
    main()