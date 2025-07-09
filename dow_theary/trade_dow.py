import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager

# 设置中文字体（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class TradingBacktest:
    def __init__(self, initial_capital=1000000.0, commission_rate=0.001):
        """
        初始化回测系统
        
        参数:
        initial_capital: 初始资金（默认100美元）
        commission_rate: 交易手续费率（默认0.1%）
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        
    def load_data(self, price_file, signal_file):
        """加载价格数据和交易信号"""
        # 加载价格数据
        price_df = pd.read_csv(price_file)
        
        # 清理列名
        price_df.columns = [col.strip() for col in price_df.columns]
        if 'Date' in price_df.columns and len(price_df.columns) > 6 and price_df.columns[1] == '':
            price_df = price_df.drop(price_df.columns[1], axis=1)
        
        price_df['Date'] = pd.to_datetime(price_df['Date'])
        
        # 计算每日平均价格
        price_df['Avg_Price'] = (price_df['High'] + price_df['Low'] + price_df['Close']) / 3
        
        # 加载交易信号
        signal_df = pd.read_csv(signal_file)
        signal_df['Date'] = pd.to_datetime(signal_df['Date'])
        
        # 合并数据
        df = pd.merge(price_df, signal_df, on='Date', how='left')
        df = df.sort_values('Date')
        df = df.reset_index(drop=True)
        
        return df
    
    def simulate_trading(self, df, index_name):
        """模拟交易过程"""
        # 初始化变量
        cash = self.initial_capital
        shares = 0
        portfolio_value = []
        cash_history = []
        shares_history = []
        trades = []
        
        # 记录每日状态
        for i in range(len(df)):
            date = df.iloc[i]['Date']
            price = df.iloc[i]['Avg_Price']
            signal = df.iloc[i]['Signal'] if pd.notna(df.iloc[i]['Signal']) else 'HOLD'

            # print(f"Date: {date}, Price: {price}, Signal: {signal}")
            
            # 执行交易
            if signal == 'BUY' and shares == 0:  # 买入（只在空仓时买入）
                # 计算可买入的股数
                max_shares = cash / (price * (1 + self.commission_rate))
                if max_shares >= 1:
                    shares = int(max_shares)
                    cost = shares * price * (1 + self.commission_rate)
                    cash -= cost
                    trades.append({
                        'Date': date,
                        'Type': 'BUY',
                        'Price': price,
                        'Shares': shares,
                        'Cost': cost,
                        'Cash_After': cash
                    })
                    
            elif signal == 'SELL' and shares > 0:  # 卖出（只在持仓时卖出）
                revenue = shares * price * (1 - self.commission_rate)
                cash += revenue
                trades.append({
                    'Date': date,
                    'Type': 'SELL',
                    'Price': price,
                    'Shares': shares,
                    'Revenue': revenue,
                    'Cash_After': cash
                })
                shares = 0
            
            # 计算当前组合价值
            current_value = cash + shares * price
            portfolio_value.append(current_value)
            cash_history.append(cash)
            shares_history.append(shares)
        
        # 添加结果到数据框
        df['Portfolio_Value'] = portfolio_value
        df['Cash'] = cash_history
        df['Shares'] = shares_history
        df['Position'] = df['Shares'] > 0
        
        # 计算买入持有策略
        initial_shares = self.initial_capital / df.iloc[0]['Avg_Price']
        df['Buy_Hold_Value'] = initial_shares * df['Avg_Price']
        
        return df, trades
    
    def calculate_metrics(self, df, trades, index_name):
        """计算性能指标"""
        # 基本收益计算
        final_value = df.iloc[-1]['Portfolio_Value']
        final_buy_hold = df.iloc[-1]['Buy_Hold_Value']
        
        total_return = (final_value - self.initial_capital) / self.initial_capital
        buy_hold_return = (final_buy_hold - self.initial_capital) / self.initial_capital
        
        # 计算年化收益率
        start_date = df.iloc[0]['Date']
        end_date = df.iloc[-1]['Date']
        years = (end_date - start_date).days / 365.25
        
        annual_return = (final_value / self.initial_capital) ** (1/years) - 1
        buy_hold_annual = (final_buy_hold / self.initial_capital) ** (1/years) - 1
        
        # 计算日收益率
        df['Daily_Return'] = df['Portfolio_Value'].pct_change()
        df['Buy_Hold_Daily_Return'] = df['Buy_Hold_Value'].pct_change()
        
        # 计算夏普比率（假设无风险利率为2%）
        risk_free_rate = 0.02 / 252  # 日化无风险利率
        strategy_sharpe = (df['Daily_Return'].mean() - risk_free_rate) / df['Daily_Return'].std() * np.sqrt(252)
        buy_hold_sharpe = (df['Buy_Hold_Daily_Return'].mean() - risk_free_rate) / df['Buy_Hold_Daily_Return'].std() * np.sqrt(252)
        
        # 计算最大回撤
        df['Cumulative_Max'] = df['Portfolio_Value'].cummax()
        df['Drawdown'] = (df['Portfolio_Value'] - df['Cumulative_Max']) / df['Cumulative_Max']
        max_drawdown = df['Drawdown'].min()
        
        df['BH_Cumulative_Max'] = df['Buy_Hold_Value'].cummax()
        df['BH_Drawdown'] = (df['Buy_Hold_Value'] - df['BH_Cumulative_Max']) / df['BH_Cumulative_Max']
        buy_hold_max_drawdown = df['BH_Drawdown'].min()
        
        # 计算胜率
        if len(trades) > 1:
            winning_trades = 0
            for i in range(0, len(trades)-1, 2):  # 假设买卖成对
                if i+1 < len(trades):
                    buy_price = trades[i]['Price']
                    sell_price = trades[i+1]['Price']
                    if sell_price > buy_price:
                        winning_trades += 1
            win_rate = winning_trades / (len(trades) // 2) if len(trades) >= 2 else 0
        else:
            win_rate = 0
        
        # 输出结果
        print(f"\n{'='*60}")
        print(f"{index_name} 回测结果")
        print(f"{'='*60}")
        print(f"回测期间: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')} ({years:.1f}年)")
        print(f"\n初始资金: ${self.initial_capital:.2f}")
        print(f"最终资金: ${final_value:.2f}")
        print(f"买入持有最终价值: ${final_buy_hold:.2f}")
        
        print(f"\n收益率对比:")
        print(f"策略总收益率: {total_return:.2%}")
        print(f"买入持有收益率: {buy_hold_return:.2%}")
        print(f"超额收益: {(total_return - buy_hold_return):.2%}")
        
        print(f"\n年化收益率:")
        print(f"策略年化收益率: {annual_return:.2%}")
        print(f"买入持有年化收益率: {buy_hold_annual:.2%}")
        print(f"年化超额收益: {(annual_return - buy_hold_annual):.2%}")
        
        print(f"\n风险指标:")
        print(f"策略夏普比率: {strategy_sharpe:.2f}")
        print(f"买入持有夏普比率: {buy_hold_sharpe:.2f}")
        print(f"策略最大回撤: {max_drawdown:.2%}")
        print(f"买入持有最大回撤: {buy_hold_max_drawdown:.2%}")
        
        print(f"\n交易统计:")
        print(f"总交易次数: {len(trades)}")
        print(f"买入次数: {len([t for t in trades if t['Type'] == 'BUY'])}")
        print(f"卖出次数: {len([t for t in trades if t['Type'] == 'SELL'])}")
        print(f"胜率: {win_rate:.2%}")
        
        # 显示前几笔交易
        if trades:
            print(f"\n前5笔交易:")
            for i, trade in enumerate(trades[:5]):
                print(f"{i+1}. {trade['Date'].strftime('%Y-%m-%d')} {trade['Type']} "
                      f"价格: ${trade['Price']:.2f} 股数: {trade['Shares']} "
                      f"现金余额: ${trade['Cash_After']:.2f}")
        
        return {
            'index': index_name,
            'total_return': total_return,
            'annual_return': annual_return,
            'buy_hold_return': buy_hold_return,
            'buy_hold_annual': buy_hold_annual,
            'sharpe_ratio': strategy_sharpe,
            'max_drawdown': max_drawdown,
            'trades': len(trades),
            'win_rate': win_rate
        }
    
    def plot_results(self, results_dict, save_plots=True):
        """绘制所有指数的结果对比图"""
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Trading Strategy Performance Comparison', fontsize=16)
        
        indices = list(results_dict.keys())
        colors = ['blue', 'green', 'red']
        
        # 1. 组合价值对比
        ax1 = axes[0, 0]
        for idx, (index_name, data) in enumerate(results_dict.items()):
            df = data['df']
            ax1.plot(df['Date'], df['Portfolio_Value'], label=f'{index_name} Strategy', 
                    color=colors[idx], linewidth=2)
            ax1.plot(df['Date'], df['Buy_Hold_Value'], label=f'{index_name} Buy&Hold', 
                    color=colors[idx], linestyle='--', alpha=0.7)
        
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 年化收益率对比
        ax2 = axes[0, 1]
        x = np.arange(len(indices))
        width = 0.35
        
        strategy_returns = [data['metrics']['annual_return'] for data in results_dict.values()]
        buyhold_returns = [data['metrics']['buy_hold_annual'] for data in results_dict.values()]
        
        ax2.bar(x - width/2, strategy_returns, width, label='Strategy', color='steelblue')
        ax2.bar(x + width/2, buyhold_returns, width, label='Buy & Hold', color='lightcoral')
        
        ax2.set_title('Annual Returns Comparison')
        ax2.set_ylabel('Annual Return (%)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(indices)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 格式化为百分比
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        # 3. 夏普比率和最大回撤
        ax3 = axes[1, 0]
        sharpe_ratios = [data['metrics']['sharpe_ratio'] for data in results_dict.values()]
        ax3.bar(indices, sharpe_ratios, color='darkgreen')
        ax3.set_title('Sharpe Ratio by Index')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.grid(True, alpha=0.3)
        
        # 4. 最大回撤
        ax4 = axes[1, 1]
        max_drawdowns = [-data['metrics']['max_drawdown'] for data in results_dict.values()]  # 转为正数显示
        ax4.bar(indices, max_drawdowns, color='darkred')
        ax4.set_title('Maximum Drawdown by Index')
        ax4.set_ylabel('Max Drawdown (%)')
        ax4.grid(True, alpha=0.3)
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('trading_strategy_comparison.png', dpi=300, bbox_inches='tight')
            print("\n图表已保存为 trading_strategy_comparison.png")
        
        plt.show()
    
    def generate_summary_report(self, results_dict):
        """生成汇总报告"""
        summary_data = []
        
        for index_name, data in results_dict.items():
            metrics = data['metrics']
            summary_data.append({
                'Index': index_name,
                'Strategy Annual Return': f"{metrics['annual_return']:.2%}",
                'Buy&Hold Annual Return': f"{metrics['buy_hold_annual']:.2%}",
                'Excess Return': f"{(metrics['annual_return'] - metrics['buy_hold_annual']):.2%}",
                'Sharpe Ratio': f"{metrics['sharpe_ratio']:.2f}",
                'Max Drawdown': f"{metrics['max_drawdown']:.2%}",
                'Total Trades': metrics['trades'],
                'Win Rate': f"{metrics['win_rate']:.2%}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # 保存汇总报告
        summary_df.to_csv('trading_strategy_summary.csv', index=False)
        print("\n汇总报告已保存为 trading_strategy_summary.csv")
        
        # 打印汇总表
        print("\n" + "="*100)
        print("策略汇总报告")
        print("="*100)
        print(summary_df.to_string(index=False))

def main():
    """主函数"""
    print("交易策略回测系统")
    print("="*50)
    
    # 创建回测实例
    backtest = TradingBacktest(initial_capital=100000.0, commission_rate=0)
    
    # 定义要回测的指数
    signal_type = "balanced"
    indices = [
        ('dow_jones_data.csv', f'dow_jones_{signal_type}_signals.csv', 'Dow Jones'),
        ('nasdaq_data.csv', f'nasdaq_{signal_type}_signals.csv', 'NASDAQ'),
        ('sp500_data.csv', f'sp500_{signal_type}_signals.csv', 'S&P 500')
    ]
    
    # 存储所有结果
    all_results = {}
    
    # 对每个指数进行回测
    for price_file, signal_file, index_name in indices:
        try:
            print(f"\n正在回测 {index_name}...")
            
            # 加载数据
            df = backtest.load_data(price_file, signal_file)
            
            # 模拟交易
            df, trades = backtest.simulate_trading(df, index_name)
            
            # 计算指标
            metrics = backtest.calculate_metrics(df, trades, index_name)
            
            # 保存结果
            all_results[index_name] = {
                'df': df,
                'trades': trades,
                'metrics': metrics
            }
            
        except Exception as e:
            print(f"处理 {index_name} 时出错: {e}")
            continue
    
    # 生成对比图表和汇总报告
    if all_results:
        print("\n生成对比图表和汇总报告...")
        backtest.plot_results(all_results)
        backtest.generate_summary_report(all_results)
    
    print("\n回测完成！")

if __name__ == "__main__":
    main()