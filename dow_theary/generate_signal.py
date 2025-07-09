import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class BalancedDowTheoryTrader:
    def __init__(self, trend_window=20, confirmation_window=5, volume_multiplier=1.2):
        """
        初始化平衡的道氏理论交易策略
        
        参数:
        trend_window: 趋势识别窗口（默认20天）
        confirmation_window: 确认窗口（默认5天）
        volume_multiplier: 成交量放大倍数（默认1.2倍）
        """
        self.trend_window = trend_window
        self.confirmation_window = confirmation_window
        self.volume_multiplier = volume_multiplier
        self.position = 0  # 0=空仓, 1=持仓
        
    def load_data(self, filename):
        """加载CSV数据"""
        try:
            df = pd.read_csv(filename)
            
            # 清理列名
            df.columns = [col.strip() for col in df.columns]
            if 'Date' in df.columns and len(df.columns) > 1 and df.columns[1] == '':
                df = df.drop(df.columns[1], axis=1)
            
            # 处理日期
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            
            print(f"成功加载 {filename}")
            print(f"数据范围: {df['Date'].min()} 到 {df['Date'].max()}")
            print(f"数据点数: {len(df)}")
            
            return df
        except Exception as e:
            print(f"加载 {filename} 时出错: {e}")
            return None
    
    def calculate_indicators(self, df):
        """计算技术指标"""
        # 1. 移动平均线
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # 2. 价格位置
        df['Price_vs_SMA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20'] * 100
        
        # 3. 成交量分析
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # 4. 价格动量
        df['ROC_5'] = df['Close'].pct_change(5) * 100
        df['ROC_20'] = df['Close'].pct_change(20) * 100
        
        # 5. 高低点
        df['High_20'] = df['High'].rolling(window=20).max()
        df['Low_20'] = df['Low'].rolling(window=20).min()
        df['High_50'] = df['High'].rolling(window=50).max()
        df['Low_50'] = df['Low'].rolling(window=50).min()
        
        # 6. 相对强度
        df['RS'] = df['Close'] / df['Close'].rolling(window=20).mean()
        
        # 7. 波动性
        df['Volatility'] = df['Close'].pct_change().rolling(window=20).std() * 100
        
        return df
    
    def identify_trend(self, df):
        """识别市场趋势"""
        df['Primary_Trend'] = 'NEUTRAL'
        
        # 主要趋势判断条件
        for i in range(50, len(df)):
            # 上升趋势条件
            if (df.iloc[i]['Close'] > df.iloc[i]['SMA_20'] and
                df.iloc[i]['SMA_20'] > df.iloc[i]['SMA_50'] and
                df.iloc[i]['Close'] > df.iloc[i-20]['Close'] and
                df.iloc[i]['Low_20'] > df.iloc[i-20]['Low_20']):
                df.loc[i, 'Primary_Trend'] = 'UP'
            
            # 下降趋势条件
            elif (df.iloc[i]['Close'] < df.iloc[i]['SMA_20'] and
                  df.iloc[i]['SMA_20'] < df.iloc[i]['SMA_50'] and
                  df.iloc[i]['Close'] < df.iloc[i-20]['Close'] and
                  df.iloc[i]['High_20'] < df.iloc[i-20]['High_20']):
                df.loc[i, 'Primary_Trend'] = 'DOWN'
        
        # 次级反应
        df['Secondary_Move'] = 0
        for i in range(5, len(df)):
            if df.iloc[i]['ROC_5'] > 3:
                df.loc[i, 'Secondary_Move'] = 1
            elif df.iloc[i]['ROC_5'] < -3:
                df.loc[i, 'Secondary_Move'] = -1
        
        return df
    
    def generate_signals(self, df):
        """生成交易信号"""
        df['Signal'] = 'HOLD'
        df['Signal_Strength'] = 0
        
        position = 0  # 跟踪仓位状态
        last_signal_idx = 0
        
        for i in range(50, len(df)):
            # 跳过距离上次信号太近的点
            if i - last_signal_idx < 10:
                continue
            
            # 买入信号评分系统
            buy_score = 0
            sell_score = 0
            
            # 1. 趋势条件（权重最高）
            if df.iloc[i]['Primary_Trend'] == 'UP':
                buy_score += 3
            elif df.iloc[i]['Primary_Trend'] == 'DOWN':
                sell_score += 3
            
            # 2. 价格突破
            if df.iloc[i]['Close'] > df.iloc[i]['High_20']:
                buy_score += 2
            elif df.iloc[i]['Close'] < df.iloc[i]['Low_20']:
                sell_score += 2
            
            # 3. 成交量确认
            if df.iloc[i]['Volume_Ratio'] > self.volume_multiplier:
                if df.iloc[i]['Close'] > df.iloc[i-1]['Close']:
                    buy_score += 2
                else:
                    sell_score += 2
            
            # 4. 动量
            if df.iloc[i]['ROC_5'] > 2:
                buy_score += 1
            elif df.iloc[i]['ROC_5'] < -2:
                sell_score += 1
            
            # 5. 相对位置
            if df.iloc[i]['Price_vs_SMA20'] > 2:
                buy_score += 1
            elif df.iloc[i]['Price_vs_SMA20'] < -2:
                sell_score += 1
            
            # 6. 趋势转换
            if (i > 0 and 
                df.iloc[i]['Primary_Trend'] == 'UP' and 
                df.iloc[i-1]['Primary_Trend'] != 'UP'):
                buy_score += 2
            elif (i > 0 and 
                  df.iloc[i]['Primary_Trend'] == 'DOWN' and 
                  df.iloc[i-1]['Primary_Trend'] != 'DOWN'):
                sell_score += 2
            
            # 信号决策
            df.loc[i, 'Signal_Strength'] = buy_score - sell_score
            
            # 买入条件（更宽松）
            if position == 0 and buy_score >= 5:
                df.loc[i, 'Signal'] = 'BUY'
                position = 1
                last_signal_idx = i
            
            # 卖出条件
            elif position == 1 and (sell_score >= 4 or 
                                   # 止损条件
                                   (i > 20 and df.iloc[i]['Close'] < df.iloc[i-20]['Close'] * 0.92)):
                df.loc[i, 'Signal'] = 'SELL'
                position = 0
                last_signal_idx = i
        
        return df
    
    def confirm_with_indices(self, dow_df, nasdaq_df, sp500_df):
        """使用道氏理论的指数相互确认原则"""
        # 提取信号日期
        signal_dates = set()
        
        for df in [dow_df, nasdaq_df, sp500_df]:
            if df is not None:
                signals = df[df['Signal'].isin(['BUY', 'SELL'])]
                signal_dates.update(signals['Date'].tolist())
        
        # 对每个信号日期进行确认
        confirmed_signals = {}
        
        for date in sorted(signal_dates):
            buy_count = 0
            sell_count = 0
            
            for df in [dow_df, nasdaq_df, sp500_df]:
                if df is not None:
                    # 查找该日期附近的信号（容差3天）
                    mask = (df['Date'] >= date - pd.Timedelta(days=3)) & \
                           (df['Date'] <= date + pd.Timedelta(days=3))
                    nearby_signals = df[mask & df['Signal'].isin(['BUY', 'SELL'])]
                    
                    if not nearby_signals.empty:
                        signal = nearby_signals.iloc[0]['Signal']
                        if signal == 'BUY':
                            buy_count += 1
                        elif signal == 'SELL':
                            sell_count += 1
            
            # 确认规则：至少2个指数同向
            if buy_count >= 2:
                confirmed_signals[date] = 'BUY'
            elif sell_count >= 2:
                confirmed_signals[date] = 'SELL'
        
        # 应用确认的信号
        for df in [dow_df, nasdaq_df, sp500_df]:
            if df is not None:
                df['Confirmed_Signal'] = 'HOLD'
                for date, signal in confirmed_signals.items():
                    mask = (df['Date'] >= date - pd.Timedelta(days=1)) & \
                           (df['Date'] <= date + pd.Timedelta(days=1))
                    if mask.any():
                        idx = df[mask].index[0]
                        df.loc[idx, 'Confirmed_Signal'] = signal
        
        return dow_df, nasdaq_df, sp500_df
    
    def apply_position_management(self, df):
        """应用仓位管理和风险控制"""
        if df is None:
            return df
        
        # 使用确认信号
        if 'Confirmed_Signal' in df.columns:
            signals = df['Confirmed_Signal'].copy()
        else:
            signals = df['Signal'].copy()
        
        # 清理连续相同信号
        position = 0
        for i in range(len(df)):
            current_signal = signals.iloc[i]
            
            if current_signal == 'BUY' and position == 0:
                position = 1
            elif current_signal == 'SELL' and position == 1:
                position = 0
            elif current_signal == 'BUY' and position == 1:
                signals.iloc[i] = 'HOLD'  # 已经持仓，忽略买入
            elif current_signal == 'SELL' and position == 0:
                signals.iloc[i] = 'HOLD'  # 已经空仓，忽略卖出
        
        df['Final_Signal'] = signals
        
        return df
    
    def save_signals(self, df, index_name):
        """保存交易信号"""
        if df is None:
            return
        
        # 准备输出数据
        output_df = df[['Date']].copy()
        
        # 使用最终信号
        if 'Final_Signal' in df.columns:
            output_df['Signal'] = df['Final_Signal']
        elif 'Confirmed_Signal' in df.columns:
            output_df['Signal'] = df['Confirmed_Signal']
        else:
            output_df['Signal'] = df['Signal']
        
        # 只保留交易信号
        output_df = output_df[output_df['Signal'].isin(['BUY', 'SELL'])]
        
        # 保存
        filename = f'{index_name}_balanced_signals.csv'
        output_df.to_csv(filename, index=False)
        
        print(f"\n{index_name} 交易信号已保存到 {filename}")
        print(f"买入信号: {(output_df['Signal'] == 'BUY').sum()}")
        print(f"卖出信号: {(output_df['Signal'] == 'SELL').sum()}")
        
        # 显示前10个信号
        print(f"\n前10个信号:")
        print(output_df.head(10))
    
    def analyze_performance(self, df, index_name):
        """分析策略表现"""
        if df is None:
            return
        
        # 获取信号
        if 'Final_Signal' in df.columns:
            signal_col = 'Final_Signal'
        elif 'Confirmed_Signal' in df.columns:
            signal_col = 'Confirmed_Signal'
        else:
            signal_col = 'Signal'
        
        # 计算收益
        trades = []
        position = 0
        entry_price = 0
        entry_date = None
        
        for i in range(len(df)):
            if df.iloc[i][signal_col] == 'BUY' and position == 0:
                position = 1
                entry_price = df.iloc[i]['Close']
                entry_date = df.iloc[i]['Date']
            elif df.iloc[i][signal_col] == 'SELL' and position == 1:
                exit_price = df.iloc[i]['Close']
                exit_date = df.iloc[i]['Date']
                returns = (exit_price / entry_price - 1) * 100
                holding_days = (exit_date - entry_date).days
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'returns': returns,
                    'holding_days': holding_days
                })
                position = 0
        
        if trades:
            returns_list = [t['returns'] for t in trades]
            wins = [r for r in returns_list if r > 0]
            losses = [r for r in returns_list if r <= 0]
            
            print(f"\n{index_name} 策略表现分析:")
            print(f"总交易次数: {len(trades)}")
            print(f"胜率: {len(wins)/len(trades)*100:.1f}%")
            print(f"平均收益率: {np.mean(returns_list):.2f}%")
            
            if wins:
                print(f"平均盈利: {np.mean(wins):.2f}%")
                print(f"最大盈利: {max(wins):.2f}%")
            if losses:
                print(f"平均亏损: {np.mean(losses):.2f}%")
                print(f"最大亏损: {min(losses):.2f}%")
            
            print(f"平均持仓天数: {np.mean([t['holding_days'] for t in trades]):.0f}")
            
            # 买入持有策略对比
            total_return = (df.iloc[-1]['Close'] / df.iloc[50]['Close'] - 1) * 100
            print(f"\n买入持有收益率: {total_return:.2f}%")
            
            # 策略累计收益
            strategy_return = 1
            for r in returns_list:
                strategy_return *= (1 + r/100)
            strategy_return = (strategy_return - 1) * 100
            print(f"策略累计收益率: {strategy_return:.2f}%")

def main():
    """主函数"""
    print("平衡的道氏理论交易策略")
    print("="*60)
    print("策略特点：")
    print("1. 使用评分系统生成平衡的买卖信号")
    print("2. 多指标确认，避免频繁交易")
    print("3. 遵循道氏理论的指数相互确认原则")
    print("="*60)
    
    # 创建策略实例
    trader = BalancedDowTheoryTrader(
        trend_window=20,
        confirmation_window=5,
        volume_multiplier=1.2
    )
    
    # 加载数据
    print("\n正在加载数据...")
    dow_df = trader.load_data('dow_jones_data.csv')
    nasdaq_df = trader.load_data('nasdaq_data.csv')
    sp500_df = trader.load_data('sp500_data.csv')
    
    # 计算指标
    print("\n计算技术指标...")
    if dow_df is not None:
        dow_df = trader.calculate_indicators(dow_df)
        dow_df = trader.identify_trend(dow_df)
        dow_df = trader.generate_signals(dow_df)
    
    if nasdaq_df is not None:
        nasdaq_df = trader.calculate_indicators(nasdaq_df)
        nasdaq_df = trader.identify_trend(nasdaq_df)
        nasdaq_df = trader.generate_signals(nasdaq_df)
    
    if sp500_df is not None:
        sp500_df = trader.calculate_indicators(sp500_df)
        sp500_df = trader.identify_trend(sp500_df)
        sp500_df = trader.generate_signals(sp500_df)
    
    # 应用道氏理论确认
    print("\n应用道氏理论指数确认...")
    dow_df, nasdaq_df, sp500_df = trader.confirm_with_indices(dow_df, nasdaq_df, sp500_df)
    
    # 仓位管理
    dow_df = trader.apply_position_management(dow_df)
    nasdaq_df = trader.apply_position_management(nasdaq_df)
    sp500_df = trader.apply_position_management(sp500_df)
    
    # 保存信号
    trader.save_signals(dow_df, 'dow_jones')
    trader.save_signals(nasdaq_df, 'nasdaq')
    trader.save_signals(sp500_df, 'sp500')
    
    # 分析表现
    trader.analyze_performance(dow_df, '道琼斯')
    trader.analyze_performance(nasdaq_df, '纳斯达克')
    trader.analyze_performance(sp500_df, '标普500')
    
    print("\n" + "="*60)
    print("策略执行完成！")
    print("注意：这是一个更平衡的策略，买卖信号应该更加合理。")
    print("="*60)

if __name__ == "__main__":
    main()