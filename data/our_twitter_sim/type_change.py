import pandas as pd

def replace_user_type_in_csv(input_file, output_file):
    # 读取CSV文件
    df = pd.read_csv(input_file)
    
    # 检查是否存在user_type列
    if 'user_type' not in df.columns:
        print("错误：CSV文件中不存在'user_type'列")
        return False
    
    # 替换user_type为'good'的值为'good_prebunking'
    df.loc[df['user_type'] == 'good', 'user_type'] = 'good_prebunking'
    
    # 保存修改后的数据到新的CSV文件
    df.to_csv(output_file, index=False)
    print(f"替换完成！结果已保存到 {output_file}")
    return True

# 使用示例
if __name__ == "__main__":
    input_file = "./test_1000_good_bad_member_random_bernoulli_xst.csv"  # 替换为你的输入文件名
    output_file = "./test_1000_good_bad_member_random_bernoulli_wlx.csv"  # 替换为你想要的输出文件名
    
    replace_user_type_in_csv(input_file, output_file)
