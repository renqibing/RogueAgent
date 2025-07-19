import sqlite3

# 连接到数据库
conn = sqlite3.connect(r"C:\Users\Einsamkeit\Downloads\test_1000_good_bad_random_bernoulli_wlx_safety.db")
cursor = conn.cursor()

# 执行统计语句
cursor.execute('SELECT COUNT(*) FROM post WHERE original_post_id IS NULL AND user_id > 900')
count = cursor.fetchone()[0]

print(f'由坏人创建的original_post_id 为 NULL 的 post 数量为: {count}')

# 关闭连接
conn.close()
