import sqlite3
from neo4j import GraphDatabase

# 1. 从SQLite读取关注关系数据
def read_follow_data():
    conn = sqlite3.connect('./data/simu_db/yaml_200/time0.db')
    cursor = conn.cursor()
    cursor.execute("SELECT follower_id, followee_id FROM follow")
    follow_data = cursor.fetchall()
    conn.close()
    return follow_data

# 2. Neo4j数据库连接配置
uri = "bolt://localhost:7687"
user = "neo4j"
password = "neo4j"
driver = GraphDatabase.driver(uri, auth=(user, password))

# 3. 创建节点和关系的函数
def create_graph(tx, follow_data):
    # 收集所有节点ID
    all_ids = set()
    for row in follow_data:
        all_ids.add(row[0])
        all_ids.add(row[1])

    # 分割好人坏人ID
    good_ids = [id for id in range(0, 900)]
    bad_ids = [id for id in range(900,1000)]

    # 批量创建好人节点
    if good_ids:
        tx.run("""
        UNWIND $ids AS id
        MERGE (:Good:Agent {id: id})
        """, ids=good_ids)
    
    # 批量创建坏人节点
    if bad_ids:
        tx.run("""
        UNWIND $ids AS id
        MERGE (:Bad:Agent {id: id})
        """, ids=bad_ids)

    # 批量创建关注关系
    tx.run("""
    UNWIND $rows AS row
    MATCH (follower:Agent {id: row.follower_id})
    MATCH (followee:Agent {id: row.followee_id})
    MERGE (follower)-[:FOLLOWS]->(followee)
    """, rows=[{"follower_id": row[0], "followee_id": row[1]} for row in follow_data])

# 4. 执行数据导入
def import_data():
    follow_data = read_follow_data()
    with driver.session() as session:
        session.execute_write(create_graph, follow_data)
    print("数据导入完成！")

# 5. 查询好人关注坏人的子图
def query_good_to_bad():
    with driver.session() as session:
        result = session.run("""
        MATCH (follower:Good)-[r:FOLLOWS]->(followee:Bad)
        RETURN follower.id AS follower_id, 
               followee.id AS followee_id,
               count(r) AS relationship_count
        """)
        
        print("\n好人关注坏人关系统计：")
        for record in result:
            print(f"好人 {record['follower_id']} → 坏人 {record['followee_id']}")

# 主程序
if __name__ == "__main__":
    import_data()
    query_good_to_bad()
    driver.close()
