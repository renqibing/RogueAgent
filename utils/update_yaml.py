from ruamel.yaml import YAML
import os
import argparse

def deep_merge(target, source):
    """
    递归合并source字典到target字典中。
    对于存在的键，若值为字典则递归合并；否则覆盖目标值。
    """
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            deep_merge(target[key], value)
        else:
            if value:
                target[key] = value
    return target

def update_yaml_files(base_config_path, target_dir):
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)

    # 加载公共配置
    with open(base_config_path, 'r', encoding='utf-8') as f:
        base_config = yaml.load(f)
    # 遍历目标目录中的YAML文件
    for filename in os.listdir(target_dir):
        if filename.lower().endswith(('.yaml', '.yml')):
            filepath = os.path.join(target_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                original_config = yaml.load(f)
            
            # 合并配置
            merged_config = deep_merge(original_config, base_config)
            
            # 写回文件
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(merged_config, f)
            print(f'已更新文件: {filepath}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='使用公共YAML文件批量更新目标文件夹中的配置文件')
    parser.add_argument('--base', '-b', required=True, help='公共配置文件的路径')
    parser.add_argument('--target-dir', '-td', required=True, help='需要更新的YAML文件所在目录')
    args = parser.parse_args()
    
    update_yaml_files(args.base, args.target_dir)
