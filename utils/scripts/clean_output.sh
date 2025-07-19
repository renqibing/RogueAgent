#!/bin/bash

# 删除指定目录下的所有文件（不包含子文件夹）
delete_files() {
    target_dirs=("$@")
    files_to_delete=()

    # 遍历目标目录列表
    for dir_path in "${target_dirs[@]}"; do
        if [ ! -d "$dir_path" ]; then
            echo "⚠️ 目录 $dir_path 不存在，跳过"
            continue
        fi

        # 查找当前目录下的所有文件，不包括以点开头的文件
        while IFS= read -r file; do
            if [ -f "$file" ] && [[ ! "$(basename "$file")" =~ ^\..* ]]; then
                files_to_delete+=("$file")
            fi
        done < <(find "$dir_path" -maxdepth 1 -type f)
    done

    if [ ${#files_to_delete[@]} -eq 0 ]; then
        echo "🎉 没有找到需要删除的文件"
        exit 0
    fi

    # 显示待删除的文件
    echo -e "\n🚨 以下文件将被删除（不包含子文件夹内容）："
    for idx in "${!files_to_delete[@]}"; do
        echo "[$(($idx + 1))] ${files_to_delete[$idx]}"
    done

    # 确认删除
    read -p $'\n🔥 确认删除以上 '"${#files_to_delete[@]}"' 个文件？(y/n): ' confirm
    if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
        success=0
        fail=0

        for fpath in "${files_to_delete[@]}"; do
            if rm "$fpath"; then
                echo "✅ 已删除：$fpath"
                ((success++))
            else
                echo "❌ 删除失败：$fpath"
                ((fail++))
            fi
        done

        echo -e "\n📊 操作完成：成功 $success 个，失败 $fail 个"
    else
        echo "🛑 操作已取消"
    fi
}

# 主程序
target_dirs=("results" "log" "results/histogram")
delete_files "${target_dirs[@]}"
