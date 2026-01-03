import json
from collections import defaultdict
import os

def split_jsonl_by_category(input_filepath, output_dir="."):
    """
    根据categories字段将JSONL文件分成不同类别
    
    Args:
        input_filepath: 输入JSONL文件路径
        output_dir: 输出目录，默认为当前目录
    """
    # 存储不同类别的数据
    category_data = defaultdict(list)
    
    # 统计信息
    total_records = 0
    records_with_categories = 0
    category_counts = defaultdict(int)
    
    try:
        # 读取JSONL文件
        with open(input_filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    total_records += 1
                    
                    # 检查是否有categories字段
                    if 'categories' not in data or not data['categories']:
                        print(f"警告: 第{line_num}行没有categories字段或为空")
                        continue
                    
                    records_with_categories += 1
                    categories = data['categories']
                    
                    # 处理categories (可能是列表)
                    if isinstance(categories, list):
                        # 检查是否包含目标类别
                        for category in categories:
                            if category in ['Crypto', 'Election', 'Politics']:
                                category_data[category].append(data)
                                category_counts[category] += 1
                                break  # 只归类到第一个匹配的类别
                        else:
                            # 如果不在三个目标类别中
                            print(f"警告: 第{line_num}行的categories {categories} 不在目标类别中")
                    else:
                        print(f"警告: 第{line_num}行的categories不是列表: {categories}")
                
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误在第{line_num}行: {e}")
                    continue
        
        # 从输入文件名提取基础名称
        input_basename = os.path.basename(input_filepath)
        # 假设文件名格式为 polymarket_data_processed_with_news_test_2024-11-01.jsonl
        # 提取日期部分
        if input_basename.endswith('.jsonl'):
            base_name = input_basename[:-6]  # 去掉.jsonl
        else:
            base_name = input_basename
        
        # 提取日期 (假设格式为 ..._YYYY-MM-DD)
        parts = base_name.split('_')
        date_part = parts[-1] if len(parts) > 0 else "2024-11-01"
        
        # 写入分类后的文件
        output_files = {}
        for category, records in category_data.items():
            output_filename = f"polymarket_data_processed_with_news_test_{category}_{date_part}.jsonl"
            output_filepath = os.path.join(output_dir, output_filename)
            
            with open(output_filepath, 'w', encoding='utf-8') as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            output_files[category] = output_filepath
            print(f"✓ 已创建 {category} 文件: {output_filename} ({len(records)} 条记录)")
        
        # 打印统计信息
        print("\n" + "="*60)
        print("统计摘要:")
        print("="*60)
        print(f"总记录数: {total_records}")
        print(f"有categories字段的记录数: {records_with_categories}")
        print(f"\n各类别分布:")
        for category in ['Crypto', 'Election', 'Politics']:
            count = category_counts.get(category, 0)
            percentage = (count / total_records * 100) if total_records > 0 else 0
            print(f"  {category}: {count} ({percentage:.2f}%)")
        print("="*60)
        
        return output_files
    
    except FileNotFoundError:
        print(f"错误: 文件未找到 - {input_filepath}")
        return None
    except Exception as e:
        print(f"错误: {e}")
        return None

# 使用示例
if __name__ == "__main__":
    # 输入文件路径
    input_file = "/data/haofeiy2/social-world-model/data/splitted_polymarket/polymarket_data_processed_with_news_test_2024-11-01.jsonl"
    
    # 输出目录 (可选，默认为当前目录)
    output_directory = "."
    
    print(f"正在处理文件: {input_file}")
    print("-"*60)
    
    output_files = split_jsonl_by_category(input_file, output_directory)
    
    if output_files:
        print("\n处理完成!")