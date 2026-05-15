import os
import re
import glob
import argparse
import sys

def extract_table_data(trace_file_path):
    """
    核心逻辑：解析单个 trace.md，只提取纯数据行
    """
    if not os.path.exists(trace_file_path):
        return []

    try:
        with open(trace_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        # 如果文件不存在或读不到，直接返回空列表，不报错
        return []

    # 1. 定位表格区域
    # 只要找到这个标题，我们就认为下面跟着表格
    start_idx = content.find('## 汇总表报告')
    if start_idx == -1:
        return [] # 没找到标题，直接跳过，不产生空行

    # 截取标题之后的内容
    section_content = content[start_idx:]
    lines = section_content.split('\n')
    
    valid_rows = []
    
    # 2. 逐行扫描
    for line in lines:
        line = line.strip()
        
        # 核心过滤逻辑：
        # A. 必须是表格行 (以 | 开头)
        # B. 不能是表头 (不包含 Level)
        # C. 不能是分隔线 (不包含 ---)  <-- 这一步是去除空行的关键！
        # D. 不能是空行 (防止提取到纯回车)
        if line.startswith('|') and 'Level' not in line and '---' not in line and len(line) > 5:
            valid_rows.append(line)
            
    return valid_rows

def main():
    # --- 1. 配置命令行参数 ---
    parser = argparse.ArgumentParser(
        description="🚀 算子性能汇总工具 - 命令行版",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python generate_report_dynamic.py -i /path/to/output_0410_l2_ziji
  python generate_report_dynamic.py -i ./my_ops -o ./result/batch_report.md
        """
    )
    
    # 输入路径参数
    parser.add_argument(
        '-i', '--input', 
        type=str, 
        required=True, 
        help='【必填】算子输出根目录路径 (例如: /home/user/output_0410_l2_ziji)'
    )
    
    # 输出文件路径参数
    parser.add_argument(
        '-o', '--output', 
        type=str, 
        default='batch_report.md', 
        help='【可选】汇总报告保存路径 (默认: batch_report.md)'
    )

    args = parser.parse_args()
    
    base_dir = args.input
    output_path = args.output

    # --- 2. 校验输入路径 ---
    if not os.path.exists(base_dir):
        print(f"❌ 错误：源目录不存在 -> {base_dir}")
        sys.exit(1)
    
    if not os.path.isdir(base_dir):
        print(f"❌ 错误：输入的路径不是目录 -> {base_dir}")
        sys.exit(1)

    # --- 3. 动态搜索文件 ---
    # 匹配 base_dir 下任意一级子目录中的 trace.md
    search_pattern = os.path.join(base_dir, '*', 'trace.md')
    trace_files = glob.glob(search_pattern)
    
    # 排序以保证输出顺序一致
    trace_files.sort()

    if not trace_files:
        print(f"⚠️ 警告：在 {base_dir} 下未找到任何算子 trace.md 文件")
        print("   请检查目录结构是否包含 '子目录/trace.md'")
        sys.exit(0)

    print(f"📂 正在扫描目录: {base_dir}")
    print(f"🔍 找到 {len(trace_files)} 个算子报告，正在处理...")

    # --- 4. 收集数据 ---
    all_rows = []
    for file in trace_files:
        rows = extract_table_data(file)
        if rows:
            all_rows.extend(rows)
            # 打印简单的进度点
            print(".", end="", flush=True)
    
    print("\n") # 换行

    # --- 5. 写入汇总文件 ---
    # 确保输出文件的目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir: # 如果输出路径包含目录
        os.makedirs(output_dir, exist_ok=True)

    # 定义表头
    header = [
        "| Level | Problem ID | 算子名称 | 算子类型 | 编译通过 | 精度正确 | PyTorch 参考延迟 | 生成AscendC代码延迟 | 加速比 | 最终状态 | 精度正确 | 性能0.6x pytorch | 性能0.8x pytorch |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    ]

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("## 📊 算子批量测试汇总报告\n\n")
            f.write("\n".join(header) + "\n")
            f.write("\n".join(all_rows) + "\n")
        
        print(f"✅ 处理完成！")
        print(f"📄 报告已保存至: {os.path.abspath(output_path)}")
    except Exception as e:
        print(f"❌ 写入文件失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()